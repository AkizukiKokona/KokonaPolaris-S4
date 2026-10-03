"""G0 · NVFP4 (block-scaled) 吞吐**严格复测** —— 解决 archive 里 238.8 的存疑。

背景：`tools/bench_115w.py` 一次测出 FP4 4096³=238.8 / 8192³=150.9 TFLOPS，
两档差 1.58×。这个矛盾**没有被解释**，而 FP4 是项目的硬约束（NVFP4 原生），
所以「本机 FP4 到底多少」必须给一个可辩护的数字。

本脚本的设计要点（都是踩过坑才写对的）：
  ① **算力（纯 CUDA event）与功耗（另开线程采样）严格分离**
     —— 绝不要把 nvidia-smi 子进程放进 event 计时区间（会虚低 10×）。
  ② **每档充分预热**，否则冷启动数字毫无意义。
  ③ **每档同时报「时间 / 等效带宽 / 功耗」**：FP4 在 8GB + 384GB/s 的卡上
     很可能**不是算力瓶颈而是带宽/缓存瓶颈** ⇒ 只报 TFLOPS 会误导规划。
  ④ 明确标注每档的 **A+B+C 驻留量**，便于判断哪些档被缓存效应放大。
  ⑤ 采样 `clocks_event_reasons.active` 判定有无降频。
"""
from __future__ import annotations

import subprocess
import threading
import time

import torch

DEV = "cuda"
BLK = 16   # NVFP4 block size（与 kp/quant/nvfp4.py 一致）


def gpu_sample() -> dict:
    """单次采样（**只在计时区间之外调用**）。"""
    try:
        o = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=power.draw,clocks.sm,temperature.gpu,utilization.gpu,pstate,"
             "clocks_event_reasons.active",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip()
        p, clk, t, u, st, thr = [x.strip() for x in o.split(",")]
        return {"power": float(p), "clk": float(clk), "temp": float(t),
                "util": float(u), "pstate": st, "throttle": thr}
    except Exception as e:  # noqa: BLE001
        return {"power": float("nan"), "clk": float("nan"), "temp": float("nan"),
                "util": float("nan"), "pstate": "?", "throttle": f"err {e}"}


class PowerSampler(threading.Thread):
    """另开线程采样功耗 —— 与 event 计时**互不干扰**。"""

    def __init__(self, interval=0.5):
        super().__init__(daemon=True)
        self.interval, self.samples = interval, []
        # ⚠️ 不要命名为 `_stop`：`threading.Thread` **内部已有 `self._stop` 方法**，
        #    覆盖它会在 join() 时炸 `TypeError: 'bool' object is not callable`。
        self._halt = False

    def run(self):
        while not self._halt:
            self.samples.append(gpu_sample())
            time.sleep(self.interval)

    def stop(self):
        self._halt = True
        self.join(timeout=5)
        return self.samples


def fp4_mod(M, N, K, block=BLK):
    """NVFP4 block-scaled matmul（E2M1 packed + block16 E4M3 scale）。"""
    a = torch.randint(0, 256, (M, K // 2), device=DEV, dtype=torch.uint8).view(
        torch.float4_e2m1fn_x2)
    b = torch.randint(0, 256, (N, K // 2), device=DEV, dtype=torch.uint8).view(
        torch.float4_e2m1fn_x2)
    a_s = torch.ones(M, K // block, device=DEV, dtype=torch.float8_e4m3fn)
    b_s = torch.ones(N, K // block, device=DEV, dtype=torch.float8_e4m3fn)
    return torch._scaled_mm(a, b.t(), a_s, b_s, out_dtype=torch.bfloat16)


def bench_fp4(n, warmup_s=4.0, measure_s=3.0, sample_interval=0.25):
    """按**时间**跑（不是按迭代数）—— 因为功耗采样需要足够的采样窗口。

    ⚠️ 血泪教训：第一版按固定迭代数跑，每档只有 0.15–7 ms ⇒ **功耗采样器
      根本没采到运行期**，于是报出「257 TFLOPS @ 27.9 W」这种物理上不可能的搭配。
      功耗统计只有在「测量窗口 ≫ 采样间隔」时才有意义。
    """
    # 预热：按时间跑，确保 profile 稳定（而非按次数）
    t0 = time.time()
    while time.time() - t0 < warmup_s:
        fp4_mod(n, n, n)
    torch.cuda.synchronize()

    # 测量：纯 event 计时（**区间内绝不放 nvidia-smi 调用**）
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    iters, t0 = 0, time.time()
    while time.time() - t0 < measure_s:
        fp4_mod(n, n, n)
        iters += 1
    e.record()
    torch.cuda.synchronize()
    wall = time.time() - t0

    ms_total = s.elapsed_time(e)
    ms = ms_total / iters
    flops = 2 * n ** 3
    # FP4 操作数字节数：A(n·n/2) + B(n·n/2) + scale 各 n·n/16 + 输出 bf16(n·n·2)
    bytes_moved = (n * n / 2) * 2 + (n * n / 16) * 2 + n * n * 2
    return {
        "n": n, "iters": iters, "wall_s": wall, "ms_per_iter": ms,
        "tflops": flops / (ms / 1000) / 1e12,
        "eff_bw_gbps": bytes_moved / (ms / 1000) / 1e9,
        "resid_mb": 3 * n * n * 2 / 2 ** 20,
        "L2_multiple": (3 * n * n * 2) / (32 * 2 ** 20),
    }


def soak(seconds: float) -> list:
    """连续 8192³ 满载，把功耗/温度拉到稳态并返回采样。"""
    x = torch.randn(8192, 8192, device=DEV, dtype=torch.bfloat16)
    y = torch.randn(8192, 8192, device=DEV, dtype=torch.bfloat16)
    ps = PowerSampler(); ps.start()
    t0 = time.time()
    while time.time() - t0 < seconds:
        _ = x @ y
    torch.cuda.synchronize()
    smp = ps.stop()
    del x, y, _
    torch.cuda.empty_cache()
    return smp


def main() -> int:
    print("=" * 78)
    print("  KokonaPolaris-S4 · NVFP4 吞吐严格复测（【viim】RTX 5070 Laptop）")
    print("=" * 78)
    print(f"  torch {torch.__version__} | {torch.cuda.get_device_name(0)}")
    print(f"  block_size = {BLK}（对齐 kp/quant/nvfp4.py）")
    print()
    print("  口径：① 算力用纯 CUDA event 计时；功耗另开线程采样（严格分离）")
    print("        ② **按时间跑**（每档 ≥3s）—— 否则功耗采样采不到运行期")
    print("        ③ 每档前重新 soak，保证测的是稳态而非冷机峰值")
    print()

    print("-" * 78)
    print("  [0] 初始稳态 soak：8192³ 满载 20s")
    print("-" * 78)
    smp = soak(20.0)
    pw = [s["power"] for s in smp if s["power"] == s["power"]]
    print(f"    功耗：峰 {max(pw):.1f} W / 中位 {sorted(pw)[len(pw)//2]:.1f} W"
          f" / 末次 {pw[-1]:.1f} W，最高温 {max(s['temp'] for s in smp):.0f}°C"
          f"，降频标志 {smp[-1]['throttle']}")
    print()

    print("-" * 78)
    print("  [1] FP4 逐档（每档：先 soak 12s → 再纯 event 计时 3s，功耗全程采样）")
    print("-" * 78)
    print(f"  {'n':>6} {'iters':>7} {'ms/iter':>9} {'TFLOPS':>8} {'等效带宽':>10} "
          f"{'驻留量':>8} {'L2倍':>6}  功耗 峰/中位   温度")
    results = []
    for n in (1024, 2048, 4096, 6144, 8192):
        soak(12.0)                                  # 每档前重置到稳态
        ps = PowerSampler(interval=0.25); ps.start()
        r = bench_fp4(n)
        smp2 = ps.stop()
        pw = [s["power"] for s in smp2 if s["power"] == s["power"]]
        pk = max(pw) if pw else float("nan")
        md = sorted(pw)[len(pw) // 2] if pw else float("nan")
        tp = max((s["temp"] for s in smp2), default=float("nan"))
        results.append({**r, "power_peak": pk, "power_med": md, "temp": tp,
                        "throttle": smp2[-1]["throttle"] if smp2 else "?"})
        print(f"  {n:>6} {r['iters']:>7} {r['ms_per_iter']:>9.4f} {r['tflops']:>8.1f} "
              f"{r['eff_bw_gbps']:>7.0f}GB/s {r['resid_mb']:>6.0f}MB "
              f"{r['L2_multiple']:>5.1f}x  {pk:>5.1f}/{md:<5.1f}W {tp:>4.0f}°C")
    print()
    print("-" * 78)
    print("  判读")
    print("-" * 78)
    peak = max(results, key=lambda r: r["tflops"])
    print(f"  · 峰值 {peak['tflops']:.1f} TFLOPS @ n={peak['n']}"
          f"（该档驻留量 {peak['resid_mb']:.0f}MB = L2 的 {peak['L2_multiple']:.1f}×）")
    for r in results:
        regime = ("L2 内（可能被缓存放大）" if r["L2_multiple"] <= 1.0
                  else "超 L2（走显存，更接近真实）")
        print(f"  · n={r['n']:<5} {r['tflops']:>7.1f} TFLOPS  {regime}")
    print()
    print("  ⚠️ 用于规划的口径：**驻留量明显超 L2 的档**才是可信的持续算力；")
    print("     L2 内的小档会被缓存放大，不得当作可用峰值。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
