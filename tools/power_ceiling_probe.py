"""KokonaPolaris · 功耗天花板探针（E7 修正版）

背景：
  上一轮 bench_115w.py 实测 soak 30s 稳定在 78-80W，据此下了「功耗天花板 ~80W」的结论。
  但 nvidia-smi 报告 Max Power Limit = 115.00 W（Default 55W）。
  说明 80W 是「负载没吃到」，不是「墙在 80W」。

本探针要回答：
  Q1. 115W 到底吃不吃得到？（换负载形态试）
  Q2. 如果吃不到，是什么在限制？→ 读 Clocks Throttle Reasons
  Q3. 不同负载形态下的功耗/吞吐，哪一档对本项目的训练/推理最划算？

负载形态（关键：FurMark 是「光栅+显存+显示引擎同时压」，与纯 tensor 完全不同）：
  P1 bf16 大 GEMM          —— 纯 tensor core，SM 时钟需求低
  P2 FP4 block-scaled GEMM —— 纯 tensor core，极限吞吐
  P3 D2D 大拷贝            —— 纯显存带宽（FurMark 的显存压力部分）
  P4 fp32 大 elementwise   —— SM ALU/寄存器重载，逼高频
  P5 三流并发 mix          —— GEMM + 拷贝 + elementwise 同时压（最接近 FurMark 形态）

用法：python power_ceiling_probe.py
"""
import os
import subprocess
import threading
import time

import torch

DEV = "cuda"
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "power_ceiling_log.csv")
PHASE_SEC = 18

# ---------------------------------------------------------------- 采样
_samples = []
_stop = False


def _sampler():
    while not _stop:
        try:
            o = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=power.draw,clocks.sm,temperature.gpu,utilization.gpu,"
                 "utilization.memory,pstate",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5).stdout.strip()
            _samples.append((time.time(), o))
        except Exception:
            pass
        time.sleep(0.8)


def throttles():
    """返回当前处于 Active 的降频原因列表"""
    try:
        o = subprocess.run(["nvidia-smi", "-q", "-d", "PERFORMANCE"],
                           capture_output=True, text=True, timeout=10).stdout
    except Exception as e:
        return [f"<读取失败 {e}>"]
    out, insec = [], False
    for ln in o.splitlines():
        s = ln.strip()
        if s.startswith("Clocks Throttle Reasons") or s.startswith("Clocks Event Reasons"):
            insec = True
            continue
        if insec:
            if ":" in s:
                k, v = s.rsplit(":", 1)
                if v.strip().startswith("Active"):
                    out.append(k.strip())
            else:
                insec = False
    return out


def phase_stats(t0=0.0, t1=None):
    """截取时间段内的样本，给出峰值/稳态功耗"""
    t1 = t1 or time.time()
    ps = []
    for ts, o in _samples:
        if ts < t0 or ts > t1:
            continue
        f = [x.strip() for x in o.split(",")]
        try:
            ps.append((float(f[0]), float(f[1]), float(f[2])))
        except Exception:
            continue
    if not ps:
        return None
    powers = [p[0] for p in ps]
    # 稳态：去掉前 30% 爬升段
    tail = powers[max(1, int(len(powers) * 0.3)):]
    return {
        "peak": max(powers),
        "steady": sum(tail) / len(tail),
        "clk": max(p[1] for p in ps),
        "temp": max(p[2] for p in ps),
        "n": len(ps),
    }


# ---------------------------------------------------------------- 负载
def load_tensor(dtype, n, sec):
    x = torch.randn(n, n, device=DEV, dtype=dtype)
    y = torch.randn(n, n, device=DEV, dtype=dtype)
    for _ in range(3):
        x @ y
    torch.cuda.synchronize()
    t0, it = time.time(), 0
    while time.time() - t0 < sec:
        x @ y
        it += 1
    torch.cuda.synchronize()
    del x, y
    torch.cuda.empty_cache()
    return it


def load_fp4(n, sec):
    def one():
        a = torch.randint(0, 256, (n, n // 2), device=DEV, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
        b = torch.randint(0, 256, (n, n // 2), device=DEV, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
        a_s = torch.ones(n, n // 16, device=DEV, dtype=torch.float8_e4m3fn)
        b_s = torch.ones(n, n // 16, device=DEV, dtype=torch.float8_e4m3fn)
        return torch._scaled_mm(a, b.t(), a_s, b_s, out_dtype=torch.bfloat16)

    for _ in range(4):
        one()
    torch.cuda.synchronize()
    t0, it = time.time(), 0
    while time.time() - t0 < sec:
        one()
        it += 1
    torch.cuda.synchronize()
    return it


def load_bw(sec):
    """D2D 大拷贝 —— 纯显存带宽压力（~2GB 级缓冲）"""
    n = 400_000_000 // 4  # 400MB fp32
    a = torch.empty(n, device=DEV, dtype=torch.float32)
    b = torch.empty_like(a)
    a.uniform_()
    for _ in range(3):
        b.copy_(a)
    torch.cuda.synchronize()
    t0, it = time.time(), 0
    while time.time() - t0 < sec:
        b.copy_(a)
        it += 1
    torch.cuda.synchronize()
    del a, b
    torch.cuda.empty_cache()
    return it


def load_alu(sec):
    """fp32 大 elementwise —— SM ALU 重载，逼出高 SM 时钟"""
    n = 200_000_000 // 4
    a = torch.randn(n, device=DEV, dtype=torch.float32)
    for _ in range(3):
        a.mul_(2.0).add_(1.0)
    a.mul_(0.5)
    torch.cuda.synchronize()
    t0, it = time.time(), 0
    while time.time() - t0 < sec:
        a.mul_(2.0).add_(1.0)
        a.mul_(0.5)
        it += 1
    torch.cuda.synchronize()
    del a
    torch.cuda.empty_cache()
    return it


def load_mix(sec):
    """三流并发：GEMM + 拷贝 + elementwise —— 最接近 FurMark 的「全单元同时压」"""
    s1, s2, s3 = torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream()
    x = torch.randn(6144, 6144, device=DEV, dtype=torch.bfloat16)
    y = torch.randn(6144, 6144, device=DEV, dtype=torch.bfloat16)
    nb = 150_000_000 // 4
    a = torch.empty(nb, device=DEV, dtype=torch.float32)
    b = torch.empty_like(a)
    a.uniform_()
    na = 100_000_000 // 4
    c = torch.randn(na, device=DEV, dtype=torch.float32)

    def r1():
        with torch.cuda.stream(s1):
            while not _done[0]:
                x @ y

    def r2():
        with torch.cuda.stream(s2):
            while not _done[0]:
                b.copy_(a)

    def r3():
        with torch.cuda.stream(s3):
            while not _done[0]:
                c.mul_(1.0001).add_(0.0001)

    _done = [False]
    ths = [threading.Thread(target=f, daemon=True) for f in (r1, r2, r3)]
    for t in ths:
        t.start()
    time.sleep(sec)
    _done[0] = True
    torch.cuda.synchronize()
    for t in ths:
        t.join(timeout=10)
    del x, y, a, b, c
    torch.cuda.empty_cache()
    return 0


# ---------------------------------------------------------------- 主流程
print("=" * 78)
print("  KokonaPolaris-S4 · 功耗天花板探针（修正 bench_115w 的 80W 结论）")
print("=" * 78)
print(f"  torch {torch.__version__}   device {torch.cuda.get_device_name(0)}")
lim = subprocess.run(["nvidia-smi", "--query-gpu=power.default_limit,power.max_limit",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
print(f"  功耗限制（Default / Max）: {lim}")
print(f"  起始降频原因: {throttles() or ['无']}")
print()

th = threading.Thread(target=_sampler, daemon=True)
th.start()
time.sleep(2)

PHASES = []


def run_phase(name, fn, desc):
    print("-" * 78)
    print(f"  [{name}] {desc}")
    print("-" * 78)
    print(f"    起始: {throttles() or ['无活跃降频']}")
    t0 = time.time()
    try:
        it = fn(PHASE_SEC)
    except Exception as ex:
        print(f"    ✗ 失败 {type(ex).__name__}: {str(ex)[:160]}")
        PHASES.append((name, desc, None, f"{type(ex).__name__}"))
        return
    t1 = time.time()
    st = phase_stats(t0, t1)
    act = throttles()
    if st:
        print(f"    稳态 {st['steady']:6.1f} W | 峰值 {st['peak']:6.1f} W | "
              f"SM 峰值 {st['clk']:5.0f} MHz | 温峰 {st['temp']:.0f}°C | iter {it}")
    print(f"    结束降频原因: {act or ['无活跃降频']}")
    PHASES.append((name, desc, st, "; ".join(act) or "无"))
    torch.cuda.synchronize()
    time.sleep(3)


run_phase("P1", lambda s: load_tensor(torch.bfloat16, 8192, s), "bf16 大 GEMM（纯 tensor core）")
run_phase("P2", lambda s: load_fp4(8192, s), "FP4 block-scaled GEMM（极限 tensor 吞吐）")
run_phase("P3", load_bw, "D2D 大拷贝（纯显存带宽）")
run_phase("P4", load_alu, "fp32 大 elementwise（SM ALU 重载，逼高频）")
run_phase("P5", load_mix, "三流并发 mix（GEMM+拷贝+elementwise，FurMark 形态）")

_stop = True
time.sleep(1)

# 落盘
with open(LOG, "w", encoding="utf-8") as f:
    f.write("ts,power_w,sm_mhz,temp_c,gpu_util,mem_util,pstate\n")
    for ts, o in _samples:
        f.write(f"{time.strftime('%H:%M:%S', time.localtime(ts))},{o}\n")

print()
print("=" * 78)
print("  汇总：各负载形态的功耗天花板")
print("=" * 78)
print(f"  {'形态':<8}{'稳态功耗':>10}{'峰值功耗':>10}{'SM峰值':>9}{'温峰':>7}   降频原因")
for name, desc, st, act in PHASES:
    if st:
        print(f"  {name:<8}{st['steady']:>9.1f}W{st['peak']:>9.1f}W{st['clk']:>8.0f}M{st['temp']:>6.0f}°C   {act}")
    else:
        print(f"  {name:<8}   -- 失败 / 跳过 --")
print()
print(f"  采样明细已写入 {LOG}")
print(f"  最终状态: {throttles() or ['无活跃降频']}")
