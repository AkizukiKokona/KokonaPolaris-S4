"""⏱️ 可中断训练：让长任务**随时能停、停了不丢、接着能续**。

═══ 为什么需要这个 ═══

用户可能随时关机（"我要关电脑了，训练大概训练不完"）。
⇒ 长任务**必须**满足三条，**不满足就是设计缺陷**：
    ① 停 ⇒ **不丢**已完成的进度（checkpoint）
    ② 续 ⇒ 接着上次的地方跑（不是从头）
    ③ 存 ⇒ checkpoint 在**工作区**（关机不丢），不在临时目录

⚠️ 本项目已踩过的坑（别重犯）：
    · `tempfile.gettempdir()` 在受沙箱限制的机器上会退化成 cwd
      ⇒ **临时产物一律落 `KP_OUT` / `artifacts/`，关机不丢**
    · `nohup ... &` 起的进程**会随父进程退出被带走**（实测进程数归 0）
      ⇒ 长期挂机必须用 `DETACHED_PROCESS`（见 `kp/data/daemon.py`）

═══ 用法 ═══

    # 1) 带自动保存的长任务
    from kp.train.checkpoint import Ckpt, auto_save_every
    ck = Ckpt(Path("out/vae/run1"), every=50)
    for step in range(10000):
        loss = train_one_step()
        if ck.maybe_save(step, {"loss": loss, "model": model.state_dict()}):
            print(ck.status(step))          # 已自动保存
        if user_wants_stop():               # ← 由你决定何时停
            ck.save(step, {...}, reason="user-stop")
            break
    # 2) 续跑
    st = ck.load()                            # ⛔ 没有 checkpoint 时返回 None（不报错）

    # 3) 查进度
    python -m kp.train.checkpoint --status out/vae/run1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch


class Ckpt:
    """带**自动保存 + 断点续跑**的训练检查点。

    ⭐ 三条保证（对应上面的①②③）：
        · `maybe_save(step, payload)` —— 每 `every` 步自动存
        · `save(step, payload, reason=...)` —— 随时可停时手动存
        · 路径**必须**在 `out/` 或 `artifacts/` 下（关机不丢）
    """

    def __init__(self, root, every: int = 100):
        self.root = Path(root)
        self.every = max(1, int(every))
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "ckpt.pt"
        self.meta = self.root / "progress.json"

    # ---------- 存 ----------
    def save(self, step: int, payload: Dict[str, Any], reason: str = "") -> Path:
        tmp = self.path.with_suffix(".pt.tmp")
        # ⭐ 先写临时文件再原子替换 ⇒ 断电/强杀时**不会留下半个坏文件**
        torch.save({"step": step, **payload}, tmp)
        tmp.replace(self.path)
        self.meta.write_text(json.dumps({
            "step": step, "reason": reason, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "keys": [k for k in payload if k != "model"],
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        return self.path

    def maybe_save(self, step: int, payload: Dict[str, Any]) -> bool:
        if step % self.every == 0:
            self.save(step, payload, reason="auto")
            return True
        return False

    # ---------- 续 ----------
    def load(self) -> Optional[Dict[str, Any]]:
        """⚠️ 损坏的 checkpoint **如实返回 None**，不抛异常 —— 让调用方决定重训。"""
        if not self.path.exists():
            return None
        try:
            return torch.load(self.path, weights_only=False, map_location="cpu")
        except Exception:                                  # noqa: BLE001
            return None

    def status(self, step: Optional[int] = None) -> str:
        if not self.meta.exists():
            return "无检查点（未开始）"
        m = json.loads(self.meta.read_text(encoding="utf-8"))
        return (f"step {m.get('step')} · {m.get('saved_at')} · {m.get('reason')}"
                + (f" · 刚到 {step}" if step is not None else ""))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="检查点工具（可中断训练）")
    ap.add_argument("--status", default=None, help="查某个 run 的进度")
    a = ap.parse_args(argv)
    if a.status:
        c = Ckpt(a.status)
        st = c.load()
        print(f"检查点: {'✅ 存在，可续跑' if st else '⛔ 无或损坏（需从头）'}")
        print(f"状态  : {c.status()}")
        if st:
            print(f"step  : {st.get('step')}")
        return 0
    print(__doc__.split("═══ 用法")[0])
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
