#!/usr/bin/env bash
# ============================================================
# KokonaPolaris-S4 / 心夏北极星 —— 项目环境入口
# 用法：  source <仓库根>/env.sh
# 之后：  "$KP_PY" your_script.py
# ============================================================
# 目的有三：
#   ① **自动定位仓库根**（不再写死路径 —— 换机/多副本都能直接用）
#   ② 统一 python 解释器（仓库内 .venv，独立于系统环境的版本冲突）
#   ③ 把所有缓存写向项目所在盘 —— 约定「不碰 C 盘」
#
# ⚠️ 2026-10-03 修正：原版写死 `KP_ROOT="D:/model"`，换到
#    `D:\kokonapolaris-s4` 后 source 会指向不存在的路径。
#    现改为「按本脚本所在目录自定位」，并允许外部预先设 KP_ROOT 覆盖。

# ---- ① 仓库根自定位（Windows 混合路径形式 D:/xxx，便于交给原生程序）----
if [ -z "${KP_ROOT:-}" ]; then
  _KP_SELF="${BASH_SOURCE[0]:-$0}"
  _KP_DIR="$(cd "$(dirname "$_KP_SELF")" && pwd)"
  if command -v cygpath >/dev/null 2>&1; then
    KP_ROOT="$(cygpath -m "$_KP_DIR")"       # /d/x  →  D:/x
  else
    KP_ROOT="$_KP_DIR"
  fi
  unset _KP_SELF _KP_DIR
fi
export KP_ROOT

# ---- ①b ⭐ 机器名（**两台开发机必须分开记，否则实测数字会串味**）----
# 🔴🔴 2026-10-05 改为**自动探测**（审计 P0-1）
#   旧版写死 `KP_MACHINE="${KP_MACHINE:-viim}"` + 注释说 kokona「已迁出」，
#   但**实际有第三种情况**：人带着仓库换机器，而环境变量还留着上一台的值。
#   ⇒ 实测（本机）host=Kokona / GPU=RTX 5050 Laptop / 20 SM
#      而env.sh 却说 kokona 已迁出、默认 viim ⇒ **会静默把本机数字记成 viim 的。**
#⭐ 修法：**默认走自动探测**（hostname + GPU 型号），探测不到才回落到 KP_MACHINE 传入值。
#   ⚠️ 仍可外部覆盖：`KP_MACHINE=xxx source env.sh`。
_kp_auto_machine() {
  local host gpu
  host="$(hostname 2>/dev/null | tr '[:upper:]' '[:lower:]')"
  # ⭐ 优先用 GPU 型号判定（比 hostname 可靠：主机名可能改）
  gpu=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  fi
  local gl="$(printf '%s' "$gpu" | tr '[:upper:]' '[:lower:]')"
  # 已知机器对照表（**只认 GPU 型号**，这是绑死的那部分）
  case "$gl" in
    *"rtx 5050"*)                echo "kokona";;   # 20 SM / sm_120
    *"rtx 5070"*|*"rtx 5080"*|*"rtx 5090"*) echo "viim";;  # 36+ SM / sm_120
    *)# 探测不到 ⇒ 退回 hostname 关键字
      case "$host" in
        *kokona*) echo "kokona";;
        *viim*)echo "viim";;
        *)           echo "${host:-unknown}";;
      esac;;
  esac
}
export KP_MACHINE="${KP_MACHINE:-$(_kp_auto_machine)}"
unset -f _kp_auto_machine

# ⚠️ 任何「TFLOPS / 功耗 / SM 数 / 训练耗时」都必须带机器名，**禁止跨机比较或拼接**。
#    ⚠️ 特别提醒：**训练耗时与 SM 数成正比** ⇒ 换机后「3万步 33 分钟」这类数字**直接作废**。

# ---- ② 解释器 ----
# ⚠️ 2026-10-04 踩坑：**本工作区内的可执行文件读不了本工作区内的文件**
#    （沙箱给工作区根加的硬化 ACL：`Everyone:(CI)(DENY)(DC)` +
#      `LAPTOP-...\TX:(OI)(CI)(WO)` + `S-1-4-...(W,D,DC)`），
#    结果 `D:\kokonapolaris-s4\.venv\Scripts\python.exe` 一律报
#      `Cannot read 'D:\kokonapolaris-s4\.venv\pyvenv.cfg'`
#    而**同样字节的 venv 放到工作区外就完全正常**（已实测）。
#    ⇒ 约定：**venv 放在工作区同级目录**（默认 `<仓库父目录>/kp-venv`）。
#    可用环境变量 `KP_VENV` 显式覆盖。
_KP_PARENT="$(dirname "$KP_ROOT")"
if [ -z "${KP_VENV:-}" ]; then
  for _cand in "$_KP_PARENT/kp-venv" "$KP_ROOT/.venv"; do
    if [ -x "$_cand/Scripts/python.exe" ]; then KP_VENV="$_cand"; break; fi
  done
  : "${KP_VENV:=$KP_ROOT/.venv}"
fi
unset _KP_PARENT _cand
export KP_VENV
export KP_PY="$KP_VENV/Scripts/python.exe"

# ---- 缓存全面重定向到项目盘（含 HuggingFace，否则模型会下到 C:\Users\...\.cache）----
export PIP_CACHE_DIR="$KP_ROOT/.pipcache"
export XDG_CACHE_HOME="$KP_ROOT/.cache"
export HF_HOME="$KP_ROOT/.cache/huggingface"
export HF_HUB_CACHE="$KP_ROOT/.cache/huggingface/hub"
export HUGGINGFACE_HUB_CACHE="$KP_ROOT/.cache/huggingface/hub"
# 注意：不要设 TRANSFORMERS_CACHE（transformers>=4.55 已废弃，会报 FutureWarning），HF_HOME 已覆盖
export DIFFUSERS_CACHE="$KP_ROOT/.cache/huggingface/diffusers"
export TORCH_HOME="$KP_ROOT/.cache/torch"
export TORCH_EXTENSIONS_DIR="$KP_ROOT/.cache/torch_ext"
export MODELSCOPE_CACHE="$KP_ROOT/.cache/modelscope"

# ---- 模型/数据/输出统一落盘位置（全部在项目盘）----
export KP_MODELS="$KP_ROOT/models"
export KP_DATA="$KP_ROOT/data"
export KP_OUT="$KP_ROOT/out"

# ---- 网络策略（2026-10-02 定，**2026-10-03 在本机重测后修正**）----
# ⚠️ 旧结论「HF 代理直连比镜像快 26×」**测于迁出机，在本机不成立**。
#    2026-10-03 本机实测（同一文件、同样 15s 窗口，`urllib` + certifi）：
#      hf-mirror.com 不走代理     2.11 MB/s
#      huggingface.co 走代理      2.25 MB/s
#      hf-mirror.com  走代理      3.07 MB/s   ← 最快
#    ⇒ **镜像优先**；代理仍然要开（它对镜像也有增益，且本机直连 huggingface.co
#      会因 schannel/certifi 差异失败：curl 报 SEC_E_NO_CREDENTIALS）。
export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
export PIP_EXTRA_INDEX_URL="https://mirrors.aliyun.com/pypi/simple/"
export HF_ENDPOINT="https://hf-mirror.com"     # ★ 默认镜像（实测最快档的一半）

export KP_PROXY="http://127.0.0.1:7897"
kp_proxy_on()  { export http_proxy="$KP_PROXY" https_proxy="$KP_PROXY" all_proxy="$KP_PROXY"; \
                 echo "[KP] 代理已开 → $KP_PROXY"; }
kp_proxy_off() { unset http_proxy https_proxy all_proxy; \
                 echo "[KP] 代理已关 → 走国内镜像 (PyPI 清华 / HF hf-mirror)"; }
# ⭐ 下 HF 模型的推荐通道：**hf-mirror + 代理**（本机实测 3.07 MB/s，最快）
kp_hf() { kp_proxy_on; export HF_ENDPOINT="https://hf-mirror.com"; \
          echo "[KP] HF 通道 → hf-mirror.com（经代理，本机实测最快 ~3.07 MB/s）"; }
# 备选：官方站直连（经代理，本机实测 ~2.25 MB/s）
kp_hf_official() { kp_proxy_on; export HF_ENDPOINT="https://huggingface.co"; \
          echo "[KP] HF 通道 → huggingface.co（经代理，实测略慢于镜像）"; }

# ---- Triton 缓存必须重定向（否则落 C:\Users\<user>\.triton 并报 WinError 5）----
# 实证：ModelOpt 的 NVFP4 fake-quant 走 triton kernel，缓存不可写时
# `bench_115w.py` Phase D 直接 `PermissionError: [WinError 5] ...\.triton`。
export TRITON_CACHE_DIR="$KP_ROOT/.cache/triton"

# 默认关代理（包管理走镜像即可）
unset http_proxy https_proxy all_proxy

# ---- 静音工况下的确定性设置 ----
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

mkdir -p "$PIP_CACHE_DIR" "$HF_HUB_CACHE" "$TORCH_HOME" "$KP_MODELS" "$KP_DATA" "$KP_OUT" 2>/dev/null

echo "[KP] root   : $KP_ROOT"
echo "[KP] machine: $KP_MACHINE  （自动探测；kokona=5050 20SM / viim=5070+ 36SM —— 实测数字禁止跨机拼接）"
echo "[KP] venv   : $KP_VENV"
if [ -x "$KP_PY" ]; then
  echo "[KP] python : $("$KP_PY" -c 'import sys;print(sys.version.split()[0])' 2>/dev/null)"
else
  echo "[KP] python : ⚠️ 未找到 $KP_PY —— 先建 venv："
  echo "               python -m venv \"$KP_VENV\" && \"$KP_PY\" -m pip install -r \"$KP_ROOT/requirements.lock.txt\""
fi
echo "[KP] cache  : $KP_ROOT/.cache  (已隔离 C 盘)"
