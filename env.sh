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

# ---- ② 解释器 ----
export KP_VENV="$KP_ROOT/.venv"
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

# ---- 网络策略（2026-10-02 实测校准，勿凭直觉改）----
# 实测结论：包管理走国内镜像够快；但 HuggingFace 的 hf-mirror 很慢，
#   hf-mirror   : 单流 192 KB/s，4 并发 3.3 MB/s
#   代理直连 HF :           86 MB/s   ← 快约 26×
# ⇒ 分流原则：PyPI→镜像；HF 模型下载→ kp_hf（开代理 + 直连原站）
export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
export PIP_EXTRA_INDEX_URL="https://mirrors.aliyun.com/pypi/simple/"
export HF_ENDPOINT="https://hf-mirror.com"     # 仅作「无代理时兜底」，慢

export KP_PROXY="http://127.0.0.1:7897"
kp_proxy_on()  { export http_proxy="$KP_PROXY" https_proxy="$KP_PROXY" all_proxy="$KP_PROXY"; \
                 echo "[KP] 代理已开 → $KP_PROXY"; }
kp_proxy_off() { unset http_proxy https_proxy all_proxy; \
                 echo "[KP] 代理已关 → 走国内镜像 (PyPI 清华 / HF hf-mirror)"; }
# 下载 HF 模型专用通道（实测快 ~26×）
kp_hf() { kp_proxy_on; export HF_ENDPOINT="https://huggingface.co"; \
          echo "[KP] HF 通道 → huggingface.co 直连（经代理）"; }

# 默认关代理（包管理走镜像即可）
unset http_proxy https_proxy all_proxy

# ---- 静音工况下的确定性设置 ----
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

mkdir -p "$PIP_CACHE_DIR" "$HF_HUB_CACHE" "$TORCH_HOME" "$KP_MODELS" "$KP_DATA" "$KP_OUT" 2>/dev/null

echo "[KP] root   : $KP_ROOT"
echo "[KP] venv   : $KP_VENV"
if [ -x "$KP_PY" ]; then
  echo "[KP] python : $("$KP_PY" -c 'import sys;print(sys.version.split()[0])' 2>/dev/null)"
else
  echo "[KP] python : ⚠️ 未找到 $KP_PY —— 先建 venv："
  echo "               python -m venv \"$KP_VENV\" && \"$KP_PY\" -m pip install -r \"$KP_ROOT/requirements.lock.txt\""
fi
echo "[KP] cache  : $KP_ROOT/.cache  (已隔离 C 盘)"
