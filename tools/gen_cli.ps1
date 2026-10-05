<#
================================================================================
  KP local image generator  ·  console edition (A/B natural-language)
================================================================================
  Double-click gen_one_click.cmd -> a black console window opens.

    STEP 1  it PRINTS the base description on screen (your last one, or the
            built-in default). Edit it, or press Enter to keep it.
    STEP 2  you type the EXTRA description (plain natural language, e.g. the
            tags/params you want to add). Press Enter to skip.
    SHOT A  base description only
    SHOT B  base description + extra description
            * both use the SAME seed, so the only difference between the two
              images is the extra words -- that is what makes the A/B honest.
    both images open automatically, press Enter to close.

  PRIVACY: what you type stays on this machine. It is never sent to any cloud
  or AI assistant, never written to a log, and never passed as a command-line
  argument. The sidecar json records size/steps/seed only.

  MEASURED (RTX 5050 Laptop / 8GB, CPU offload):
    model load ~3s, then ~22s per 1024px/20-step image.
================================================================================
#>

$ErrorActionPreference = "Continue"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

$here   = Split-Path -Parent $MyInvocation.MyCommand.Path
$root   = Split-Path -Parent $here
$py     = Join-Path $root ".venv\Scripts\python.exe"
$outDir = Join-Path $root "out\local_gen"
$posFile = Join-Path $root "out\_local_prompt.txt"
$negFile = Join-Path $root "out\_local_negative.txt"
$utf8   = New-Object System.Text.UTF8Encoding($false)

# ─── 默认 prompt（用户指定：二次元 + 夕阳海滩）──────────────────────────
# ⭐ 2026-10-05 用户要求：「框架默认就是二次元，不管是不是 nsfw」
#   ⇒ 把风格词写进**默认**里，这样 A 图（基础）本身就是二次元，
#     B 图（+增量词）是在二次元基础上加内容 ⇒ 差距才是"内容"的差距。
#
# ⚠️ 为什么风格词放 base 而不是让你每次自己加：
#   A/B 是**同seed** 对照，唯一的变量应该是「你加的那些词」。
#   若二次元词也算增量，A 就不是二次元 ⇒ 两张图的风格会一起变 ⇒ 看不清差异来源。
$STYLE_SUFFIX = "，日式动漫风格，二次元插画，干净的线条，柔和的赛璐璐上色，高细节"
$BUILTIN_DEFAULT = "夕阳海滩上的少女$STYLE_SUFFIX"

function Die([string]$msg) {
    Write-Host ""
    Write-Host "  [X] $msg" -ForegroundColor Red
    Write-Host ""
    Read-Host "  press Enter to close" | Out-Null
    exit 1
}

if (-not (Test-Path $py))     { Die "python not found: $py (create the venv first)" }
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir -Force | Out-Null }

# 读取「上次用的 base」（若存在）
# ⚠️ 2026-10-05 用户实际踩到的坑：这里**无条件优先旧文件**⇒
#    别的对话测试时留了英文在 out/_local_prompt.txt ⇒ 界面一直显示英文，
#    且**看不出它来自旧文件还是内置默认**。
#✅ 修法：① 明确标注来源② 让你能一键回落到内置默认。
$last = ""
$fromFile = $false
if (Test-Path $posFile) {
    try {
        $last = ([System.IO.File]::ReadAllText($posFile, [System.Text.Encoding]::UTF8)).Trim()
        $fromFile = -not [string]::IsNullOrWhiteSpace($last)
    } catch { }
}

Clear-Host
Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host "  KP image A/B test   Sana 1.6B/ fully offline" -ForegroundColor Cyan
Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host "  what you type below never leaves this PC." -ForegroundColor Gray
Write-Host "  A = base only    B = base + extra    (same seed for both)" -ForegroundColor Gray
Write-Host ""

# ------------------------------------------------------------- STEP 1: base
Write-Host "  STEP 1 / 2   BASE DESCRIPTION" -ForegroundColor Yellow
if ($fromFile) {
    Write-Host "  (from your last run -- type 'd' to fall back to the built-in default)" -ForegroundColor DarkGray
} else {
    Write-Host "  (built-in default)" -ForegroundColor DarkGray
}
Write-Host ""
Write-Host "      $last" -ForegroundColor White
Write-Host ""
$base = Read-Host "  edit it, 'd' = use built-in default, or press Enter to keep as-is"
if ($base -match '^[dD]$') { $base = $BUILTIN_DEFAULT }
if ([string]::IsNullOrWhiteSpace($base)) { $base = $last }
Write-Host ""
# ⚠️ 2026-10-05 改：**不把内容回显到屏幕**。
#    原版会把base/extra 原文打印在控制台（+ 留在终端 scrollback 里）——
#    那与本脚本自己声明的「never written to a log」矛盾。
#    ⇒ 这里只报**长度**；内容只写本地文件、只进模型。
$baseLen = $base.Length
Write-Host "  >> A will use your base text ($baseLen chars, not shown here)." -ForegroundColor Green
Write-Host ""

# ---------------------------------------------------------- STEP 2: extra
Write-Host "  STEP 2 / 2   EXTRA DESCRIPTION (plain words, not a 'negative')" -ForegroundColor Yellow
$extra = Read-Host "  what to add for B? (Enter = skip, B then equals A)"
if ([string]::IsNullOrWhiteSpace($extra)) { $extra = "" }

if ($extra -eq "") {
    $combined = $base
} else {
    $combined = $base + ", " + $extra
}

Write-Host ""
# ⚠️ 同上：只报长度，不回显内容（STEP 1 那次是"显示给你改"，
#    STEP 2 是纯增量提示，没有回显的必要 ⇒ 一律不打）。
Write-Host "  >> A : your base text ($($base.Length) chars)" -ForegroundColor Green
if ($extra -eq "") {
    Write-Host "  >> B  (base + extra)  : (empty)  -- B would just repeat A" -ForegroundColor Yellow
} else {
    Write-Host "  >> B  (base + extra)  : your base +$($extra.Length) extra chars" -ForegroundColor Green
}
Write-Host ""

# --------------------------------------------------------------- confirm
$size  = 1024
$steps = 20
$seed  = Get-Random -Minimum 100000 -Maximum 999999999

Write-Host "  ===========================================================" -ForegroundColor DarkCyan
Write-Host "   size=$size  steps=$steps  seed=$seed   (about 22s each)"
if ($extra -eq "") { Write-Host "   WARNING: extra is empty -> A and B come out identical" -ForegroundColor Yellow }
Write-Host "  ===========================================================" -ForegroundColor DarkCyan
Write-Host ""
$go = Read-Host "  press Enter to start  (or type x then Enter to cancel)"
if ($go -match "^[xX]") { Write-Host "  cancelled."; Start-Sleep -Seconds 2; exit 0 }

$before = @(Get-ChildItem $outDir -Filter "*.png" -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty Name)

# shot A -- base only, NO negative string at all
Write-Host ""
Write-Host "  ---- shot A : base only ----" -ForegroundColor Magenta
[System.IO.File]::WriteAllText($posFile, $base, $utf8)
[System.IO.File]::WriteAllText($negFile, "", $utf8)
& $py -u (Join-Path $here "local_gen.py") --size $size --steps $steps --seed $seed
$rcA = $LASTEXITCODE

# shot B -- base + extra, also plain natural language
Write-Host ""
Write-Host "  ---- shot B : base + extra ----" -ForegroundColor Magenta
[System.IO.File]::WriteAllText($posFile, $combined, $utf8)
[System.IO.File]::WriteAllText($negFile, "", $utf8)
& $py -u (Join-Path $here "local_gen.py") --size $size --steps $steps --seed $seed
$rcB = $LASTEXITCODE

# --------------------------------------------------------------- results
$new = @(Get-ChildItem $outDir -Filter "*.png" -ErrorAction SilentlyContinue |
         Where-Object { $before -notcontains $_.Name } |
         Sort-Object LastWriteTime)

Write-Host ""
Write-Host "  ===========================================================" -ForegroundColor Cyan
if ($new.Count -ge 2) {
    Write-Host "  done. compare these two:" -ForegroundColor Green
    Write-Host "    A  base only     : $($new[0].Name)" -ForegroundColor Green
    Write-Host "    B  base + extra  : $($new[1].Name)" -ForegroundColor Green
    Write-Host "  opening both..." -ForegroundColor Gray
    Start-Process explorer.exe $outDir
    Start-Process explorer.exe $new[0].FullName
    Start-Process explorer.exe $new[1].FullName
} elseif ($new.Count -eq 1) {
    Write-Host "  only 1 image came out (exit A=$rcA  B=$rcB) - check log above." -ForegroundColor Yellow
    Write-Host "    $($new[0].Name)" -ForegroundColor Yellow
    Start-Process explorer.exe $new[0].FullName
} else {
    Write-Host "  no image produced (exit A=$rcA  B=$rcB). read the log above." -ForegroundColor Red
}
Write-Host "  ===========================================================" -ForegroundColor Cyan
Write-Host ""
Read-Host "  press Enter to close the window" | Out-Null
