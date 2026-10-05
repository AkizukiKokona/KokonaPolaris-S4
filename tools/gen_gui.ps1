<#
================================================================================
  KP 本地出图器 · GUI 启动器
================================================================================
  双击本文件即可（PowerShell 会弹出一个窗口让你填内容）。

  ⭐ 隐私设计：
     - 你在弹窗里填的内容 **只写进本地文件 out/_local_prompt.txt**
     - **不经过任何云端 / AI 助手**
     - Agent 只负责「起窗口 → 你点生成 → 看图」，看不到你写了什么

  用法：右键本文件 → 「使用 PowerShell 运行」
     或在 bash 里： powershell -ExecutionPolicy Bypass -File tools/gen_gui.ps1
================================================================================
#>

$ErrorActionPreference = "Stop"

# ---- 定位仓库根（不写死盘符）----
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $here
$py   = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    Add-Type -AssemblyName System.Windows.Forms
    [System.Windows.Forms.MessageBox]::Show(
        "找不到 python：$py`n`n请先建 venv（见 env.sh / README）。",
        "KP 出图器", "OK", "Error") | Out-Null
    exit 1
}

$outDir = Join-Path $root "out\local_gen"
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir -Force | Out-Null }

# ---- 载入上次填的内容（纯本地，方便你改）----
$promptPath = Join-Path $root "out\_local_prompt.txt"
$negPath    = Join-Path $root "out\_local_negative.txt"
# ⭐ 内置默认（用户指定）。**必须在读 $lastPrompt 之前定义**——
#    之前它定义在后面 ⇒ 读文件时该变量还是空的（PowerShell 未定义变量= null）。
$DEFAULT_PROMPT = "夕阳海滩上的少女"

$lastPrompt = if (Test-Path $promptPath) { (Get-Content $promptPath -Raw -Encoding UTF8).Trim() } else { "" }
# ⚠️ 2026-10-05 用户实际踩到的坑：旧文件里残留别的对话的测试内容
#   ⇒ 无条件填进输入框 ⇒ 看起来"默认提示词和内置默认没关系"。
# ✅ 修：加一行小字**说明来源**（不清空它 —— 用户上次填的词不该被丢掉）。
if ([string]::IsNullOrWhiteSpace($lastPrompt)) {
    $lastPrompt = $DEFAULT_PROMPT
    $fromFile = $false
} else {
    $fromFile = $true
}
$lastNeg    = if (Test-Path $negPath)    { Get-Content $negPath    -Raw -Encoding UTF8 } else { "" }

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

[System.Windows.Forms.Application]::EnableVisualStyles()

$form = New-Object System.Windows.Forms.Form
$form.Text = "KP 本地出图器  ·  内容只留在你这台机器"
$form.Size = New-Object System.Drawing.Size(820, 620)
$form.StartPosition = "CenterScreen"
$form.BackColor = [System.Drawing.Color]::FromArgb(250, 250, 252)

$fontLabel = New-Object System.Drawing.Font("Microsoft YaHei UI", 9)
$fontTitle = New-Object System.Drawing.Font("Microsoft YaHei UI", 11, [System.Drawing.FontStyle]::Bold)
$fontBox   = New-Object System.Drawing.Font("Consolas", 10)
$fontSmall = New-Object System.Drawing.Font("Microsoft YaHei UI", 8)

# ---- 标题 ----
$lblTitle = New-Object System.Windows.Forms.Label
$lblTitle.Text = "你在这里写的内容不会离开这台电脑"
$lblTitle.Font = $fontTitle
$lblTitle.AutoSize = $true
$lblTitle.Location = New-Object System.Drawing.Point(18, 14)
$form.Controls.Add($lblTitle)

# ---- 提示 ----
$lblTip = New-Object System.Windows.Forms.Label
# ⭐ 明确标注**框里这个值从哪来**（2026-10-05 用户实际踩的坑：
#    旧文件残留别的对话的测试内容 ⇒ 看起来"默认提示词和内置默认没关系"）。
$tipSuffix = if ($fromFile) { "（当前显示的是你上次填的内容）" } else { "（内置默认）" }
$lblTip.Text = "想要什么就写什么（含任何内容）。留空则用「夕阳海滩上的少女」。" + $tipSuffix
$lblTip.Font = $fontSmall
$lblTip.AutoSize = $true
$lblTip.ForeColor = [System.Drawing.Color]::FromArgb(110, 110, 120)
$lblTip.Location = New-Object System.Drawing.Point(20, 44)
$form.Controls.Add($lblTip)

# ---- prompt 输入框 ----
$lblP = New-Object System.Windows.Forms.Label
$lblP.Text = "想画什么："
$lblP.Font = $fontLabel
$lblP.AutoSize = $true
$lblP.Location = New-Object System.Drawing.Point(20, 70)
$form.Controls.Add($lblP)

$txtPrompt = New-Object System.Windows.Forms.TextBox
$txtPrompt.Multiline = $true
$txtPrompt.AcceptsReturn = $true
$txtPrompt.ScrollBars = "Vertical"
$txtPrompt.Font = $fontBox
$txtPrompt.Location = New-Object System.Drawing.Point(18, 92)
$txtPrompt.Size = New-Object System.Drawing.Size(784, 200)
$txtPrompt.Text = $lastPrompt
$form.Controls.Add($txtPrompt)

# ---- 负向提示 ----
$lblN = New-Object System.Windows.Forms.Label
$lblN.Text = "不想出现什么（可留空）："
$lblN.Font = $fontLabel
$lblN.AutoSize = $true
$lblN.Location = New-Object System.Drawing.Point(20, 300)
$form.Controls.Add($lblN)

$txtNeg = New-Object System.Windows.Forms.TextBox
$txtNeg.Multiline = $true
$txtNeg.Font = $fontBox
$txtNeg.Location = New-Object System.Drawing.Point(18, 322)
$txtNeg.Size = New-Object System.Drawing.Size(784, 60)
$txtNeg.Text = $lastNeg
$form.Controls.Add($txtNeg)

# ---- 参数 ----
$lblOpt = New-Object System.Windows.Forms.Label
$lblOpt.Text = "尺寸 / 步数 / 种子 / 4bit"
$lblOpt.Font = $fontLabel
$lblOpt.AutoSize = $true
$lblOpt.Location = New-Object System.Drawing.Point(20, 390)
$form.Controls.Add($lblOpt)

$cSize = New-Object System.Windows.Forms.ComboBox
$cSize.Items.AddRange(@("512", "768", "1024"))
$cSize.Text = "1024"
$cSize.Location = New-Object System.Drawing.Point(110, 386)
$cSize.Size = New-Object System.Drawing.Size(90, 26)
$form.Controls.Add($cSize)

$txtSteps = New-Object System.Windows.Forms.TextBox
$txtSteps.Text = "20"
$txtSteps.Location = New-Object System.Drawing.Point(225, 386)
$txtSteps.Size = New-Object System.Drawing.Size(60, 26)
$form.Controls.Add($txtSteps)

$txtSeed = New-Object System.Windows.Forms.TextBox
$txtSeed.Text = "-1"
$txtSeed.Location = New-Object System.Drawing.Point(320, 386)
$txtSeed.Size = New-Object System.Drawing.Size(80, 26)
$form.Controls.Add($txtSeed)

$chk4bit = New-Object System.Windows.Forms.CheckBox
$chk4bit.Text = "用 4bit 省显存"
$chk4bit.Location = New-Object System.Drawing.Point(420, 387)
$chk4bit.Size = New-Object System.Drawing.Size(150, 26)
$form.Controls.Add($chk4bit)

# ---- 生成按钮 ----
$btn = New-Object System.Windows.Forms.Button
$btn.Text = "生成（Ctrl+Enter）"
$btn.Font = New-Object System.Drawing.Font("Microsoft YaHei UI", 11, [System.Drawing.FontStyle]::Bold)
$btn.Location = New-Object System.Drawing.Point(18, 424)
$btn.Size = New-Object System.Drawing.Size(200, 42)
$btn.BackColor = [System.Drawing.Color]::FromArgb(70, 130, 220)
$btn.ForeColor = [System.Drawing.Color]::White
$form.Controls.Add($btn)

# ---- 状态 / 日志 ----
$txtLog = New-Object System.Windows.Forms.TextBox
$txtLog.Multiline = $true
$txtLog.ReadOnly = $true
$txtLog.ScrollBars = "Vertical"
$txtLog.Font = New-Object System.Drawing.Font("Consolas", 9)
$txtLog.BackColor = [System.Drawing.Color]::FromArgb(30, 32, 38)
$txtLog.ForeColor = [System.Drawing.Color]::FromArgb(220, 220, 220)
$txtLog.Location = New-Object System.Drawing.Point(18, 474)
$txtLog.Size = New-Object System.Drawing.Size(784, 96)
$form.Controls.Add($txtLog)

function Add-Log([string]$m) {
    $txtLog.AppendText("[$([DateTime]::Now.ToString('HH:mm:ss'))] $m`r`n")
    $txtLog.SelectionStart = $txtLog.Text.Length
    $txtLog.ScrollToCaret()
}

# ---- 打开输出文件夹 ----
$btnOpen = New-Object System.Windows.Forms.Button
$btnOpen.Text = "打开输出文件夹"
$btnOpen.Location = New-Object System.Drawing.Point(238, 432)
$btnOpen.Size = New-Object System.Drawing.Size(150, 34)
$form.Controls.Add($btnOpen)

$btnOpen.Add_Click({
    Start-Process explorer.exe $outDir
})

$doGenerate = {
    # ---- 保存内容到本地文件（Agent 不读）----
    $p = $txtPrompt.Text.Trim()
    if ([string]::IsNullOrWhiteSpace($p)) { $p = $DEFAULT_PROMPT }
    $n = $txtNeg.Text.Trim()

    $utf8 = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($promptPath, $p, $utf8)
    [System.IO.File]::WriteAllText($negPath,    $n, $utf8)

    $size  = $cSize.Text
    $steps = $txtSteps.Text.Trim()
    $seed  = $txtSeed.Text.Trim()
    $flag4 = if ($chk4bit.Checked) { "--4bit" } else { "" }

    Add-Log "开始生成… size=$size steps=$steps 4bit=$([bool]$chk4bit.Checked)"
    $btn.Enabled = $false
    $btn.Text = "生成中…"
    [System.Windows.Forms.Application]::DoEvents()

    $args = @("-u", (Join-Path $here "local_gen.py"),
              "--size", $size, "--steps", $steps, "--seed", $seed)
    if ($flag4 -ne "") { $args += $flag4 }

    # ⭐ 同步执行，输出直接进日志框（不弹黑窗，便于看进度）
    $out = & $py @args 2>&1 | Out-String

    # ⚠️ 只把「路径行」和「耗时行」显示出来，**不回显 prompt 内容**
    foreach ($line in ($out -split "`r?`n")) {
        if ($line -match '^\s*$') { continue }
        if ($line -match '🖼|✅ 完成|加载模型|模型就绪|输出目录|⛔|Error|Traceback|error') {
            Add-Log ($line.Trim())
        }
    }

    Add-Log "完成。图片在：$outDir"
    $btn.Enabled = $true
    $btn.Text = "生成（Ctrl+Enter）"
    [System.Windows.Forms.Application]::DoEvents()
}

$btn.Add_Click($doGenerate)

# Ctrl+Enter 快捷键
$form.KeyPreview = $true
$form.Add_KeyDown({
    if ($_.KeyCode -eq [System.Windows.Forms.Keys]::Enter -and $_.Control) {
        & $doGenerate
    }
})

$form.Add_Shown({
    Add-Log "就绪。填内容 → 点生成。Ctrl+Enter 快捷。"
    Add-Log "输出：$outDir"
    $txtPrompt.Focus()
})

[void]$form.ShowDialog()
