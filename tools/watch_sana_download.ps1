# ============================================================
#  KokonaPolaris-S4 · G1 靶子下载进度（实时窗口）
#  用法：  双击本文件，或在本目录执行  pwsh -File tools\watch_sana_download.ps1
#  特性：  只读监视 —— **不碰下载进程**，不占 GPU，不影响下载
#  机器：  viim (RTX 5070 Laptop)   /   目标 10.25 GB
#
#  ⚠️ 本脚本刻意**只用 ASCII 字符**：中文 Windows 控制台默认 GBK，
#     框线(═║)与 emoji(⏳✅) 会被渲染成 `?`，看起来像脚本坏了。
# ============================================================
$ErrorActionPreference = 'SilentlyContinue'
$ROOT   = 'D:\kokonapolaris-s4'
$MODELS = Join-Path $ROOT 'models'
$TARGET_BYTES = 10.25GB

function Get-AllFiles {
    Get-ChildItem $MODELS -Recurse -File -Force -ErrorAction SilentlyContinue
}

function Get-Dl {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like '*fetch_sana*' } | Select-Object -First 1
}

# 目标大小（用于显示进度比例；键为文件名尾部）
$EXPECT = @{
    'diffusion_pytorch_model.bf16.safetensors' = 3060MB   # transformer 与 vae 同名，按路径区分
    'diffusion_pytorch_model.int4.safetensors' = 1221MB
    'model.bf16-00001-of-00002.safetensors'    = 2500MB
    'model.bf16-00002-of-00002.safetensors'    = 2500MB
}

$prevBytes = $null          # ⚠️ 必须是 $null 而不是 0 —— 否则第一轮会把「已下载总量」当成增量
$prevTime  = Get-Date
$speedHist = New-Object System.Collections.ArrayList
$firstRound = $true

while ($true) {
    $now   = Get-Date
    $files = Get-AllFiles
    $bytes = ($files | Measure-Object -Property Length -Sum).Sum
    if (-not $bytes) { $bytes = 0 }

    $dt  = ($now - $prevTime).TotalSeconds
    $spd = if ($null -ne $prevBytes -and $dt -gt 0) { ($bytes - $prevBytes) / 1MB / $dt } else { $null }
    if ($null -ne $spd) { [void]$speedHist.Add($spd); if ($speedHist.Count -gt 15) { $speedHist.RemoveAt(0) } }
    $avg = if ($speedHist.Count) { ($speedHist | Measure-Object -Average).Average } else { $null }
    $prevBytes = $bytes; $prevTime = $now

    $pct  = [math]::Min(100, [math]::Round($bytes / $TARGET_BYTES * 100, 1))
    $barW = 44
    $fill = [int]($pct / 100 * $barW)
    $bar  = ('#' * $fill) + ('-' * ($barW - $fill))

    Clear-Host
    Write-Host ''
    Write-Host '  ==================================================================' -ForegroundColor Cyan
    Write-Host '    KokonaPolaris-S4  |  G1 靶子下载进度 (Sana 1.6B)  |  机器 viim' -ForegroundColor Cyan
    Write-Host '  ==================================================================' -ForegroundColor Cyan
    Write-Host ''
    Write-Host "    $($now.ToString('yyyy-MM-dd HH:mm:ss'))      目标 10.25 GB" -ForegroundColor DarkGray
    Write-Host ''
    Write-Host "    [$bar] $pct%" -ForegroundColor Green
    Write-Host ''
    Write-Host ("    已下载      {0:N2} GB" -f ($bytes / 1GB))
    if ($null -eq $spd) {
        Write-Host '    当前速度    （正在建立基线，下轮显示）' -ForegroundColor DarkGray
    } else {
        Write-Host ("    当前速度    {0:N2} MB/s      近 15 窗口均值 {1:N2} MB/s" -f $spd, $avg) -ForegroundColor Yellow
        if ($avg -gt 0.05 -and $bytes -lt $TARGET_BYTES) {
            $left = ($TARGET_BYTES - $bytes) / 1MB / $avg
            Write-Host ("    预计剩余    {0:N0} 分 {1:N0} 秒" -f [math]::Floor($left / 60), ($left % 60))
        }
    }
    Write-Host ''

    # ---- 下载进程 ----
    $p = Get-Dl
    Write-Host '    -- 下载进程 -------------------------------------------------'
    if ($p) {
        $run = ((Get-Date) - $p.CreationDate).ToString('hh\:mm\:ss')
        $cpu = [math]::Round($p.UserModeTime / 10000000, 1)
        Write-Host "      [运行中]  PID $($p.ProcessId)   已运行 $run   用户态CPU $cpu s" -ForegroundColor Green
    } else {
        Write-Host '      [未运行]  下载进程不存在（已完成 / 已终止）' -ForegroundColor Red
    }
    Write-Host ''

    # ---- 文件明细（含 .part）----
    Write-Host '    -- 文件明细 -------------------------------------------------'
    $list = $files | Where-Object { $_.Length -gt 100KB } | Sort-Object Length -Descending
    foreach ($f in $list) {
        $rel  = $f.FullName.Replace("$MODELS\", '')
        $tail = $rel.Split('\')[-1]
        $isPart = $f.Name -like '*.part'
        $tag  = if ($isPart) { '[>>]' } else { '[OK]' }
        $col  = if ($isPart) { 'Yellow' } else { 'Green' }
        $line = "      $tag {0,9:N0} MB  {1}" -f ($f.Length / 1MB), $rel
        if ($EXPECT.ContainsKey($tail)) {
            $line += ("  (目标 {0:N0} MB)" -f ($EXPECT[$tail] / 1MB))
        }
        Write-Host $line -ForegroundColor $col
    }
    if (-not $list) { Write-Host '      （暂无可显示的文件）' -ForegroundColor DarkGray }
    Write-Host ''

    # ---- 提示 ----
    $parts = $files | Where-Object { $_.Name -like '*.part' }
    if ($parts) {
        $newest = ($parts | Sort-Object LastWriteTime -Descending)[0]
        $idle = ((Get-Date) - $newest.LastWriteTime).TotalSeconds
        if ($idle -gt 90) {
            Write-Host ("      [注意] 最大分片已 {0:N0} 秒无写入 —— 下载可能停滞（看门狗 90s 会重连）" -f $idle) -ForegroundColor Red
        } else {
            Write-Host ("      [进行中] 最新分片 {0:N0} 秒前有写入" -f $idle) -ForegroundColor Yellow
        }
    }
    if (-not $p -and $bytes -ge $TARGET_BYTES) {
        Write-Host '      [完成] 下载已完成，可以关闭本窗口。' -ForegroundColor Green
    }
    Write-Host ''
    Write-Host '    -------------------------------------------------------------' -ForegroundColor DarkGray
    Write-Host '    只读监视，不干扰下载。Ctrl+C 或关闭窗口即退出。' -ForegroundColor DarkGray

    $firstRound = $false
    Start-Sleep -Seconds 3
}
