param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("healthcheck", "train", "identify")]
    [string]$Command,

    [string[]]$Args = @()
)

$Project = "/mnt/e/Projects/RL-GenRisk-main"
$CondaInit = "~/miniconda3/etc/profile.d/conda.sh"
# Train/Validation 标签不在此硬编码：
# src/project_paths.py 优先读 config/local_paths.yaml（当前指向项目内 experiments/protocol_B）。
# 如需覆盖，通过 -Args 传入 --train_label_path/--val_label_path。
# checkpoint 不硬编码时间戳路径：自动取 outputs 下最新的 checkpoint_best.pt。
# 脚本写入临时文件（LF 换行）后交给 wsl bash 执行，避免 PowerShell 管道 CRLF 与参数引号问题。

function Join-BashArgs {
    param([string[]]$Items)
    return ($Items | ForEach-Object { "'" + ($_ -replace "'", "'\''") + "'" }) -join " "
}

function Invoke-WslBash {
    param([string]$Script)
    $tmp = Join-Path $env:TEMP ("rl_genrisk_cmd_" + [System.Guid]::NewGuid().ToString("N") + ".sh")
    [System.IO.File]::WriteAllText($tmp, $Script.TrimEnd() + "`n")
    $wslPath = '/mnt/' + $tmp[0].ToString().ToLower() + $tmp.Substring(2).Replace('\', '/')
    wsl bash $wslPath
    Remove-Item $tmp -ErrorAction SilentlyContinue
}

$extra = Join-BashArgs $Args
$common = "source $CondaInit && conda activate rl_genrisk && cd '$Project'"
$ckpt = "CKPT=`$(ls -t outputs/*/hybrid6_raw/*/checkpoint_best.pt outputs/*/*/*/checkpoint_best.pt 2>/dev/null | head -1)"

if ($Command -eq "healthcheck") {
    $script = "$common && $ckpt; python scripts/project_healthcheck.py --checkpoint `"`$CKPT`" $extra"
} elseif ($Command -eq "train") {
    $script = "$common && python src/train.py $extra"
} elseif ($Command -eq "identify") {
    $script = "$common && $ckpt; python src/identify.py --checkpoint `"`$CKPT`" $extra"
}

Invoke-WslBash $script
