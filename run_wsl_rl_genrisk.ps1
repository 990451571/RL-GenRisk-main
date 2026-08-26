param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("healthcheck", "train", "identify")]
    [string]$Command,

    [string[]]$Args = @()
)

$Project = "/mnt/e/Projects/RL-GenRisk-main"
$CondaInit = "~/miniconda3/etc/profile.d/conda.sh"
$TrainLabel = "/mnt/e/codex_file/一阶段/driver_label_protocol/protocol_B/train_driver_genes.csv"
$ValLabel = "/mnt/e/codex_file/一阶段/driver_label_protocol/protocol_B/validation_driver_genes.csv"

function Join-BashArgs {
    param([string[]]$Items)
    return ($Items | ForEach-Object { "'" + ($_ -replace "'", "'\''") + "'" }) -join " "
}

$extra = Join-BashArgs $Args

if ($Command -eq "healthcheck") {
    $inner = "source $CondaInit && conda activate rl_genrisk && cd '$Project' && python scripts/project_healthcheck.py --train-label-path '$TrainLabel' --val-label-path '$ValLabel' $extra"
} elseif ($Command -eq "train") {
    $inner = "source $CondaInit && conda activate rl_genrisk && cd '$Project' && python src/train.py --train_label_path '$TrainLabel' --val_label_path '$ValLabel' $extra"
} elseif ($Command -eq "identify") {
    $inner = "source $CondaInit && conda activate rl_genrisk && cd '$Project' && python src/identify.py $extra"
}

wsl bash -lc $inner
