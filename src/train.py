import argparse
import csv
import hashlib
import json
import logging
import math
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _preconfigure_process_from_argv():
    """在导入 PyTorch 前设置进程级随机种子、确定性和设备环境变量。"""
    seed = "42"
    device = "auto"
    for idx, item in enumerate(sys.argv):
        if item == "--seed" and idx + 1 < len(sys.argv):
            seed = sys.argv[idx + 1]
        elif item.startswith("--seed="):
            seed = item.split("=", 1)[1]
        elif item == "--device" and idx + 1 < len(sys.argv):
            device = sys.argv[idx + 1].lower()
        elif item.startswith("--device="):
            device = item.split("=", 1)[1].lower()
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


_preconfigure_process_from_argv()

import numpy as np
import pandas as pd
import torch

import inputall
import lowfreq_evidence
from project_paths import train_label_path as configured_train_label_path
from project_paths import val_label_path as configured_val_label_path
from DQN import DeepQNetwork


SCRIPT_DIR = Path(__file__).resolve().parent
if (SCRIPT_DIR / "data").exists():
    REPO_DIR = SCRIPT_DIR
elif (SCRIPT_DIR.parent / "data").exists():
    REPO_DIR = SCRIPT_DIR.parent
else:
    REPO_DIR = SCRIPT_DIR.parent
SRC_DIR = SCRIPT_DIR
DATA_DIR = REPO_DIR / "data"
DEFAULT_OUTPUT_ROOT = REPO_DIR / "outputs" / "hybrid6_raw_training"
DEFAULT_PROTOCOL_B = DATA_DIR / "driver_label_protocol" / "protocol_B"
DEFAULT_TRAIN_LABEL_PATH = configured_train_label_path()
DEFAULT_VAL_LABEL_PATH = configured_val_label_path()
DEFAULT_MULTIOMICS_3OMICS_PATH = DATA_DIR / "processed" / "KIRC_multiomics_3omics.csv"
DEFAULT_MULTIOMICS_4OMICS_PATH = DATA_DIR / "processed" / "KIRC_multiomics_4omics.csv"
DEFAULT_CNV_MISSING_GENE_PATH = DATA_DIR / "processed" / "cnv_kirc" / "multiomics_genes_missing_cnv.csv"
FEATURE_MODE = "hybrid6_raw"
FEATURE_COLUMNS_BY_MODE = {
    mode: inputall.canonicalize_feature_columns(columns)
    for mode, columns in inputall.FEATURE_MODE_COLUMNS.items()
}
FEATURE_DIM_BY_MODE = {mode: len(columns) for mode, columns in FEATURE_COLUMNS_BY_MODE.items()}
FOUR_OMICS_FEATURE_MODES = {"multiomics4_raw", "hybrid7_raw", "multiomics4_zscore", "hybrid7_zscore"}
Z_SCORE_FEATURE_MODES = {"original3_zscore", "hybrid6_zscore", "multiomics4_zscore", "hybrid7_zscore"}
K_VALUES = [20, 50, 100, 150]
REWARD_MODES = [
    "legacy",
    "multiomics_mutation",
    "multiomics_no_mutation",
    "multiomics_lowfreq",
    "label_conditioned_lowfreq",
    "lowfreq_unlabeled_evidence",
    "lowfreq_unlabeled_evidence_v2",
    "lowfreq_unlabeled_no_network",
    "lowfreq_unlabeled_no_omics",
    "lowfreq_unlabeled_no_rarity",
]
LOWFREQ_UNLABELED_REWARD_MODES = {
    "lowfreq_unlabeled_evidence",
    "lowfreq_unlabeled_evidence_v2",
    "lowfreq_unlabeled_no_network",
    "lowfreq_unlabeled_no_omics",
    "lowfreq_unlabeled_no_rarity",
}
REWARD_COMPONENT_KEYS = [
    "reward_total",
    "reward_legacy",
    "reward_train_label",
    "reward_mutation",
    "reward_expression",
    "reward_methylation",
    "reward_lowfreq",
    "reward_evidence_bonus",
    "reward_penalty",
]
TERMINAL_REASON_CN = {
    "unknown": "未知",
    "no_legal_action": "无合法动作",
    "selection_budget": "达到选择预算",
    "max_steps_truncation": "达到最大步数并截断",
}


ACTION_REWARD_LOG_FIELDNAMES = [
    "episode",
    "step",
    "action_index",
    "Gene",
    "is_train_driver",
    "mutation_count",
    "mutation_frequency",
    "MutationFrequencyPct",
    "MutationGroup",
    "MutationRarityScore",
    "ExpressionSupport",
    "MethylationSupport",
    "CNVFunctionalSupport",
    "NonMutationOmicsSupport",
    "DegreeCorrectedNetworkSupport",
    "DegreeCorrectedNetworkSupportV2",
    "LowFrequencyEvidenceScore",
    "LowFrequencyEvidenceScoreV2",
    "base_reward",
    "evidence_bonus",
    "final_reward",
    "done",
    "terminal_reason",
    "reward_mode",
]


class TeeStream:
    def __init__(self, original, log_file):
        self.original = original
        self.log_file = log_file

    def write(self, data):
        self.original.write(data)
        self.log_file.write(data)
        self.flush()

    def flush(self):
        self.original.flush()
        self.log_file.flush()


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def resolve_path(path, base=DATA_DIR):
    if path is None:
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return base / candidate


def normalize_feature_mode(feature_mode):
    mode = str(feature_mode or FEATURE_MODE).strip().lower()
    if mode not in FEATURE_COLUMNS_BY_MODE:
        raise ValueError(f"Unsupported feature_mode: {feature_mode!r}")
    return mode


def feature_columns_for_mode(feature_mode):
    return list(FEATURE_COLUMNS_BY_MODE[normalize_feature_mode(feature_mode)])


def feature_dim_for_mode(feature_mode):
    return int(FEATURE_DIM_BY_MODE[normalize_feature_mode(feature_mode)])


def default_multiomics_path_for_mode(feature_mode):
    if normalize_feature_mode(feature_mode) in FOUR_OMICS_FEATURE_MODES:
        return DEFAULT_MULTIOMICS_4OMICS_PATH
    return DEFAULT_MULTIOMICS_3OMICS_PATH


def resolve_multiomics_path_for_args(args):
    if args.multiomics_feature_path is None:
        return default_multiomics_path_for_mode(args.feature_mode)
    return resolve_path(args.multiomics_feature_path)


def sha256_text(items):
    digest = hashlib.sha256()
    for item in items:
        digest.update(str(item).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def q_parameter_count(model):
    return sum(param.numel() for param in model.parameters())


def get_git_commit():
    """读取当前 Git commit，失败时返回 None。"""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_DIR,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def get_git_diff_sha256():
    """计算当前 Git diff 的哈希，并隐藏无关的换行符警告。"""
    try:
        diff = subprocess.check_output(
            ["git", "diff"],
            cwd=REPO_DIR,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        diff = ""

    return hashlib.sha256(
        diff.encode("utf-8")
    ).hexdigest()


def load_driver_label_set(path):
    labels = []
    invalid_count = 0
    seen = set()
    path = resolve_path(path)
    with Path(path).open("r", encoding="utf-8", errors="replace", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        delimiter = "," if "," in sample else ("\t" if "\t" in sample else None)
        rows = csv.reader(handle, delimiter=delimiter) if delimiter else (line.split() for line in handle)
        for row_index, row in enumerate(rows):
            if not row:
                invalid_count += 1
                continue
            gene = inputall.clean_gene_symbol(row[0])
            if row_index == 0 and gene in {"GENE", "GENE_SYMBOL", "GENE SYMBOL"}:
                continue
            if gene is None:
                invalid_count += 1
                continue
            if gene not in seen:
                seen.add(gene)
                labels.append(gene)
    return labels, Path(path), invalid_count


def ensure_existing_label_path(path, label_name):
    resolved = resolve_path(path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"{label_name} label file not found: {resolved}. "
            "Pass an explicit --train_label_path/--val_label_path, set "
            "RL_GENRISK_TRAIN_LABEL_PATH/RL_GENRISK_VAL_LABEL_PATH, or create "
            "config/local_paths.yaml from config/local_paths.example.yaml. "
            "Test and external holdout labels are never used as fallbacks."
        )
    if not resolved.is_file():
        raise FileNotFoundError(f"{label_name} label path is not a file: {resolved}")
    return resolved


def parse_args():
    parser = argparse.ArgumentParser(
        description="仅使用固定 hybrid6_raw 特征模式训练 RL-GenRisk。"
    )
    parser.add_argument("--original_feature_path", default=None, help="可选：原始三维特征 CSV 路径；不提供时由 PPI、权重和患者覆盖动态构建。")
    parser.add_argument(
        "--feature-mode",
        "--feature_mode",
        dest="feature_mode",
        choices=sorted(FEATURE_COLUMNS_BY_MODE),
        default=FEATURE_MODE,
        help="节点输入特征模式；默认保持 hybrid6_raw。",
    )
    parser.add_argument(
        "--multiomics-feature-path",
        "--multiomics_feature_path",
        dest="multiomics_feature_path",
        default=None,
        help="多组学特征 CSV 路径；未提供时根据 feature_mode 自动选择 3omics 或 4omics 文件。",
    )
    parser.add_argument(
        "--cnv-missing-gene-path",
        "--cnv_missing_gene_path",
        dest="cnv_missing_gene_path",
        default=str(DEFAULT_CNV_MISSING_GENE_PATH),
        help="CNV 未匹配后补零基因清单，仅用于 CNV 缺失感知 z-score 和审计。",
    )
    parser.add_argument("--ppi_path", default=str(DATA_DIR / "HPRD.txt"), help="PPI 网络文件路径。")
    parser.add_argument("--mutation_path", default=str(DATA_DIR / "KIRC.txt"), help="KIRC 突变/患者基因文件路径。")
    parser.add_argument("--weight_path", default=str(DATA_DIR / "weights.txt"), help="基因权重文件路径。")
    parser.add_argument("--train_label_path", default=str(DEFAULT_TRAIN_LABEL_PATH), help="训练集 driver gene 标签路径。")
    parser.add_argument("--val_label_path", default=str(DEFAULT_VAL_LABEL_PATH), help="验证集 driver gene 标签路径。")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_ROOT), help="训练结果根目录；程序会自动创建带随机种子和时间戳的子目录。")
    parser.add_argument("--seed", type=int, default=0, help="随机种子。")
    parser.add_argument("--max_episodes", type=int, default=30, help="最大训练轮数。")
    parser.add_argument("--max_steps", type=int, default=160, help="每轮最大动作步数。")
    parser.add_argument("--warmup_steps", type=int, default=128, help="经验回放池达到该数量后才开始学习。")
    parser.add_argument("--batch_size", type=int, default=128, help="每次学习采样的批量大小。")
    parser.add_argument("--buffer_size", type=int, default=2048, help="优先经验回放池容量。")
    parser.add_argument("--gamma", type=float, default=0.95, help="折扣因子 gamma。")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率。")
    parser.add_argument("--tau", type=float, default=0.001, help="目标网络软更新系数 tau。")
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="初始随机探索概率。")
    parser.add_argument("--epsilon_end", type=float, default=0.15, help="最低随机探索概率。")
    parser.add_argument("--epsilon_decay", type=float, default=600.0, help="按全局步数计算的 epsilon 衰减尺度。")
    parser.add_argument("--per_alpha", type=float, default=0.2, help="PER 优先级指数 alpha。")
    parser.add_argument("--per_beta_start", type=float, default=0.1, help="PER 重要性采样权重 beta 初始值。")
    parser.add_argument("--per_beta_frames", type=int, default=2_000_000, help="beta 递增到 1.0 所需的采样次数。")
    parser.add_argument("--per_eps", type=float, default=1e-5, help="PER priority 的最小稳定项。")
    parser.add_argument("--val_interval", type=int, default=1, help="每隔多少轮执行一次验证。")
    parser.add_argument("--topk", type=int, default=150, help="每轮选择预算及主要评价 K 值。")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="训练设备：auto 自动选择，cuda 使用 GPU，cpu 使用 CPU。")
    parser.add_argument("--resume", default=None, help="可选：继续训练所使用的 checkpoint 路径。")
    parser.add_argument("--cancer", default="KIRC", help="癌种名称，默认 KIRC。")
    parser.add_argument("--embedding_size", type=int, default=64, help="图节点嵌入维度。")
    parser.add_argument("--score_alpha", type=float, default=0.5, help="奖励函数中权重得分与患者覆盖得分的平衡系数。")
    parser.add_argument("--gradient_clip", type=float, default=1.0, help="梯度范数裁剪阈值。")
    parser.add_argument("--reward-mode", choices=REWARD_MODES, default="legacy", help="Stage 2 reward mode；默认 legacy 以保持兼容。")
    parser.add_argument("--multiomics-mutation-weight", type=float, default=0.08, help="multiomics_mutation 模式下 Mutation 辅助 reward 权重。")
    parser.add_argument("--multiomics-expression-weight", type=float, default=0.06, help="multiomics_mutation 模式下 Expression 辅助 reward 权重。")
    parser.add_argument("--multiomics-methylation-weight", type=float, default=0.06, help="multiomics_mutation 模式下 Methylation 辅助 reward 权重。")
    parser.add_argument("--no-mutation-expression-weight", type=float, default=0.08, help="multiomics_no_mutation 模式下 Expression 辅助 reward 权重。")
    parser.add_argument("--no-mutation-methylation-weight", type=float, default=0.08, help="multiomics_no_mutation 模式下 Methylation 辅助 reward 权重。")
    parser.add_argument("--lowfreq-expression-weight", type=float, default=0.05, help="multiomics_lowfreq 模式下 Expression 辅助 reward 权重。")
    parser.add_argument("--lowfreq-methylation-weight", type=float, default=0.05, help="multiomics_lowfreq 模式下 Methylation 辅助 reward 权重。")
    parser.add_argument("--lowfreq-bonus-cap", type=float, default=0.20, help="multiomics_lowfreq 中 Train-driver rarity bonus 上限。")
    parser.add_argument("--lowfreq-evidence-path", default=None, help="Frozen label-independent low-frequency evidence table path.")
    parser.add_argument("--lowfreq-unlabeled-bonus-scale", type=float, default=None, help="Frozen lowfreq_unlabeled_* evidence bonus scale.")
    parser.add_argument("--lowfreq-unlabeled-bonus-cap", type=float, default=None, help="Frozen lowfreq_unlabeled_* evidence bonus cap.")
    return parser.parse_args()




def validate_training_args(args):
    args.feature_mode = normalize_feature_mode(args.feature_mode)
    args.multiomics_feature_path = str(resolve_multiomics_path_for_args(args))
    args.cnv_missing_gene_path = str(resolve_path(args.cnv_missing_gene_path))
    positive_ints = {
        "max_episodes": args.max_episodes,
        "max_steps": args.max_steps,
        "warmup_steps": args.warmup_steps,
        "batch_size": args.batch_size,
        "buffer_size": args.buffer_size,
        "topk": args.topk,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            raise ValueError(f"参数 --{name} 必须为正数，当前值为 {value}。")
    if args.batch_size > args.buffer_size:
        raise ValueError("参数 --batch_size 不能大于 --buffer_size。")
    if args.warmup_steps > args.buffer_size:
        raise ValueError("参数 --warmup_steps 不能大于 --buffer_size。")
    if not 0.0 <= args.gamma <= 1.0:
        raise ValueError(f"参数 --gamma 必须位于 [0, 1]，当前值为 {args.gamma}。")
    if not 0.0 < args.tau <= 1.0:
        raise ValueError(f"参数 --tau 必须位于 (0, 1]，当前值为 {args.tau}。")
    if not 0.0 <= args.epsilon_end <= args.epsilon_start <= 1.0:
        raise ValueError(
            "必须满足 0 <= epsilon_end <= epsilon_start <= 1；"
            f"当前 epsilon_start={args.epsilon_start}，epsilon_end={args.epsilon_end}。"
        )
    if args.gradient_clip <= 0:
        raise ValueError("参数 --gradient_clip 必须为正数。")
    ensure_existing_label_path(args.train_label_path, "Train")
    ensure_existing_label_path(args.val_label_path, "Validation")
    reward_weights = {
        "multiomics_mutation_weight": args.multiomics_mutation_weight,
        "multiomics_expression_weight": args.multiomics_expression_weight,
        "multiomics_methylation_weight": args.multiomics_methylation_weight,
        "no_mutation_expression_weight": args.no_mutation_expression_weight,
        "no_mutation_methylation_weight": args.no_mutation_methylation_weight,
        "lowfreq_expression_weight": args.lowfreq_expression_weight,
        "lowfreq_methylation_weight": args.lowfreq_methylation_weight,
        "lowfreq_bonus_cap": args.lowfreq_bonus_cap,
    }
    if args.lowfreq_unlabeled_bonus_scale is not None:
        reward_weights["lowfreq_unlabeled_bonus_scale"] = args.lowfreq_unlabeled_bonus_scale
    if args.lowfreq_unlabeled_bonus_cap is not None:
        reward_weights["lowfreq_unlabeled_bonus_cap"] = args.lowfreq_unlabeled_bonus_cap
    for name, value in reward_weights.items():
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"reward weight {name} must be finite and non-negative, got {value}.")
    if args.reward_mode == "multiomics_no_mutation" and args.multiomics_mutation_weight != 0.08:
        logging.warning("multiomics_no_mutation ignores --multiomics-mutation-weight; mutation reward component remains 0.")
    if args.reward_mode in LOWFREQ_UNLABELED_REWARD_MODES:
        if not args.lowfreq_evidence_path:
            raise ValueError(f"{args.reward_mode} requires --lowfreq-evidence-path.")
        if args.lowfreq_unlabeled_bonus_scale is None or args.lowfreq_unlabeled_bonus_cap is None:
            raise ValueError(
                f"{args.reward_mode} requires frozen --lowfreq-unlabeled-bonus-scale "
                "and --lowfreq-unlabeled-bonus-cap."
            )
        evidence_path = resolve_path(args.lowfreq_evidence_path, base=REPO_DIR)
        if not evidence_path.exists():
            raise FileNotFoundError(f"low-frequency evidence table not found: {evidence_path}")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as exc:
        logging.warning("无法启用严格确定性算法，将继续运行：%s", exc)




def setup_logger(run_dir):
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path = run_dir / "console.log"
    log_file = log_path.open("w", encoding="utf-8")
    sys.stdout = TeeStream(sys.stdout, log_file)
    sys.stderr = TeeStream(sys.stderr, log_file)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return log_path


def make_run_dir(output_dir, seed, feature_mode=FEATURE_MODE):
    root = Path(output_dir)
    if not root.is_absolute():
        root = REPO_DIR / root
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    feature_mode = normalize_feature_mode(feature_mode)
    run_dir = root / feature_mode / f"seed_{seed}_{timestamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = root / feature_mode / f"seed_{seed}_{timestamp}_{suffix:02d}"
        suffix += 1
    return run_dir


def choose_device(requested):
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("已指定 --device cuda，但当前环境无法使用 CUDA。")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate_no_test_path(args):
    forbidden_tokens = [
        "test_driver_genes",
        "low_frequency_holdout",
        "external_holdout",
        "02_external_holdout",
    ]
    for name, value in vars(args).items():
        lowered = str(value).lower() if value is not None else ""
        if any(token in lowered for token in forbidden_tokens):
            raise ValueError(f"Refusing forbidden Test/external-holdout path in argument {name}: {value}")
        if value is not None and "test_driver_genes" in str(value).lower():
            raise ValueError(f"为防止测试集泄漏，拒绝通过参数 {name} 读取测试标签：{value}")


def validate_gene_order(gene_name):
    cleaned = [inputall.clean_gene_symbol(gene) for gene in gene_name]
    invalid_count = sum(gene is None for gene in cleaned)
    if invalid_count:
        raise ValueError(f"PPI 基因列表中存在 {invalid_count} 个无效基因。")
    duplicates = pd.Series(cleaned).duplicated()
    if duplicates.any():
        examples = pd.Series(cleaned)[duplicates].head(10).tolist()
        raise ValueError(f"PPI 基因列表中存在 {int(duplicates.sum())} 个重复基因，示例：{examples}")
    if len(cleaned) != 9039:
        raise ValueError(f"预期 PPI 节点数为 9039，实际为 {len(cleaned)}。")
    return cleaned


def load_multiomics_three_columns_strict(path, gene_name):
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"未找到多组学特征文件：{path}")
    df = pd.read_csv(path)
    required = ["Gene", "Mutation", "Expression", "Methylation"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"多组学文件缺少必需列：{missing}")
    raw_gene_count = len(df)
    df = df[required].copy()
    df["Gene"] = df["Gene"].map(inputall.clean_gene_symbol)
    invalid_gene_count = int(df["Gene"].isna().sum())
    df = df[df["Gene"].notna()].copy()
    duplicated_mask = df["Gene"].duplicated(keep=False)
    duplicate_count = int(df.loc[duplicated_mask, "Gene"].nunique())
    if duplicate_count:
        examples = sorted(df.loc[duplicated_mask, "Gene"].dropna().unique().tolist())[:10]
        raise ValueError(
            f"Multi-omics feature file contains duplicate Gene values after cleaning; "
            f"duplicate_gene_count={duplicate_count}; examples={examples}; "
            "refusing silent aggregation."
        )
    for column in ["Mutation", "Expression", "Methylation"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if df[["Mutation", "Expression", "Methylation"]].isna().any().any():
        raise ValueError("多组学数值列转换后包含 NaN。")
    values = df[["Mutation", "Expression", "Methylation"]].to_numpy(dtype=np.float32)
    if np.isnan(values).any() or np.isinf(values).any():
        raise ValueError("多组学数值列包含 NaN 或 Inf。")
    indexed = df.set_index("Gene")
    features = np.zeros((len(gene_name), 3), dtype=np.float32)
    matched = 0
    for idx, gene in enumerate(gene_name):
        if gene in indexed.index:
            features[idx] = indexed.loc[gene, ["Mutation", "Expression", "Methylation"]].to_numpy(dtype=np.float32)
            matched += 1
    report = {
        "path": str(path),
        "raw_rows": raw_gene_count,
        "invalid_gene_count": invalid_gene_count,
        "unique_genes_after_cleaning": int(len(df)),
        "duplicate_gene_count_after_cleaning": duplicate_count,
        "duplicate_gene_count_before_groupby": duplicate_count,
        "ppi_nodes": int(len(gene_name)),
        "matched_genes": int(matched),
        "zero_filled_ppi_genes": int(len(gene_name) - matched),
        "extra_multiomics_genes": int(len(set(df["Gene"]) - set(gene_name))),
    }
    return features, report


def load_original_three_columns_features(net, weights, gene_name, gene_final, original_feature_path=None):
    cleaned_gene_name = validate_gene_order(gene_name)
    if original_feature_path:
        original_path = resolve_path(original_feature_path)
        if not original_path.exists():
            raise FileNotFoundError(f"Missing original3 feature file: {original_path}")
        original_df = pd.read_csv(original_path)
        required_original = ["Gene", "Degree", "WeightValue", "PatientCoverageCount"]
        missing_original = [column for column in required_original if column not in original_df.columns]
        if missing_original:
            raise ValueError(f"Original feature file missing columns: {missing_original}")
        original_df = original_df[required_original].copy()
        original_df["Gene"] = original_df["Gene"].map(inputall.clean_gene_symbol)
        if original_df["Gene"].isna().any():
            raise ValueError("Original feature file contains invalid Gene values.")
        duplicate_original = int(original_df["Gene"].duplicated().sum())
        if duplicate_original:
            examples = original_df.loc[original_df["Gene"].duplicated(), "Gene"].head(10).tolist()
            raise ValueError(f"Original feature file contains duplicate genes: {duplicate_original}; examples={examples}")
        for column in required_original[1:]:
            original_df[column] = pd.to_numeric(original_df[column], errors="coerce")
        if original_df[required_original[1:]].isna().any().any():
            raise ValueError("Original feature columns contain non-numeric values or NaN.")
        indexed_original = original_df.set_index("Gene")
        missing_ppi_original = [gene for gene in cleaned_gene_name if gene not in indexed_original.index]
        if missing_ppi_original:
            raise ValueError(
                f"Original feature file is missing {len(missing_ppi_original)} PPI genes; "
                f"examples={missing_ppi_original[:10]}"
            )
        original = indexed_original.loc[
            cleaned_gene_name,
            ["Degree", "WeightValue", "PatientCoverageCount"],
        ].to_numpy(dtype=np.float32)
        source = str(original_path)
        extra_count = int(len(set(indexed_original.index) - set(cleaned_gene_name)))
    else:
        original = inputall.build_original_node_features_raw(
            net,
            weights,
            cleaned_gene_name,
            gene_final,
        )
        source = "inputall.build_original_node_features_raw(PPI degree, weights, patient coverage)"
        extra_count = 0
    if original.shape != (9039, 3):
        raise ValueError(f"original3 feature matrix must have shape (9039, 3), got {original.shape}.")
    return original.astype(np.float32), source, extra_count


def load_node_features_by_mode(
    net,
    weights,
    gene_name,
    gene_final,
    args,
    run_dir,
    normalization_metadata=None,
):
    feature_mode = normalize_feature_mode(args.feature_mode)
    feature_columns = feature_columns_for_mode(feature_mode)
    expected_dim = feature_dim_for_mode(feature_mode)
    cleaned_gene_name = validate_gene_order(gene_name)
    uses_original = feature_mode.startswith("original3") or feature_mode.startswith("hybrid")
    uses_multiomics = feature_mode.startswith("multiomics") or feature_mode.startswith("hybrid")
    needs_cnv = feature_mode in FOUR_OMICS_FEATURE_MODES

    original = None
    original_source = None
    original_extra_count = 0
    if uses_original:
        original, original_source, original_extra_count = load_original_three_columns_features(
            net,
            weights,
            cleaned_gene_name,
            gene_final,
            args.original_feature_path,
        )

    multiomics = None
    multiomics_report = {}
    multiomics_path = resolve_multiomics_path_for_args(args)
    if uses_multiomics:
        if needs_cnv:
            multiomics, multiomics_report = inputall.load_multiomics_features_for_columns(
                multiomics_path,
                cleaned_gene_name,
                ["Mutation", "Expression", "Methylation", "CNV"],
                cnv_missing_gene_path=args.cnv_missing_gene_path,
            )
        else:
            multiomics, multiomics_report = load_multiomics_three_columns_strict(
                multiomics_path,
                cleaned_gene_name,
            )

    cnv_normalization_metadata = None
    if feature_mode in {"multiomics4_zscore", "hybrid7_zscore"} and normalization_metadata is None:
        cnv_normalization_metadata = inputall.compute_full_multiomics_cnv_normalization_metadata(
            multiomics_path,
            args.cnv_missing_gene_path,
        )

    norm = {"method": "none", "feature_names": feature_columns}
    if feature_mode == "original3_raw":
        node_features = original
    elif feature_mode == "original3_zscore":
        node_features, norm = inputall._zscore_node_features(
            original,
            feature_columns,
            normalization_metadata,
        )
    elif feature_mode == "multiomics3_raw":
        node_features = multiomics
    elif feature_mode == "hybrid6_raw":
        node_features = np.concatenate([original, multiomics], axis=1).astype(np.float32)
    elif feature_mode == "hybrid6_zscore":
        raw = np.concatenate([original, multiomics], axis=1).astype(np.float32)
        node_features, norm = inputall._zscore_node_features(
            raw,
            feature_columns,
            normalization_metadata,
        )
    elif feature_mode == "multiomics4_raw":
        node_features = multiomics
    elif feature_mode == "hybrid7_raw":
        node_features = np.concatenate([original, multiomics], axis=1).astype(np.float32)
    elif feature_mode == "multiomics4_zscore":
        node_features, norm = inputall._zscore_node_features_with_cnv_missing(
            multiomics,
            feature_columns,
            multiomics_report["_cnv_observed_mask"],
            normalization_metadata,
            cnv_normalization_metadata,
        )
    elif feature_mode == "hybrid7_zscore":
        raw = np.concatenate([original, multiomics], axis=1).astype(np.float32)
        node_features, norm = inputall._zscore_node_features_with_cnv_missing(
            raw,
            feature_columns,
            multiomics_report["_cnv_observed_mask"],
            normalization_metadata,
            cnv_normalization_metadata,
        )
    else:
        raise ValueError(f"Unsupported feature_mode: {feature_mode}")

    node_features = np.asarray(node_features, dtype=np.float32)
    if node_features.shape != (9039, expected_dim):
        raise ValueError(f"{feature_mode} feature matrix must have shape (9039, {expected_dim}), got {node_features.shape}.")
    nan_count = int(np.isnan(node_features).sum())
    inf_count = int(np.isinf(node_features).sum())
    if nan_count or inf_count:
        raise ValueError(f"{feature_mode} contains non-finite values: nan={nan_count}, inf={inf_count}")

    public_multiomics_report = {
        key: value
        for key, value in (multiomics_report or {}).items()
        if not str(key).startswith("_")
    }
    report = {
        "feature_mode": feature_mode,
        "feature_columns": feature_columns,
        "feature_dim": expected_dim,
        "standardize": feature_mode in Z_SCORE_FEATURE_MODES,
        "normalization_metadata": norm,
        "original_feature_source": original_source,
        "original_feature_path_argument": args.original_feature_path,
        "original_extra_gene_count": original_extra_count,
        "multiomics_feature_path": str(multiomics_path) if uses_multiomics else None,
        "multiomics_report": public_multiomics_report,
        "ppi_node_count": len(cleaned_gene_name),
        "original_feature_match_count": len(cleaned_gene_name) if uses_original else 0,
        "multiomics_feature_match_count": int(public_multiomics_report.get("matched_genes", 0)),
        "full_missing_hprd_count": int(public_multiomics_report.get("zero_filled_ppi_genes", 0)),
        "cnv_missing_hprd_count": int(public_multiomics_report.get("cnv_missing_hprd_count", 0)),
        "node_features_shape": list(node_features.shape),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "gene_order_matches_ppi": True,
    }
    (run_dir / "feature_alignment_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logging.info("Feature mode: %s", feature_mode)
    logging.info("Feature columns: %s", feature_columns)
    logging.info("PPI node count: %s", report["ppi_node_count"])
    logging.info("Multiomics matched HPRD genes: %s", report["multiomics_feature_match_count"])
    logging.info("Full multiomics-missing HPRD genes: %s", report["full_missing_hprd_count"])
    logging.info("CNV-missing HPRD genes: %s", report["cnv_missing_hprd_count"])
    logging.info("Final node feature shape: %s", tuple(node_features.shape))
    logging.info("NaN count: %s", nan_count)
    logging.info("Inf count: %s", inf_count)
    return node_features, cleaned_gene_name, report




def build_environment(args, run_dir, normalization_metadata=None):
    train_data, test_data, patients = inputall.getInput(args.cancer, mutation_path=args.mutation_path)
    gene_num = inputall.getGene(patients)
    net, gene_final, gene_name = inputall.getNetwork(gene_num, network_path=args.ppi_path)
    weights = inputall.getWeight(gene_name, weight_path=args.weight_path)
    for gene in gene_name:
        weights.setdefault(inputall.clean_gene_symbol(gene), 0.0)
    node_features, gene_name, feature_report = load_node_features_by_mode(
        net,
        weights,
        gene_name,
        gene_final,
        args,
        run_dir,
        normalization_metadata=normalization_metadata,
    )
    train_labels, train_path, invalid_train = load_driver_label_set(args.train_label_path)
    val_labels, val_path, invalid_val = load_driver_label_set(args.val_label_path)
    ppi_set = set(gene_name)
    matched_train = [gene for gene in train_labels if gene in ppi_set]
    matched_val = [gene for gene in val_labels if gene in ppi_set]
    if not matched_train:
        raise ValueError("PPI 节点集合中没有匹配到任何训练标签。")
    if not matched_val:
        raise ValueError("PPI 节点集合中没有匹配到任何验证标签。")
    label_report = {
        "train_label_path": str(train_path),
        "train_label_count_clean": len(train_labels),
        "train_label_count_in_ppi": len(matched_train),
        "train_invalid_count": invalid_train,
        "validation_label_path": str(val_path),
        "validation_label_count_clean": len(val_labels),
        "validation_label_count_in_ppi": len(matched_val),
        "validation_invalid_count": invalid_val,
        "train_validation_overlap": len(set(matched_train) & set(matched_val)),
        "test_labels_read": False,
    }
    if label_report["train_validation_overlap"]:
        raise ValueError(f"训练标签与验证标签存在重叠：{label_report}")
    (run_dir / "label_report.json").write_text(
        json.dumps(label_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "train_data": train_data,
        "test_data": test_data,
        "patients": patients,
        "gene_num": gene_num,
        "net": net,
        "gene_final": gene_final,
        "gene_name": gene_name,
        "weights": weights,
        "node_features": node_features,
        "train_driver_genes": matched_train,
        "validation_driver_genes": matched_val,
        "feature_report": feature_report,
        "label_report": label_report,
    }


def build_agent(args, env, device):
    reward_weights = {
        "multiomics_mutation_weight": args.multiomics_mutation_weight,
        "multiomics_expression_weight": args.multiomics_expression_weight,
        "multiomics_methylation_weight": args.multiomics_methylation_weight,
        "no_mutation_expression_weight": args.no_mutation_expression_weight,
        "no_mutation_methylation_weight": args.no_mutation_methylation_weight,
        "lowfreq_expression_weight": args.lowfreq_expression_weight,
        "lowfreq_methylation_weight": args.lowfreq_methylation_weight,
        "lowfreq_bonus_cap": args.lowfreq_bonus_cap,
    }
    if args.lowfreq_unlabeled_bonus_scale is not None:
        reward_weights["lowfreq_unlabeled_bonus_scale"] = args.lowfreq_unlabeled_bonus_scale
    if args.lowfreq_unlabeled_bonus_cap is not None:
        reward_weights["lowfreq_unlabeled_bonus_cap"] = args.lowfreq_unlabeled_bonus_cap
    evidence_by_gene = None
    if args.lowfreq_evidence_path:
        evidence_by_gene = lowfreq_evidence.load_evidence_by_gene(
            resolve_path(args.lowfreq_evidence_path, base=REPO_DIR),
            env["gene_name"],
        )
    agent = DeepQNetwork(
        n_actions=len(env["gene_name"]),
        net_ori=env["net"],
        fea_ori=env["node_features"],
        embedding_size=args.embedding_size,
        train_patient_data=env["train_data"],
        test_patient_data=env["test_data"],
        gene_sta=env["train_driver_genes"],
        weights=env["weights"],
        score_alpha=args.score_alpha,
        train_driver_set=env["train_driver_genes"],
        pat_num=len(env["patients"]),
        learning_rate=args.learning_rate,
        reward_decay=args.gamma,
        memory_size=args.buffer_size,
        batch_size=args.batch_size,
        selection_budget=args.topk,
        gradient_clip=args.gradient_clip,
        reward_mode=args.reward_mode,
        reward_weights=reward_weights,
        reward_feature_columns=env["feature_report"]["feature_columns"],
        lowfreq_evidence_by_gene=evidence_by_gene,
    )
    agent.Q.to(device)
    agent.Q_target.to(device)
    agent.Q.device = device
    agent.Q_target.device = device
    agent.Q_target.load_state_dict(agent.Q.state_dict())
    agent.Q_target.eval()
    agent.gamma = args.gamma
    agent.lr = args.learning_rate
    agent.tau = args.tau
    agent.epsilon_min = args.epsilon_end
    for group in agent.Q.optimizer.param_groups:
        group["lr"] = args.learning_rate
    agent.memory.alpha = args.per_alpha
    agent.memory.beta_start = args.per_beta_start
    agent.memory.beta_frames = args.per_beta_frames
    agent.memory.eps = args.per_eps
    expected_feature_dim = int(env["feature_report"]["feature_dim"])
    if agent.feature_dim != expected_feature_dim:
        raise ValueError(f"Q network feature_dim mismatch: expected {expected_feature_dim}, got {agent.feature_dim}.")
    return agent


def epsilon_for_step(args, global_step):
    """按照全局状态转移步数计算探索概率。"""
    if args.epsilon_decay <= 0:
        return args.epsilon_end
    value = args.epsilon_end + (args.epsilon_start - args.epsilon_end) * math.exp(
        -float(global_step) / float(args.epsilon_decay)
    )
    return max(args.epsilon_end, min(args.epsilon_start, value))


def assert_finite(name, value, episode=None, step=None, action=None):
    arr = np.asarray(value)
    if np.isnan(arr).any() or np.isinf(arr).any():
        raise FloatingPointError(
            f"{name} 出现 NaN/Inf：轮次={episode}，步骤={step}，动作={action}。"
        )


def optimize_model(agent, args, episode, step):
    if len(agent.memory) < args.warmup_steps or len(agent.memory) < args.batch_size:
        return None
    before = agent.learn_step_counter
    agent.learn()
    if agent.learn_step_counter == before:
        return None
    metrics = dict(agent.last_learn_metrics)
    if metrics:
        assert_finite("loss", metrics.get("loss", 0.0), episode, step)
        assert_finite("td_error_abs_mean", metrics.get("td_error_abs_mean", 0.0), episode, step)
        assert_finite("gradient_norm", metrics.get("gradient_norm", 0.0), episode, step)
    return metrics


def append_action_reward_log(run_dir, rows):
    if not rows:
        return
    path = Path(run_dir) / "action_reward_log.csv"
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ACTION_REWARD_LOG_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def run_episode(agent, env, args, episode, run_dir=None):
    # 当前训练入口保留全局动作空间与动作掩码，不做 PPI 邻居扩展动作集。
    action_sel = list(range(agent.n_actions))
    agent.actions = []
    agent.actions_index = np.ones(agent.n_actions, dtype=np.int64)
    agent.score_be = 0
    agent.score_sta = 0
    agent.score_pat = 0
    agent.embedding = None
    episode_reward = 0.0
    reward_component_values = {key: [] for key in REWARD_COMPONENT_KEYS}
    step_count = 0
    terminal_reason = "unknown"
    loss_values = []
    td_values = []
    grad_values = []
    state = env["node_features"]
    action_reward_rows = []

    while True:
        agent.epsilon = epsilon_for_step(args, agent.memory_counter)
        current_action_mask = agent.actions_index.copy()
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=agent.Q.device)
        mask_tensor = torch.as_tensor(
            current_action_mask,
            dtype=torch.long,
            device=agent.Q.device,
        )
        agent.Q.train()
        with torch.no_grad():
            q_values, emb = agent.Q(agent.embedding, state_tensor, mask_tensor)
        q_np = q_values.detach().cpu().numpy()
        assert_finite("Q values", q_np, episode, step_count)
        agent.embedding = emb.detach()

        valid_actions = [idx for idx in action_sel if current_action_mask[idx] == 1]
        if not valid_actions:
            terminal_reason = "no_legal_action"
            break

        best_action = max(valid_actions, key=lambda idx: q_np[idx])
        if np.random.uniform() >= agent.epsilon:
            action_index = int(best_action)
        else:
            action_index = int(random.choice(valid_actions))

        if action_index not in action_sel or current_action_mask[action_index] != 1:
            raise RuntimeError(f"选择了非法或重复动作：{action_index}")

        action_sel.remove(action_index)
        agent.actions.append(action_index)
        agent.actions_index[action_index] = 0
        next_action_mask = agent.actions_index.copy()

        reward, done, _ = agent.step(
            env["net"],
            action_index,
            env["gene_num"],
            env["gene_name"],
            env["weights"],
        )
        assert_finite("reward", reward, episode, step_count, action_index)
        for key in REWARD_COMPONENT_KEYS:
            value = float(agent.last_reward_components.get(key, 0.0))
            assert_finite(key, value, episode, step_count, action_index)
            reward_component_values[key].append(value)

        next_step_count = step_count + 1
        terminal_done = bool(done)
        truncated = next_step_count >= args.max_steps and not terminal_done
        transition_done = terminal_done or truncated
        if terminal_done:
            terminal_reason = "selection_budget"
        elif truncated:
            terminal_reason = "max_steps_truncation"

        component = dict(agent.last_reward_components)
        action_reward_rows.append(
            {
                "episode": episode,
                "step": step_count + 1,
                "action_index": action_index,
                "Gene": component.get("selected_gene", env["gene_name"][action_index]),
                "is_train_driver": bool(component.get("is_train_driver", False)),
                "mutation_count": float(component.get("mutation_count", 0.0)),
                "mutation_frequency": float(component.get("mutation_frequency", 0.0)),
                "MutationFrequencyPct": float(component.get("MutationFrequencyPct", 0.0)),
                "MutationGroup": component.get("MutationGroup", "very_low"),
                "MutationRarityScore": float(component.get("MutationRarityScore", 0.0)),
                "ExpressionSupport": float(component.get("ExpressionSupport", 0.0)),
                "MethylationSupport": float(component.get("MethylationSupport", 0.0)),
                "CNVFunctionalSupport": float(component.get("CNVFunctionalSupport", 0.0)),
                "NonMutationOmicsSupport": float(component.get("NonMutationOmicsSupport", 0.0)),
                "DegreeCorrectedNetworkSupport": float(component.get("DegreeCorrectedNetworkSupport", 0.0)),
                "DegreeCorrectedNetworkSupportV2": float(component.get("DegreeCorrectedNetworkSupportV2", 0.0)),
                "LowFrequencyEvidenceScore": float(component.get("LowFrequencyEvidenceScore", 0.0)),
                "LowFrequencyEvidenceScoreV2": float(component.get("LowFrequencyEvidenceScoreV2", 0.0)),
                "base_reward": float(component.get("base_reward", 0.0)),
                "evidence_bonus": float(component.get("evidence_bonus", 0.0)),
                "final_reward": float(component.get("final_reward", reward)),
                "done": bool(transition_done),
                "terminal_reason": terminal_reason if transition_done else "",
                "reward_mode": args.reward_mode,
            }
        )

        # 经验池保存静态节点特征、动作执行前掩码、动作执行后掩码和终止标志。
        # 在动作执行前，当前掩码与下一掩码不能被错误地当作同一个状态。
        agent.remember(
            state,
            action_index,
            reward,
            current_action_mask,
            next_action_mask,
            transition_done,
        )

        learn_metrics = optimize_model(agent, args, episode, step_count)
        if learn_metrics:
            loss_values.append(float(learn_metrics.get("loss", 0.0)))
            td_values.append(float(learn_metrics.get("td_error_abs_mean", 0.0)))
            grad_values.append(float(learn_metrics.get("gradient_norm", 0.0)))

        episode_reward += float(reward)
        step_count = next_step_count
        if transition_done:
            break

    if run_dir is not None:
        append_action_reward_log(run_dir, action_reward_rows)

    result = {
        "episode": episode,
        "global_step": agent.memory_counter,
        "episode_reward": episode_reward,
        "mean_loss": float(np.mean(loss_values)) if loss_values else "",
        "td_error_abs_mean": float(np.mean(td_values)) if td_values else "",
        "gradient_norm_mean": float(np.mean(grad_values)) if grad_values else "",
        "epsilon": agent.epsilon,
        "buffer_size": len(agent.memory),
        "learning_rate": args.learning_rate,
        "steps": step_count,
        "terminal_reason": terminal_reason,
        "learn_count": agent.learn_step_counter,
        "val_ndcg_150": "",
        "val_precision_k": "",
        "val_recall_k": "",
        "elapsed_seconds": "",
    }
    for key, values in reward_component_values.items():
        if values:
            arr = np.asarray(values, dtype=np.float64)
            result[f"{key}_sum"] = float(np.sum(arr))
            result[f"{key}_mean"] = float(np.mean(arr))
            result[f"{key}_min"] = float(np.min(arr))
            result[f"{key}_max"] = float(np.max(arr))
        else:
            result[f"{key}_sum"] = 0.0
            result[f"{key}_mean"] = 0.0
            result[f"{key}_min"] = 0.0
            result[f"{key}_max"] = 0.0
    component_sum = sum(result[f"{key}_sum"] for key in REWARD_COMPONENT_KEYS if key != "reward_total")
    if not np.isclose(component_sum, result["reward_total_sum"], atol=1e-6):
        raise FloatingPointError(
            f"Reward component sums do not match total: components={component_sum}, "
            f"total={result['reward_total_sum']}"
        )
    return result


def write_ranking(path, q_values, gene_name, feature_mode=FEATURE_MODE):
    rows = []
    for idx, gene in enumerate(gene_name):
        rows.append({"Gene": gene, "Q_value": float(q_values[idx])})
    rows.sort(key=lambda row: (-row["Q_value"], row["Gene"]))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Rank", "Gene", "Q_value", "FeatureMode"])
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "Rank": rank,
                    "Gene": row["Gene"],
                    "Q_value": row["Q_value"],
                    "FeatureMode": feature_mode,
                }
            )
    return rows


def dcg(relevances):
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances))


def metrics_at_k(ranking_rows, labels, k):
    top = ranking_rows[: min(k, len(ranking_rows))]
    top_genes = {row["Gene"] for row in top}
    hits = len(top_genes & labels)
    rel = [1 if row["Gene"] in labels else 0 for row in top]
    ideal_hits = min(k, len(labels), len(ranking_rows))
    ideal = [1] * ideal_hits + [0] * (len(top) - ideal_hits)
    idcg = dcg(ideal)
    return {
        "HitCount": hits,
        "Precision": hits / k,
        "Recall": hits / len(labels) if labels else 0.0,
        "NDCG": dcg(rel) / idcg if idcg > 0 else 0.0,
    }


def evaluate_validation(agent, env, args, run_dir, episode):
    agent.Q.eval()
    state_tensor = torch.tensor(env["node_features"], dtype=torch.float32, device=agent.Q.device)
    mask_tensor = torch.LongTensor(np.ones(agent.n_actions)).to(agent.Q.device)
    with torch.no_grad():
        q_values, _ = agent.Q(None, state_tensor, mask_tensor)
    q_np = q_values.detach().cpu().numpy()
    assert_finite("validation Q values", q_np, episode)
    ranking_path = run_dir / f"validation_ranking_episode_{episode:03d}.csv"
    ranking_rows = write_ranking(ranking_path, q_np, env["gene_name"], feature_mode=args.feature_mode)
    labels = set(env["validation_driver_genes"])
    metrics = {"episode": episode, "ranking_path": str(ranking_path)}
    for k in sorted(set(K_VALUES + [args.topk])):
        item = metrics_at_k(ranking_rows, labels, k)
        metrics[f"HitCount@{k}"] = item["HitCount"]
        metrics[f"Precision@{k}"] = item["Precision"]
        metrics[f"Recall@{k}"] = item["Recall"]
        metrics[f"NDCG@{k}"] = item["NDCG"]
    rank_by_gene = {row["Gene"]: rank for rank, row in enumerate(ranking_rows, start=1)}
    present = sorted(rank_by_gene[gene] for gene in labels if gene in rank_by_gene)
    metrics["MeanRank"] = float(np.mean(present)) if present else None
    metrics["MedianRank"] = float(np.median(present)) if present else None
    metrics["MRR"] = float(np.mean([1.0 / rank for rank in present])) if present else 0.0
    metrics["MissingValidationDriverCount"] = int(len(labels) - len(present))
    metrics_path = run_dir / f"validation_metrics_episode_{episode:03d}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    agent.Q.train()
    return metrics, ranking_path


def checkpoint_payload(agent, args, env, episode, best_val_ndcg150):
    args_dict = vars(args).copy()
    return {
        "episode": int(episode),
        "online_net_state_dict": agent.Q.state_dict(),
        "target_net_state_dict": agent.Q_target.state_dict(),
        "online_state_dict": agent.Q.state_dict(),
        "target_state_dict": agent.Q_target.state_dict(),
        "optimizer_state_dict": agent.Q.optimizer.state_dict(),
        "best_val_ndcg150": float(best_val_ndcg150),
        "epsilon": float(agent.epsilon),
        "seed": int(args.seed),
        "feature_mode": args.feature_mode,
        "feature_dim": int(env["feature_report"]["feature_dim"]),
        "feature_columns": env["feature_report"]["feature_columns"],
        "args": args_dict,
        "metadata": build_training_metadata(args, env, agent),
        "replay_buffer_serialized": False,
        "resume_note": "经验回放池未序列化；使用 --resume 时仅恢复网络、优化器、训练轮次、epsilon 和最佳验证分数。",
    }


def save_checkpoint(agent, args, env, path, episode, best_val_ndcg150):
    torch.save(checkpoint_payload(agent, args, env, episode, best_val_ndcg150), path)


def load_checkpoint(agent, path, args, env):
    payload = torch.load(path, map_location=agent.Q.device, weights_only=False)
    expected_feature_mode = args.feature_mode
    expected_feature_dim = int(env["feature_report"]["feature_dim"])
    expected_feature_columns = inputall.canonicalize_feature_columns(
        env["feature_report"]["feature_columns"]
    )
    if payload.get("feature_mode") not in {None, expected_feature_mode}:
        raise ValueError(
            f"恢复检查点的 feature_mode={payload.get('feature_mode')!r}，"
            f"预期为 {expected_feature_mode!r}。"
        )
    if int(payload.get("feature_dim", expected_feature_dim)) != expected_feature_dim:
        raise ValueError(
            f"恢复检查点的 feature_dim={payload.get('feature_dim')!r}，"
            f"预期为 {expected_feature_dim}。"
        )
    checkpoint_columns = payload.get("feature_columns")
    if checkpoint_columns is not None:
        checkpoint_columns = inputall.canonicalize_feature_columns(checkpoint_columns)
    if checkpoint_columns is not None and checkpoint_columns != expected_feature_columns:
        raise ValueError(
            f"恢复检查点中的特征列不匹配：{checkpoint_columns}"
        )

    online_key = "online_net_state_dict" if "online_net_state_dict" in payload else "online_state_dict"
    target_key = "target_net_state_dict" if "target_net_state_dict" in payload else "target_state_dict"
    agent.Q.load_state_dict(payload[online_key])
    if target_key in payload:
        agent.Q_target.load_state_dict(payload[target_key])
    else:
        agent.Q_target.load_state_dict(agent.Q.state_dict())
    agent.Q_target.eval()
    if "optimizer_state_dict" in payload:
        agent.Q.optimizer.load_state_dict(payload["optimizer_state_dict"])
    agent.epsilon = float(payload.get("epsilon", agent.epsilon))
    return int(payload.get("episode", 0)) + 1, float(
        payload.get("best_val_ndcg150", float("-inf"))
    )


def read_checkpoint_feature_metadata(path):
    payload = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
    if not isinstance(payload, dict):
        return {}
    metadata = dict(payload.get("metadata") or {})
    for key in ("feature_mode", "feature_dim", "feature_columns"):
        if key not in metadata and key in payload:
            metadata[key] = payload[key]
    if "feature_columns" in metadata:
        metadata["feature_columns"] = inputall.canonicalize_feature_columns(metadata["feature_columns"])
    return metadata


def apply_resume_feature_metadata(args):
    if not args.resume:
        return None
    metadata = read_checkpoint_feature_metadata(args.resume)
    if not metadata:
        return None

    checkpoint_mode = metadata.get("feature_mode")
    if checkpoint_mode:
        checkpoint_mode = normalize_feature_mode(checkpoint_mode)
        args.feature_mode = checkpoint_mode

    checkpoint_multiomics_path = metadata.get("multiomics_feature_path")
    if checkpoint_multiomics_path:
        args.multiomics_feature_path = str(resolve_path(checkpoint_multiomics_path))

    checkpoint_cnv_missing_path = metadata.get("cnv_missing_gene_path")
    if checkpoint_cnv_missing_path:
        args.cnv_missing_gene_path = str(resolve_path(checkpoint_cnv_missing_path))

    normalization_metadata = metadata.get("normalization_metadata")
    if normalization_metadata and "feature_names" in normalization_metadata:
        normalization_metadata = dict(normalization_metadata)
        normalization_metadata["feature_names"] = inputall.canonicalize_feature_columns(
            normalization_metadata["feature_names"]
        )
    return normalization_metadata


def build_training_metadata(args, env, agent):
    ppi_path = resolve_path(args.ppi_path)
    multiomics_path = resolve_path(args.multiomics_feature_path)
    train_label_path = resolve_path(args.train_label_path)
    val_label_path = resolve_path(args.val_label_path)
    try:
        import torch_geometric

        torch_geometric_version = torch_geometric.__version__
    except Exception:
        torch_geometric_version = None
    return {
        "feature_mode": args.feature_mode,
        "feature_source": args.feature_mode,
        "feature_dim": int(env["feature_report"]["feature_dim"]),
        "feature_columns": env["feature_report"]["feature_columns"],
        "standardize": bool(env["feature_report"].get("standardize", False)),
        "normalization_metadata": env["feature_report"].get("normalization_metadata", {}),
        "node_features_shape": list(env["node_features"].shape),
        "ppi_node_count": len(env["gene_name"]),
        "gene_name_sha256": sha256_text(env["gene_name"]),
        "ppi_path": str(ppi_path),
        "ppi_sha256": sha256_file(ppi_path),
        "multiomics_feature_path": str(multiomics_path),
        "multiomics_sha256": sha256_file(multiomics_path),
        "cnv_missing_gene_path": str(resolve_path(args.cnv_missing_gene_path)),
        "cnv_missing_gene_sha256": sha256_file(resolve_path(args.cnv_missing_gene_path))
        if Path(resolve_path(args.cnv_missing_gene_path)).exists()
        else None,
        "lowfreq_evidence_path": str(resolve_path(args.lowfreq_evidence_path, base=REPO_DIR))
        if args.lowfreq_evidence_path
        else None,
        "lowfreq_evidence_sha256": sha256_file(resolve_path(args.lowfreq_evidence_path, base=REPO_DIR))
        if args.lowfreq_evidence_path
        else None,
        "lowfreq_unlabeled_bonus_scale": args.lowfreq_unlabeled_bonus_scale,
        "lowfreq_unlabeled_bonus_cap": args.lowfreq_unlabeled_bonus_cap,
        "train_label_path": str(train_label_path),
        "train_label_sha256": sha256_file(train_label_path),
        "validation_label_path": str(val_label_path),
        "validation_label_sha256": sha256_file(val_label_path),
        "test_labels_read": False,
        "external_holdout_read": False,
        "historical_test_read": False,
        "seed": args.seed,
        "max_episodes": args.max_episodes,
        "max_steps": args.max_steps,
        "selection_budget": args.topk,
        "gamma": agent.gamma,
        "tau": agent.tau,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "buffer_size": args.buffer_size,
        "per_alpha": agent.memory.alpha,
        "per_beta_start": agent.memory.beta_start,
        "per_beta_frames": agent.memory.beta_frames,
        "per_eps": agent.memory.eps,
        "reward_config": agent.reward_config(),
        "q_parameter_count": q_parameter_count(agent.Q),
        "python": sys.version,
        "torch_version": torch.__version__,
        "torch_geometric_version": torch_geometric_version,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": get_git_commit(),
        "git_diff_sha256": get_git_diff_sha256(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def save_training_metrics(path, rows):
    fieldnames = [
        "episode",
        "global_step",
        "episode_reward",
        "mean_loss",
        "td_error_abs_mean",
        "gradient_norm_mean",
        "epsilon",
        "buffer_size",
        "learning_rate",
        "steps",
        "terminal_reason",
        "learn_count",
        "val_ndcg_150",
        "val_precision_k",
        "val_recall_k",
        "elapsed_seconds",
    ]
    for key in REWARD_COMPONENT_KEYS:
        for suffix in ["sum", "mean", "min", "max"]:
            fieldnames.append(f"{key}_{suffix}")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def copy_best_artifacts(ranking_path, metrics, run_dir):
    shutil.copy2(ranking_path, run_dir / "validation_ranking_best.csv")
    write_json(run_dir / "validation_metrics_best.json", metrics)


def main():
    args = parse_args()
    validate_no_test_path(args)
    validate_training_args(args)
    run_dir = make_run_dir(args.output_dir, args.seed, args.feature_mode)
    setup_logger(run_dir)
    start_time = time.perf_counter()
    set_seed(args.seed)
    device = choose_device(args.device)
    logging.info("Feature mode: %s", args.feature_mode)
    logging.info("Reward mode: %s", args.reward_mode)
    logging.info("本次训练输出目录：%s", run_dir)
    logging.info("训练设备：%s", device)
    resume_normalization_metadata = apply_resume_feature_metadata(args)
    validate_training_args(args)
    env = build_environment(
        args,
        run_dir,
        normalization_metadata=resume_normalization_metadata,
    )
    if args.topk > len(env["gene_name"]):
        raise ValueError(
            f"参数 --topk={args.topk} 超过 PPI 节点数 {len(env['gene_name'])}。"
        )
    agent = build_agent(args, env, device)
    config = {
        "args": vars(args),
        "run_dir": str(run_dir),
        "feature_mode": args.feature_mode,
        "feature_dim": int(env["feature_report"]["feature_dim"]),
        "feature_columns": env["feature_report"]["feature_columns"],
        "device": str(device),
        "python": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": get_git_commit(),
        "git_diff_sha256": get_git_diff_sha256(),
        "reward_config": agent.reward_config(),
        "feature_report": env["feature_report"],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "test_labels_read": False,
        "external_holdout_read": False,
        "historical_test_read": False,
        "notes": [
            "本训练脚本不会读取最终测试集标签。",
            "feature_mode controls input columns only and is independent from reward_mode.",
            "CNV is not introduced as an additional raw input column for this stage; lowfreq_unlabeled_* may use frozen CNV evidence inside reward.",
        ],
    }
    write_json(run_dir / "config.json", config)
    start_episode = 1
    best_val_ndcg150 = float("-inf")
    best_episode = None
    if args.resume:
        start_episode, best_val_ndcg150 = load_checkpoint(agent, args.resume, args, env)
        logging.info("已从 %s 恢复训练，将从第 %s 轮开始；经验回放池为空并重新积累。", args.resume, start_episode)
    rows = []
    for episode in range(start_episode, args.max_episodes + 1):
        episode_start = time.perf_counter()
        metrics = run_episode(agent, env, args, episode, run_dir=run_dir)
        metrics["elapsed_seconds"] = time.perf_counter() - episode_start
        if args.val_interval > 0 and episode % args.val_interval == 0:
            val_metrics, ranking_path = evaluate_validation(agent, env, args, run_dir, episode)
            metrics["val_ndcg_150"] = val_metrics.get("NDCG@150", "")
            metrics["val_precision_k"] = val_metrics.get(f"Precision@{args.topk}", "")
            metrics["val_recall_k"] = val_metrics.get(f"Recall@{args.topk}", "")
            current = float(val_metrics.get("NDCG@150", 0.0))
            if current > best_val_ndcg150:
                best_val_ndcg150 = current
                best_episode = episode
                save_checkpoint(agent, args, env, run_dir / "checkpoint_best.pt", episode, best_val_ndcg150)
                copy_best_artifacts(ranking_path, val_metrics, run_dir)
        save_checkpoint(agent, args, env, run_dir / "checkpoint_last.pt", episode, best_val_ndcg150)
        rows.append(metrics)
        save_training_metrics(run_dir / "train_metrics.csv", rows)
        logging.info(
            "训练轮次=%s，累计奖励=%.6f，步骤数=%s，结束原因=%s，平均损失=%s，验证集NDCG@150=%s",
            episode,
            metrics["episode_reward"],
            metrics["steps"],
            TERMINAL_REASON_CN.get(metrics["terminal_reason"], metrics["terminal_reason"]),
            metrics["mean_loss"],
            metrics["val_ndcg_150"],
        )
    summary = {
        "status": "COMPLETED",
        "run_dir": str(run_dir),
        "feature_mode": args.feature_mode,
        "feature_dim": int(env["feature_report"]["feature_dim"]),
        "feature_columns": env["feature_report"]["feature_columns"],
        "node_features_shape": list(env["node_features"].shape),
        "best_episode": best_episode,
        "best_val_ndcg150": best_val_ndcg150 if best_val_ndcg150 != float("-inf") else None,
        "checkpoint_best": str(run_dir / "checkpoint_best.pt") if (run_dir / "checkpoint_best.pt").exists() else None,
        "checkpoint_last": str(run_dir / "checkpoint_last.pt"),
        "test_labels_read": False,
        "external_holdout_read": False,
        "historical_test_read": False,
        "runtime_seconds": time.perf_counter() - start_time,
    }
    write_json(run_dir / "summary.json", summary)
    logging.info("训练完成。汇总信息：%s", summary)
    return summary


if __name__ == "__main__":
    main()
