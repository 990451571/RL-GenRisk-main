from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def get_sample_type(sample_name):
    parts = str(sample_name).split("-")
    if len(parts) < 4:
        return "Unknown"
    return parts[3][:2]


# =========================
# 1. 突变
# =========================
mutation_path = RAW_DIR / "KIRC_mc3.txt"

mutation = pd.read_csv(
    mutation_path,
    sep="\t",
    usecols=["sample", "gene", "effect"],
    low_memory=False,
)

mutation["sample_type"] = mutation["sample"].map(get_sample_type)

print("=" * 60)
print("突变样本类型统计")
print(
    mutation.groupby("sample_type")["sample"]
    .nunique()
    .sort_index()
)

print("\n突变类型统计")
print(mutation["effect"].value_counts().head(30))


# =========================
# 2. 表达
# =========================
expression_path = RAW_DIR / "HiSeqV2"

expression_header = pd.read_csv(
    expression_path,
    sep="\t",
    nrows=0,
)

expression_samples = expression_header.columns[1:]
expression_types = pd.Series(
    [get_sample_type(x) for x in expression_samples]
)

print("\n" + "=" * 60)
print("表达样本类型统计")
print(expression_types.value_counts().sort_index())


# =========================
# 3. 甲基化
# =========================
methylation_path = RAW_DIR / "HumanMethylation450"

methylation_header = pd.read_csv(
    methylation_path,
    sep="\t",
    nrows=0,
)

methylation_samples = methylation_header.columns[1:]
methylation_types = pd.Series(
    [get_sample_type(x) for x in methylation_samples]
)

print("\n" + "=" * 60)
print("甲基化样本类型统计")
print(methylation_types.value_counts().sort_index())