from pathlib import Path

import pandas as pd


# 当前脚本位于 RL-GenRisk-main/src
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

print("项目根目录：", PROJECT_ROOT)
print("数据目录：", RAW_DIR)
print("数据目录是否存在：", RAW_DIR.exists())

patterns = [
    "KIRC_mc3*",
    "HiSeqV2*",
    "HumanMethylation450*",
]

for pattern in patterns:
    files = list(RAW_DIR.glob(pattern))

    if not files:
        print(f"\n没有找到：{pattern}")
        continue

    path = files[0]

    print("\n" + "=" * 80)
    print("文件：", path.name)
    print("完整路径：", path)
    print("文件大小：", round(path.stat().st_size / 1024 / 1024, 2), "MB")

    try:
        df = pd.read_csv(
            path,
            sep="\t",
            nrows=3,
            low_memory=False,
        )

        print("列数：", len(df.columns))
        print("前10个列名：")
        print(df.columns[:10].tolist())

        print("前3行、前8列：")
        print(df.iloc[:, :8])

    except Exception as exc:
        print("读取失败：", repr(exc))