from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MULTIOMICS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "KIRC_multiomics_3omics.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ppi_overlap_check"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def clean_gene(value):
    """统一Gene Symbol格式。"""
    if pd.isna(value):
        return None

    gene = str(value).strip().upper()

    if "|" in gene:
        gene = gene.split("|", 1)[0].strip()

    if gene in {
        "",
        "?",
        "NA",
        "N/A",
        "NAN",
        "NONE",
        "NULL",
    }:
        return None

    return gene


def check_multiomics_ppi_overlap(
    ppi_gene_list,
):
    """
    ppi_gene_list：
        模型当前PPI网络实际使用的基因列表。
    """

    # ========================================================
    # 1. 读取三组学数据
    # ========================================================

    multiomics = pd.read_csv(
        MULTIOMICS_PATH
    )

    if "Gene" not in multiomics.columns:
        raise ValueError(
            "三组学文件缺少Gene列。"
        )

    multiomics["Gene"] = (
        multiomics["Gene"]
        .map(clean_gene)
    )

    multiomics = (
        multiomics[
            multiomics["Gene"].notna()
        ]
        .drop_duplicates("Gene")
        .copy()
    )

    multiomics_genes = set(
        multiomics["Gene"]
    )

    # ========================================================
    # 2. 清理PPI基因
    # ========================================================

    ppi_clean = [
        clean_gene(gene)
        for gene in ppi_gene_list
    ]

    ppi_clean = [
        gene
        for gene in ppi_clean
        if gene is not None
    ]

    ppi_genes = set(
        ppi_clean
    )

    # ========================================================
    # 3. 计算交集和差集
    # ========================================================

    matched_genes = (
        multiomics_genes
        & ppi_genes
    )

    multiomics_only = (
        multiomics_genes
        - ppi_genes
    )

    ppi_only = (
        ppi_genes
        - multiomics_genes
    )

    # ========================================================
    # 4. 统计
    # ========================================================

    multiomics_total = len(
        multiomics_genes
    )

    ppi_total = len(
        ppi_genes
    )

    matched_total = len(
        matched_genes
    )

    multiomics_match_rate = (
        matched_total
        / multiomics_total
        if multiomics_total > 0
        else 0
    )

    ppi_coverage_rate = (
        matched_total
        / ppi_total
        if ppi_total > 0
        else 0
    )

    print("=" * 70)
    print("三组学与PPI基因匹配结果")
    print("=" * 70)

    print(
        "三组学唯一基因数：",
        multiomics_total,
    )

    print(
        "PPI唯一基因数：",
        ppi_total,
    )

    print(
        "成功匹配基因数：",
        matched_total,
    )

    print(
        "三组学基因进入PPI的比例："
        f"{multiomics_match_rate:.2%}"
    )

    print(
        "PPI节点具有三组学特征的比例："
        f"{ppi_coverage_rate:.2%}"
    )

    print(
        "仅三组学中存在的基因数：",
        len(multiomics_only),
    )

    print(
        "仅PPI中存在的基因数：",
        len(ppi_only),
    )

    # ========================================================
    # 5. 检查匹配基因的三组学数值
    # ========================================================

    matched_data = (
        multiomics[
            multiomics["Gene"].isin(
                matched_genes
            )
        ]
        .copy()
        .sort_values("Gene")
    )

    features = [
        "Mutation",
        "Expression",
        "Methylation",
    ]

    print("\n匹配基因特征统计：")

    print(
        matched_data[
            features
        ]
        .describe()
        .transpose()
    )

    print("\n匹配基因各特征零值比例：")

    print(
        (
            matched_data[
                features
            ] == 0
        ).mean()
    )

    # 三个特征都为0的基因
    all_zero_mask = (
        matched_data[
            features
        ]
        .eq(0)
        .all(axis=1)
    )

    all_zero_genes = (
        matched_data.loc[
            all_zero_mask,
            ["Gene"] + features
        ]
        .copy()
    )

    print(
        "\n匹配后三个组学全部为0的基因数：",
        len(all_zero_genes),
    )

    # ========================================================
    # 6. 保存结果
    # ========================================================

    matched_data.to_csv(
        OUTPUT_DIR
        / "matched_multiomics_ppi_genes.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        {
            "Gene": sorted(
                multiomics_only
            )
        }
    ).to_csv(
        OUTPUT_DIR
        / "multiomics_not_in_ppi.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        {
            "Gene": sorted(
                ppi_only
            )
        }
    ).to_csv(
        OUTPUT_DIR
        / "ppi_without_multiomics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_zero_genes.to_csv(
        OUTPUT_DIR
        / "matched_genes_all_omics_zero.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = pd.DataFrame(
        {
            "Metric": [
                "Multiomics_unique_genes",
                "PPI_unique_genes",
                "Matched_genes",
                "Multiomics_match_rate",
                "PPI_coverage_rate",
                "Multiomics_not_in_PPI",
                "PPI_without_multiomics",
                "Matched_all_omics_zero",
            ],
            "Value": [
                multiomics_total,
                ppi_total,
                matched_total,
                multiomics_match_rate,
                ppi_coverage_rate,
                len(multiomics_only),
                len(ppi_only),
                len(all_zero_genes),
            ],
        }
    )

    summary.to_csv(
        OUTPUT_DIR
        / "overlap_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n结果已保存到：",
        OUTPUT_DIR,
    )

    return {
        "multiomics_total": multiomics_total,
        "ppi_total": ppi_total,
        "matched_total": matched_total,
        "multiomics_match_rate": (
            multiomics_match_rate
        ),
        "ppi_coverage_rate": (
            ppi_coverage_rate
        ),
    }




if __name__ == "__main__":

    # 示例：
    # from inputall import gene_list
    # check_multiomics_ppi_overlap(gene_list)

    raise RuntimeError(
        "请把这里替换为模型实际使用的PPI gene_list。"
    )
