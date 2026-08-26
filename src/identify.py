from inputall import *
from DQN import *
import matplotlib.pyplot as plt
import os
import csv
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import networkx as nx
import random
import inputall
from sklearn import preprocessing
from statsmodels.stats.multitest import multipletests

# 🚀 关键修复 1：把显卡视野切回你电脑上的唯一主卡 (GPU 0)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch

torch.set_num_threads(4)
gene_sta = []


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("1", "true", "yes", "y")


def sha256_text(items):
    digest = hashlib.sha256()
    for item in items:
        digest.update(str(item).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def q_parameter_count(model):
    return sum(param.numel() for param in model.parameters())


def load_checkpoint_payload(path, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    if isinstance(payload, dict) and "online_state_dict" in payload:
        return payload, payload.get("metadata", {}), False
    print("legacy checkpoint，缺少metadata，需要显式指定特征配置；")
    return {"online_state_dict": payload}, None, True


def resolve_identify_feature_config(args, metadata, legacy):
    explicit_feature_source = args.feature_source is not None
    explicit_multiomics_path = args.multiomics_feature_path is not None
    explicit_standardize = args.standardize_multiomics is not None

    if legacy:
        if not explicit_feature_source:
            raise ValueError("Legacy checkpoint has no metadata; please provide --feature-source explicitly.")
        feature_source = args.feature_source
        multiomics_path = args.multiomics_feature_path
        standardize = bool(args.standardize_multiomics) if explicit_standardize else False
        return feature_source, multiomics_path, standardize

    feature_source = metadata.get("feature_source")
    multiomics_path = metadata.get("multiomics_feature_path")
    standardize = metadata.get("standardize_multiomics", False)

    requested = {
        "feature_source": args.feature_source if explicit_feature_source else feature_source,
        "multiomics_feature_path": args.multiomics_feature_path if explicit_multiomics_path else multiomics_path,
        "standardize_multiomics": bool(args.standardize_multiomics) if explicit_standardize else standardize,
    }
    metadata_values = {
        "feature_source": feature_source,
        "multiomics_feature_path": multiomics_path,
        "standardize_multiomics": standardize,
    }
    overrides = {
        key: (metadata_values[key], requested[key])
        for key in metadata_values
        if requested[key] != metadata_values[key]
    }
    if overrides and not args.allow_config_override:
        raise ValueError(f"Refusing to override checkpoint metadata without --allow-config-override: {overrides}")
    if overrides:
        print(f"⚠️ WARNING: overriding checkpoint metadata: {overrides}")
    return requested["feature_source"], requested["multiomics_feature_path"], requested["standardize_multiomics"]


def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


test_sta_total = []
test_sta = []
test_sta2 = []
sample_index = []

def Normalized(feature):
    X_scaler = preprocessing.StandardScaler()
    X_train = X_scaler.fit_transform(feature)
    return X_train


def Normalized_minmax(feature):
    X_scaler = preprocessing.MinMaxScaler()
    X_train = copy.deepcopy(feature)
    for i in range(len(feature[0])):
        X_train[:, [i]] = X_scaler.fit_transform(feature[:, [i]])
    return X_train


def get_feature(net, weights, gene_name, gene_final):
    return build_original_node_features(net, weights, gene_name, gene_final)


def get_feature1(net, actions, weights, gene_name, gene_final):
    feature = build_original_node_features(net, weights, gene_name, gene_final)
    if len(actions) > 0:
        feature[actions, :] = 0.0
    return feature


def laplacian(net):
    lap = copy.deepcopy(net)
    lap = lap * (-1)
    for i in range(net.shape[0]):
        lap[i][i] = np.sum(net[i])
    return lap


def evaluate(gene2):
    gene_sta = []
    with open('../data/GeneID.csv', 'r') as f:
        reader = csv.reader(f)
        for i in reader:
            gene_sta.append(i[0])

    gene1_num = []
    num = 0
    for i in list(gene2.keys()):
        if i in gene_sta:
            num = num + 1
    num1 = 0
    num2 = 0
    for i in list(gene2.keys()):
        if i in test_sta:
            num1 = num1 + 1
        if i in test_sta2:
            num2 = num2 + 1

    return num, num1, num2


def parse_args():
    parser = argparse.ArgumentParser(description="Rank genes with a trained RL-GenRisk DDQN model.")
    parser.add_argument("--cancer", default=os.getenv("RL_GENRISK_CANCER", "KIRC"))
    parser.add_argument("--mutation-path", default=os.getenv("RL_GENRISK_MUTATION_PATH"))
    parser.add_argument("--label-path", default=os.getenv("RL_GENRISK_LABEL_PATH"))
    parser.add_argument("--network-path", default=os.getenv("RL_GENRISK_NETWORK_PATH"))
    parser.add_argument("--weight-path", default=os.getenv("RL_GENRISK_WEIGHT_PATH"))
    parser.add_argument("--model-path", "--checkpoint", dest="model_path", default=os.getenv("RL_GENRISK_MODEL_PATH"))
    parser.add_argument("--output-path", "--output", dest="output_path", default=os.getenv("RL_GENRISK_OUTPUT_PATH"))
    parser.add_argument("--embedding-path", default=os.getenv("RL_GENRISK_EMBEDDING_PATH", "../data/Embedding.npy"))
    parser.add_argument("--embedding-index-path", default=os.getenv("RL_GENRISK_EMBEDDING_INDEX_PATH", "../data/Embedding_Gene_idx.txt"))
    parser.add_argument("--feature-source", choices=["multiomics", "original"], default=os.getenv("RL_GENRISK_FEATURE_SOURCE"))
    parser.add_argument(
        "--feature-mode",
        choices=sorted(inputall.FEATURE_MODE_COLUMNS),
        default=os.getenv("RL_GENRISK_FEATURE_MODE"),
    )
    parser.add_argument("--strict-config", type=str_to_bool, default=str_to_bool(os.getenv("RL_GENRISK_STRICT_CONFIG", "True")))
    parser.add_argument("--allow-config-override", action="store_true")
    parser.add_argument(
        "--multiomics-feature-path",
        "--multiomics_feature_path",
        dest="multiomics_feature_path",
        default=os.getenv("RL_GENRISK_MULTIOMICS_FEATURE_PATH"),
    )
    parser.add_argument(
        "--cnv-missing-gene-path",
        "--cnv_missing_gene_path",
        dest="cnv_missing_gene_path",
        default=os.getenv("RL_GENRISK_CNV_MISSING_GENE_PATH"),
    )
    parser.add_argument(
        "--use-multiomics",
        "--use_multiomics",
        dest="use_multiomics",
        type=str_to_bool,
        default=str_to_bool(os.getenv("RL_GENRISK_USE_MULTIOMICS", "True")),
    )
    parser.add_argument(
        "--standardize-multiomics",
        "--standardize_multiomics",
        dest="standardize_multiomics",
        type=str_to_bool,
        default=None if os.getenv("RL_GENRISK_STANDARDIZE_MULTIOMICS") is None else str_to_bool(os.getenv("RL_GENRISK_STANDARDIZE_MULTIOMICS")),
    )
    return parser.parse_args()


def get_data_output(network_path, label_path, embedding_path, embedding_index_path):
    df_HPRD = pd.read_csv(resolve_data_path(network_path or 'HPRD.txt'), sep=' ', header=None)
    edge_list = np.array(df_HPRD)
    G_hprd = nx.Graph()
    G_hprd.add_edges_from(edge_list)
    nodelist = list(G_hprd.nodes())
    df_gold = pd.read_csv(resolve_data_path(label_path), header=None)
    lst_gold = df_gold[0].tolist()
    lst_gold = np.intersect1d(nodelist, lst_gold)
    embedding = np.load(resolve_data_path(embedding_path))
    df_gene_idx = pd.read_csv(resolve_data_path(embedding_index_path), sep='\t', header=None)
    emb_gene_idx = df_gene_idx[0].tolist()
    dict_embedding = {}
    for i in range(len(emb_gene_idx)):
        dict_embedding[emb_gene_idx[i]] = embedding[i]
    return G_hprd, lst_gold, dict_embedding


def one_side_ttest(value, lst_data):
    from scipy import stats
    data = lst_data
    t_stat, p_value = stats.ttest_1samp(data, value)
    p_value_lesser = p_value / 2 if t_stat > 0 else 1 - p_value / 2
    p_value_greater = p_value / 2 if t_stat < 0 else 1 - p_value / 2
    return p_value_greater, p_value_lesser


def get_average_STPL(gene_ranking, G_hprd, lst_gold, random_samples):
    dict_average_STPL = {}
    dict_average_STPL_p = {}
    for x in gene_ranking:
        dict_average_STPL[x] = 0
        dict_average_STPL_p[x] = 0
    lst_data = []
    for x in gene_ranking:
        lst_lengths = []
        for y in lst_gold:
            if x == y:
                continue
            if nx.has_path(G_hprd, x, y):
                length = nx.shortest_path_length(G_hprd, y, x)
                lst_lengths.append(length)
            else:
                lst_lengths.append(15)
        dict_average_STPL[x] = np.nanmean(lst_lengths)
        lst_data.append(np.nanmean(lst_lengths))

    lst_random_value = []
    for x in random_samples:
        lst_random_value.append(dict_average_STPL[x])
    for x in gene_ranking:
        p_greater, p_lesser = one_side_ttest(dict_average_STPL[x], lst_random_value)
        dict_average_STPL_p[x] = p_lesser
    return dict_average_STPL, dict_average_STPL_p


def get_average_CS(gene_ranking, lst_gold, dict_embedding, random_samples):
    from sklearn.metrics.pairwise import cosine_similarity
    dict_average_CS = {}
    dict_average_CS_p = {}
    for x in gene_ranking:
        dict_average_CS[x] = 0
        dict_average_CS_p[x] = 0
    lst_data = []
    for x in gene_ranking:
        lst_lengths = []
        tmp_emb = dict_embedding[x].reshape(1, -1)
        sim_lst = []
        for y in lst_gold:
            if x == y:
                continue
            tgt_emb = dict_embedding[y].reshape(1, -1)
            sim = cosine_similarity(tmp_emb, tgt_emb)[0][0]
            sim_lst.append(sim)
        dict_average_CS[x] = np.nanmean(sim_lst)
        lst_data.append(np.nanmean(sim_lst))
    lst_random_value = []
    for x in random_samples:
        lst_random_value.append(dict_average_CS[x])
    for x in gene_ranking:
        p_greater, p_lesser = one_side_ttest(dict_average_CS[x], lst_random_value)
        dict_average_CS_p[x] = p_greater
    return dict_average_CS, dict_average_CS_p


def FDR_adj_P(dict_average_STPL_p, dict_average_CS_p, gene_ranking):
    dict_average_STPL_FDR_p = {}
    dict_average_CS_FDR_p = {}
    p_value_lst_STPL = []
    p_value_lst_CS = []
    for x in gene_ranking:
        p_value_lst_STPL.append(dict_average_STPL_p[x])
        p_value_lst_CS.append(dict_average_CS_p[x])

    rejected, pvals_corrected_STPL, _, _ = multipletests(p_value_lst_STPL, method='fdr_bh')
    rejected, pvals_corrected_CS, _, _ = multipletests(p_value_lst_CS, method='fdr_bh')
    base = 9039
    for i in range(len(gene_ranking)):
        tmp_gene = gene_ranking[i]
        tmp_FDR_p_STPL = pvals_corrected_STPL[i]
        tmp_FDR_p_CS = pvals_corrected_CS[i]
        dict_average_STPL_FDR_p[tmp_gene] = tmp_FDR_p_STPL
        dict_average_CS_FDR_p[tmp_gene] = tmp_FDR_p_CS

    return dict_average_STPL_FDR_p, dict_average_CS_FDR_p


def _canonical_normalization_metadata(metadata):
    if not metadata:
        return metadata
    normalized = dict(metadata)
    if "feature_names" in normalized:
        normalized["feature_names"] = inputall.canonicalize_feature_columns(
            normalized["feature_names"]
        )
    return normalized


def build_identify_node_features(net, weights, gene_name, gene_final, args, metadata=None, legacy_checkpoint=False):
    metadata = metadata or {}
    feature_source_cfg, multiomics_path_cfg, standardize_cfg = resolve_identify_feature_config(
        args,
        metadata,
        legacy_checkpoint,
    )
    if metadata.get("feature_mode"):
        feature_mode = metadata["feature_mode"]
        multiomics_path = (
            multiomics_path_cfg
            or metadata.get("multiomics_feature_path")
            or inputall.default_multiomics_feature_path_for_mode(feature_mode)
        )
        metadata_cnv_missing_path = metadata.get("cnv_missing_gene_path")
        if (
            metadata_cnv_missing_path
            and args.cnv_missing_gene_path
            and str(args.cnv_missing_gene_path) != str(metadata_cnv_missing_path)
            and not args.allow_config_override
        ):
            raise ValueError(
                "Refusing to override checkpoint cnv_missing_gene_path without "
                "--allow-config-override."
            )
        cnv_missing_path = (
            metadata_cnv_missing_path
            or args.cnv_missing_gene_path
            or inputall.DEFAULT_CNV_MISSING_GENE_PATH
        )
        normalization_metadata = _canonical_normalization_metadata(
            metadata.get("normalization_metadata")
        )
        feature, feature_names, feature_source, normalization_metadata, report = inputall.get_node_features_by_mode(
            net,
            weights,
            gene_name,
            gene_final,
            feature_mode=feature_mode,
            multiomics_feature_path=multiomics_path,
            cnv_missing_gene_path=cnv_missing_path,
            normalization_metadata=normalization_metadata,
            return_report=True,
        )
        feature_names = inputall.canonicalize_feature_columns(feature_names)
        return {
            "feature": feature,
            "feature_names": feature_names,
            "feature_source": feature_source,
            "feature_mode": feature_mode,
            "normalization_metadata": normalization_metadata,
            "multiomics_feature_path": str(multiomics_path) if multiomics_path else None,
            "cnv_missing_gene_path": str(cnv_missing_path) if cnv_missing_path else None,
            "standardize_multiomics": feature_mode in inputall.Z_SCORE_FEATURE_MODES,
            "feature_report": report,
        }

    use_multiomics = feature_source_cfg == "multiomics"
    feature, feature_source = get_node_features(
        net,
        weights,
        gene_name,
        gene_final,
        use_multiomics=use_multiomics,
        multiomics_feature_path=multiomics_path_cfg,
        standardize_multiomics=standardize_cfg,
    )
    return {
        "feature": feature,
        "feature_names": [],
        "feature_source": feature_source,
        "feature_mode": args.feature_mode,
        "normalization_metadata": {},
        "multiomics_feature_path": str(multiomics_path_cfg) if multiomics_path_cfg else None,
        "cnv_missing_gene_path": args.cnv_missing_gene_path,
        "standardize_multiomics": standardize_cfg,
        "feature_report": {},
    }


def compute_identify_q_values(agent, checkpoint_payload, feature):
    if "online_state_dict" not in checkpoint_payload:
        raise ValueError("Checkpoint payload missing online_state_dict.")
    agent.clear_mem()
    agent.epsilon = 1
    agent.Q.load_state_dict(checkpoint_payload["online_state_dict"])
    if "target_state_dict" in checkpoint_payload:
        agent.Q_target.load_state_dict(checkpoint_payload["target_state_dict"])
    agent.Q.eval()
    state_tensor = torch.tensor(feature, dtype=torch.float32).to(agent.Q.device)
    action_mask_tensor = torch.LongTensor(np.ones(agent.n_actions)).to(agent.Q.device)
    with torch.no_grad():
        q_values, _ = agent.Q(None, state_tensor, action_mask_tensor)
    q_values_np = q_values.detach().cpu().numpy()
    if np.isnan(q_values_np).any() or np.isinf(q_values_np).any():
        raise ValueError("Q values contain NaN or Inf.")
    return q_values_np


def run(gene_final, score_alpha, args):
    save_path = args.model_path or str(resolve_data_path(f"agent_{args.cancer}_driver_DDQN_PER.th"))

    print("🧠 正在加载checkpoint...")
    checkpoint_payload, metadata, legacy_checkpoint = load_checkpoint_payload(save_path, RL.Q.device)
    feature_bundle = build_identify_node_features(
        network,
        weights,
        gene_name,
        gene_final,
        args,
        metadata,
        legacy_checkpoint,
    )
    feature = feature_bundle["feature"]
    feature_names = feature_bundle["feature_names"]
    feature_source = feature_bundle["feature_source"]
    feature_mode_value = feature_bundle["feature_mode"]
    normalization_metadata = feature_bundle["normalization_metadata"]
    multiomics_path_cfg = feature_bundle["multiomics_feature_path"]
    standardize_cfg = feature_bundle["standardize_multiomics"]
    print(f"📌 当前节点特征来源：{feature_source}")
    print(f"📌 当前 node_features.shape：{feature.shape}")

    gene_hash = sha256_text(gene_name)
    validation_lines = []
    validation_lines.append(f"legacy_checkpoint={legacy_checkpoint}")
    validation_lines.append(f"feature_source={feature_source}")
    validation_lines.append(f"feature_mode={feature_mode_value}")
    validation_lines.append(f"feature_names={feature_names}")
    validation_lines.append(f"normalization_metadata={normalization_metadata}")
    validation_lines.append(f"multiomics_feature_path={multiomics_path_cfg}")
    validation_lines.append(f"standardize_multiomics={standardize_cfg}")
    validation_lines.append(f"node_features_shape={tuple(feature.shape)}")
    validation_lines.append(f"gene_name_sha256={gene_hash}")
    validation_lines.append(f"ppi_node_count={len(gene_name)}")
    validation_lines.append(f"q_parameter_count={q_parameter_count(RL.Q)}")

    if metadata:
        checks = {
            "feature_source": (metadata.get("feature_source") is None) or feature_source == metadata.get("feature_source"),
            "feature_mode": (not metadata.get("feature_mode")) or feature_mode_value == metadata.get("feature_mode"),
            "feature_columns": inputall.canonicalize_feature_columns(
                metadata.get("feature_columns", feature_names)
            ) == feature_names,
            "feature_dim": int(feature.shape[1]) == int(metadata.get("feature_dim")),
            "node_features_shape": list(feature.shape) == metadata.get("node_features_shape"),
            "gene_name_sha256": gene_hash == metadata.get("gene_name_sha256"),
            "ppi_node_count": len(gene_name) == int(metadata.get("ppi_node_count")),
            "q_parameter_count": q_parameter_count(RL.Q) == int(metadata.get("q_parameter_count")),
        }
        validation_lines.append(f"metadata_checks={checks}")
        if args.strict_config and not all(checks.values()):
            raise ValueError(f"Checkpoint metadata validation failed: {checks}")
    elif args.strict_config:
        raise ValueError("Strict config requires new-format checkpoint metadata.")

    print("📊 AI 正在给全部PPI节点打分，请稍候...")
    q_values_np = compute_identify_q_values(RL, checkpoint_payload, feature)

    output_path = args.output_path or f"Ranking_List_{args.cancer}_driver_DDQN_PER.txt"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = []
    for idx, gene in enumerate(gene_name):
        row = {
            "Gene": gene,
            "Q_value": float(q_values_np[idx]),
            "FeatureSource": feature_source,
            "Checkpoint": str(save_path),
            "Seed": metadata.get("seed") if metadata else "",
        }
        if feature.shape[1] >= 3 and feature_source == "multiomics":
            row.update(
                {
                    "Mutation": float(feature[idx, 0]),
                    "Expression": float(feature[idx, 1]),
                    "Methylation": float(feature[idx, 2]),
                }
            )
        result.append(row)
    result.sort(key=lambda x: (-x["Q_value"], x["Gene"]))
    for rank, row in enumerate(result, start=1):
        row["Rank"] = rank

    fieldnames = ["Rank", "Gene", "Q_value", "FeatureSource", "Checkpoint", "Seed"]
    if feature_source == "multiomics" and feature.shape[1] >= 3:
        fieldnames += ["Mutation", "Expression", "Methylation"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result)

    config_log = output_path.with_name("identify_config_validation.txt")
    config_log.write_text("\n".join(validation_lines) + "\n", encoding="utf-8")
    print(f"🎉 Ranking已生成：{output_path}")
    return result


if __name__ == "__main__":
    import sys

    args = parse_args()
    cancer = args.cancer
    pre_metadata = None
    pre_legacy = False
    pre_checkpoint_path = args.model_path or str(resolve_data_path(f"agent_{args.cancer}_driver_DDQN_PER.th"))
    if pre_checkpoint_path:
        pre_payload = torch.load(
            pre_checkpoint_path,
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            weights_only=False,
        )
        if isinstance(pre_payload, dict) and "online_state_dict" in pre_payload:
            pre_metadata = pre_payload.get("metadata", {})
            if args.feature_mode is None:
                args.feature_mode = pre_metadata.get("feature_mode")
            if args.feature_source is None:
                args.feature_source = pre_metadata.get("feature_source")
            if args.multiomics_feature_path is None:
                args.multiomics_feature_path = pre_metadata.get("multiomics_feature_path")
            if args.cnv_missing_gene_path is None:
                args.cnv_missing_gene_path = pre_metadata.get("cnv_missing_gene_path")
            if args.standardize_multiomics is None:
                args.standardize_multiomics = pre_metadata.get("standardize_multiomics", False)
            args.use_multiomics = args.feature_source == "multiomics"
        else:
            pre_legacy = True
            if args.feature_source is None:
                raise ValueError("Legacy checkpoint has no metadata; please provide --feature-source explicitly.")
            if args.standardize_multiomics is None:
                args.standardize_multiomics = False
            args.use_multiomics = args.feature_source == "multiomics"
    seed = 1
    weightnumber = 50
    seed_torch(seed)
    # f = open("test_" + cancer + ".txt", "w")
    # for i in range(test_sta_number):
    #     if i in sample_index:
    #         test_sta.append(test_sta_total[i])
    #     else:
    #         test_sta2.append(test_sta_total[i])
    #         print(test_sta_total[i], file=f)
    # f.close()

    test_sta_total = load_gene_list(args.label_path) if args.label_path else []

    test_sta = test_sta_total
    test_sta2 = []

    train_patient_data, test_patient_data, patients = getInput(cancer, mutation_path=args.mutation_path)
    gene_data = getGene(patients)
    network, gene_final, gene_name = getNetworkall(gene_data, network_path=args.network_path)
    weights = getWeight(gene_name, weight_path=args.weight_path)

    # 🚀 关键修复 2：防弹装甲，补齐缺失的突变权重（防止报错 A1BG 找不到）
    for g in gene_name:
        if g not in weights:
            weights[g] = 0.0

    feature_bundle = build_identify_node_features(
        network,
        weights,
        gene_name,
        gene_final,
        args,
        pre_metadata,
        pre_legacy,
    )
    feature = feature_bundle["feature"]
    feature_source = feature_bundle["feature_source"]
    print(f"📌 当前节点特征来源：{feature_source}")
    print(f"📌 当前 node_features.shape：{feature.shape}")

    len_gene = len(gene_name)
    score_alpha = 0.5

    # 实例化模型大总管
    RL = DeepQNetwork(len_gene, network[:, :], feature[:, :], 64,
                      train_patient_data=train_patient_data,
                      test_patient_data=test_patient_data,
                      gene_sta=test_sta,
                      weights=weights,
                      learning_rate=0.001,
                      reward_decay=0.9,
                      e_greedy=0.95,
                      replace_target_iter=100,
                      memory_size=3000,
                      score_alpha=score_alpha
                      )
    gene_sort = run(gene_final, score_alpha, args)
