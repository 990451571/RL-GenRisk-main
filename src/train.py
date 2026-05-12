import numpy as np
import copy
import random
import torch
import os
import inputall
import matplotlib.pyplot as plt
from DQN import DeepQNetwork


def plot_training_results(rewards, losses, save_path="../data/RL_GenRisk_Training_Curves_PER.png"):
    """
    一键生成并保存双拼训练曲线图（奖励值 & 损失值）
    """
    plt.figure(figsize=(12, 5))

    # ========== 图 1：累计奖励曲线 ==========
    plt.subplot(1, 2, 1)
    plt.plot(rewards, color='#1f77b4', linewidth=1.5, alpha=0.9)
    plt.title("Cumulative Reward per Episode", fontsize=14, fontweight='bold')
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Reward", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # ========== 图 2：损失函数曲线 ==========
    plt.subplot(1, 2, 2)
    if len(losses) > 100:
        window = 50
        smoothed_losses = np.convolve(losses, np.ones(window) / window, mode='valid')
        plt.plot(smoothed_losses, color='#d62728', linewidth=1.5, alpha=0.9)
    else:
        plt.plot(losses, color='#d62728', linewidth=1.5, alpha=0.9)

    plt.title("Training Loss (Smoothed)", fontsize=14, fontweight='bold')
    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # ========== 保存图片 ==========
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📈 训练图表已自动生成并保存为：{save_path}")

def train_model():

    print("🚀 正在加载原始数据集，这可能需要一点时间...")
    cancer = "KIRC"

    # 加载数据
    train_data, test_data, patients = inputall.getInput(cancer)
    gene_num = inputall.getGene(patients)
    net, gene_final, gene_name = inputall.getNetwork(gene_num)
    weights = inputall.getWeight(gene_name)
    # 🩹 补丁 1：填补缺失的基因权重，防止 KeyError
    for g in gene_name:
        if g not in weights:
            weights[g] = 0.0

    gene_sta = []
    with open("../data/sta_ccRCC_Merged.txt", 'r') as f:
        for line in f.readlines():
            gene_sta.append(line.strip())

    print(f"✅ 数据加载完成！共有 {len(gene_name)} 个节点参与训练。")

    n_actions = len(gene_name)
    fea_ori = np.zeros((n_actions, 3))

    RL = DeepQNetwork(
        n_actions=n_actions,
        net_ori=net,
        fea_ori=fea_ori,
        embedding_size=64,
        train_patient_data=train_data,
        test_patient_data=test_data,
        gene_sta=gene_sta,
        weights=weights,
        score_alpha=0.5,
        pat_num=len(patients.keys())
    )

    feature = RL.get_feature(net, [], weights, gene_name, gene_final)

    MAX_EPISODES = 500  # 训练轮数
    MAX_STEPS = 150
    WARMUP_STEPS = 1000

    print(f"🔥 开始模型训练，目标 {MAX_EPISODES} 轮...")
    episode_rewards_history = []

    for episode in range(MAX_EPISODES):
        RL.epsilon = max(0.1, 1.0 - episode / 100)

        action_sel = [i for i in range(RL.n_actions)]
        RL.actions = []
        RL.actions_index = np.ones(n_actions)

        RL.score_be = 0
        RL.score_sta = 0
        RL.score_pat = 0
        RL.embedding = None

        episode_reward = 0
        s = feature
        step_count = 0

        while True:
            state_tensor = torch.tensor(s, dtype=torch.float32).to(RL.Q.device)
            action_mask_tensor = torch.LongTensor(RL.actions_index).to(RL.Q.device)

            actions_value, emb = RL.Q(RL.embedding, state_tensor, action_mask_tensor)
            actions_value_np = actions_value.detach().cpu().numpy()
            RL.embedding = emb.detach()

            valid_actions = [a for a in action_sel if RL.actions_index[a] == 1]

            if len(valid_actions) == 0:
                done = 1
                RL.embedding = None
                break

            action_best = max(valid_actions, key=lambda idx: actions_value_np[idx])

            if np.random.uniform() >= RL.epsilon:
                action_index = int(action_best)
            else:
                action_index = int(random.choice(valid_actions))

            if action_index in action_sel:
                action_sel.remove(action_index)
            else:
                continue

            RL.actions.append(action_index)
            RL.actions_index[action_index] = 0

            reward, done, actions = RL.step(
                net,
                action_index,
                gene_num,
                gene_name,
                weights
            )

            episode_reward += reward

            sel_action = copy.deepcopy(RL.actions_index)
            RL.remember(s, action_index, reward, RL.actions_index, sel_action)

            if len(RL.memory) > max(RL.batch_size, WARMUP_STEPS):
                RL.learn()

            step_count += 1

            if step_count >= MAX_STEPS:
                done = 1
                RL.embedding = None

            if done == 1:
                print(
                    f"🏁 第 {episode + 1}/{MAX_EPISODES} 轮结束！"
                    f"累计获得奖励: {episode_reward:.2f} | "
                    f"epsilon: {RL.epsilon:.3f} | "
                    f"buffer: {len(RL.memory)}"
                )
                episode_rewards_history.append(episode_reward)
                break


    print("💾 训练完成！正在保存模型权重...")
    RL.save("../data", "ccRCC_train_PER")
    print("🎉 恭喜！模型已成功保存为 agent_ccRCC_retrain.th！")
    plot_training_results(episode_rewards_history, RL.cost_his)




if __name__ == "__main__":
    train_model()