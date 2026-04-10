import numpy as np
import copy
import random
import torch
import inputall
from DQN import DeepQNetwork


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

    MAX_EPISODES = 1000  # 训练轮数
    print(f"🔥 开始模型训练，目标 {MAX_EPISODES} 轮...")

    for episode in range(MAX_EPISODES):
        # 初始化状态遮罩
        action_sel = [i for i in range(RL.n_actions)]
        RL.actions = []
        RL.actions_index = np.ones(n_actions)

        # 🛡️ 补丁 2：防弹装甲！强制打底所有可能被底层代码漏掉的计分器变量
        RL.score_be = 0
        RL.score_sta = 0
        RL.score_pat = 0

        # 🌟 我们自己建一个独立的计分板！彻底抛弃底层的 RL.reward_all
        episode_reward = 0

        s = feature
        step_count = 0

        while True:
            # 1. 纯净前向传播
            state_tensor = torch.tensor(s, dtype=torch.float32).to(RL.Q.device)
            action_mask_tensor = torch.LongTensor(RL.actions_index).to(RL.Q.device)

            actions_value, emb = RL.Q(RL.embedding, state_tensor, action_mask_tensor)
            actions_value_np = actions_value.detach().cpu().numpy()
            RL.embedding = emb.detach()

            # 2. 纯 Python 选拔逻辑
            action_best = max(action_sel, key=lambda idx: actions_value_np[idx])

            if np.random.uniform() >= RL.epsilon:
                action_index = int(action_best)
            else:
                if len(action_sel) == 1:
                    action_index = int(action_sel[0])
                else:
                    while True:
                        rand_idx = random.choice(action_sel)
                        if rand_idx != action_best:
                            action_index = int(rand_idx)
                            break

            # 3. 稳健写入历史
            RL.actions.append(action_index)
            RL.actions_index[action_index] = 0
            action_sel.remove(action_index)

            # 4. 执行动作 (拿到本步的真实奖励)
            reward, done, actions = RL.step(net, action_index, gene_num, gene_name, weights)

            # 🌟 每次获得奖励，都立刻记在咱们自己的账上！
            episode_reward += reward

            sel_action = copy.deepcopy(RL.actions_index)
            RL.remember(s, action_index, reward, RL.actions_index, sel_action)

            # 5. 模型更新
            if RL.memory_counter > RL.batch_size:
                RL.learn()

            step_count += 1
            if done == 1:
                # 🌟 打印咱们自己记录的总分，不管 DQN 怎么清零，咱们的分数都不会丢！
                print(f"🏁 第 {episode + 1}/{MAX_EPISODES} 轮结束！累计获得奖励: {episode_reward:.2f}")
                break

    print("💾 训练完成！正在保存模型权重...")
    RL.save("../data", "ccRCC_retrain")
    print("🎉 恭喜！模型已成功保存为 agent_ccRCC_retrain.th！")


if __name__ == "__main__":
    train_model()