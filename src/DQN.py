import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from sklearn import preprocessing
from replay_buffer import PrioritizedReplayBuffer
from qfunction import Q_Fun

mpl.use('Agg')


class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            net_ori,
            fea_ori,
            embedding_size,
            train_patient_data,
            test_patient_data,
            gene_sta,
            weights,
            score_alpha,
            pat_num=0,
            learning_rate=0.0001,
            reward_decay=0.95,
            e_greedy=0.95,
            replace_target_iter=100,
            memory_size=50000,
            batch_size=128,
            e_greedy_increment=-0.000002,
            output_graph=False,
    ):
        self.net_ori = copy.deepcopy(net_ori)
        self.fea_ori = copy.deepcopy(fea_ori)
        self.train_patient_data = train_patient_data
        self.test_patient_data = test_patient_data
        self.train_cover = []
        self.test_cover = []
        self.weights = weights
        self.gene_sta = gene_sta
        self.actions = []
        self.actions_index = np.ones(n_actions)
        self.n_actions = n_actions
        self.embedding_size = embedding_size
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_min = 0.01
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 1.0
        self.score_be = 0
        self.score_sta = 0
        self.score_pat = 0
        self.learn_step_counter = 0
        self.a_ori = np.zeros((1, self.embedding_size))

        self.memory_counter = 0
        self.tau = 0.001

        self.gene_ori = []
        self.reward_all = 0
        self.reward_list = []
        self.train_sta = 0
        self.train_before = 0

        self.n_step = 4
        self.cost_his = []
        self.cost_his_emb = []
        self.cost_his_q = []
        T = 3
        ALPHA = 0.0001
        self.Q = Q_Fun(self.embedding_size, self.embedding_size, T, ALPHA, self.net_ori)
        self.Q_target = Q_Fun(self.embedding_size, self.embedding_size, T, ALPHA, self.net_ori)
        self.memory = PrioritizedReplayBuffer(
            self.memory_size,
            self.n_actions,
            alpha=0.2,
            beta_start=0.1,
            beta_frames=2000000,
            eps=1e-5,
        )

        self.score_alpha = score_alpha
        self.pat_num = pat_num
        self.embedding = None

        for name, param in self.Q.named_parameters():
            pass

    def laplacian(self, net):
        lap = copy.deepcopy(net)
        lap = lap * (-1)
        for i in range(net.shape[0]):
            lap[i][i] = np.sum(net[i])
        return lap

    def getBatch(self):
        selete = [x for x in range(self.memory_size - self.n_step) if self.memory_temp[x] == 1]
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(selete, size=self.batch_size)
        else:
            sample_index = np.random.choice(selete, size=self.batch_size)
        batch_r, q_next, batch_s_ = np.zeros((self.batch_size, 1)), np.zeros((self.batch_size, 1)), np.zeros(
            (self.batch_size, self.embedding_size))
        batch_a = self.memory_ar[sample_index, :self.embedding_size]
        batch_s = self.memory_s[sample_index, :]
        batch_fea = self.memory_fea[sample_index, :, :]
        batch_net = self.memory_net[sample_index, :, :]
        betch_lap = self.memory_lap[sample_index, :, :]
        for i in range(self.batch_size):
            index = sample_index[i]
            reward = 0
            for j in range(self.n_step):
                reward = reward + self.memory_ar[index + self.n_step, self.embedding_size]
            batch_r[i, :] = reward
            action_list = self.memory_actions[index + self.n_step, :]
            network = self.memory_net[index + self.n_step, :, :]
            feature = self.memory_fea[index + self.n_step, :, :]
            s_ = self.memory_s_[index + self.n_step, :]
            q_next[i, :] = self.get_train_Q(action_list, network, feature, s_)
            batch_s_[i, :] = self.memory_s_[index + self.n_step, :]
        return batch_fea, batch_net, betch_lap, batch_s, batch_a, batch_r, q_next, batch_s_

    def store_transition(self, s, a, r, s_, network_new1, feature_new1, lap):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack(([a, r]))

        index = self.memory_counter % self.memory_size
        self.memory_ar[index, :] = transition
        self.memory_s[index, :] = s
        self.memory_s_[index, :] = s_
        self.memory_fea[index, :] = feature_new1
        self.memory_net[index, :] = network_new1
        self.memory_lap[index, :] = lap
        self.memory_actions[index, :] = self.actions_index[:]
        if len(self.actions) >= 150 - self.n_step:
            self.memory_temp[index] = 0
        else:
            self.memory_temp[index] = 1
        self.memory_counter += 1

    def getState(self, feature_new, network_new):
        # 1. 执行TensorFlow计算图，获取所有节点的嵌入向量
        embedding = self.sess.run(self.emb_node,
                                  feed_dict={self.feature_ori: feature_new,
                                             self.net: network_new})
        # 2. 对所有节点的嵌入向量按维度求和，聚合为全局状态
        s = np.sum(embedding, axis=0, keepdims=True)
        # 3. 返回图的全局状态向量
        return s

    def getAction(self, network_new, action_selold, action_index):
        i = self.actions[-1]# 取最后一个被选中的节点（上一步的动作）
        action_selold.remove(i) # 从可选列表中删掉它（选过的不能再选）
        action_index[i] = 0# 标记这个节点：不可选
        # 遍历图中所有节点j
        for j in range(network_new.shape[0]):
            # 节点i和j是邻居、j不在可选列表里、j没被选过（不在历史动作里）
            if self.net_ori[i][j] > 0 and j not in action_selold and j not in self.actions:
                action_selold.append(j)# 加入可选列表
                action_index[j] = 1# 标记为可选

        if len(action_selold) == 0:# 如果可选列表空了
            for i in self.gene_ori:
            # 遍历所有原始节点
                if i not in self.actions:
                    action_selold.append(i)
                    action_index[i] = 1
        return action_selold, action_index

    def get_train_Q(self, action_list, network, feature, s_):
        nei_list = np.dot(action_list, self.net_ori)
        node_emb_all = self.sess.run(self.emb_node, feed_dict={self.feature_ori: feature,
                                                               self.net: network})
        action_sel = []
        for i in range(self.n_actions):
            if nei_list[i] > 0 and action_list[i] == 0:
                action_sel.append(i)
        action_emb = np.zeros((len(action_sel), self.embedding_size))
        for i in range(len(action_sel)):
            index = action_sel[i]
            action_emb[i, :] = node_emb_all[index, :]
        s = np.expand_dims(s_, 0)
        s_all = np.repeat(s, len(action_sel), axis=0)

        actions_value = self.sess.run(self.q_target_, feed_dict={self.s_: s_all,
                                                                 self.a_: action_emb})
        actions_value = np.squeeze(actions_value)
        qt = np.max(actions_value)
        return qt

    def choose_action(self, state, action_sel, action_index):
        # 判断是否有可选动作，有则执行决策
        if len(action_sel) > 0:
            state = torch.tensor(state, dtype=torch.float32).to(self.Q.device)
            action_index = torch.LongTensor(action_index).to(self.Q.device)
            # 调用在线Q网络，计算所有可选动作的Q值
            actions_value, self.embedding = self.Q(self.embedding, state, action_index)
            actions_value = actions_value.detach().cpu().numpy()
            # 返回所有可选动作的Q值评分
            return actions_value
        # 无可选动作时，返回0
        return 0

    def getQt(self, feature_new, network_new, s_):
        all_embedding = self.getEmbedding(feature_new, network_new)
        action_sel, action_emb = self.getAction(all_embedding, network_new)
        qt = -100
        if len(action_sel) != 0:
            s = np.expand_dims(s_, 0)
            s_all = np.repeat(s, len(action_sel), axis=0)
            actions_value = self.sess.run(self.q_target_, feed_dict={self.s_: s_all,
                                                                     self.a_: action_emb})
            actions_value = np.squeeze(actions_value)
            qt = np.max(actions_value)
        return qt

    def get_reward(self, gene_num, gene_name):
        weight_sum = 0 # 选中基因的总权重
        gene_name = list(gene_name)
        patient_num = [] # 选中基因关联的所有病人
        gene_sta_num = 0 # 选中的【目标基因】数量
        for i in self.actions:
            gene = gene_name[i] # 节点编号 → 转换成真实基因名
            # 统计1：如果选中的基因是【目标基因】，计数+1
            if gene in self.gene_sta:
                gene_sta_num = gene_sta_num + 1
            # 统计2：累加基因权重 + 收集关联病人
            if gene not in list(gene_num.keys()):
                # 基因无关联病人，只累加权重
                weight_sum = weight_sum + self.weights[gene]
            else:
                # 基因有关联病人，加入病人列表 + 累加权重
                patient_num.extend(gene_num[gene])
                weight_sum = weight_sum + self.weights[gene]
        return (weight_sum * self.n_actions / 150, # 奖励1：基因权重奖励
                (self.pat_num - len(set(patient_num))) / self.pat_num, # 奖励2：病人覆盖奖励
                (len(self.actions) - gene_sta_num) / len(self.actions))  # 奖励3：目标基因匹配奖励

    def getAcc(self, actions, patients, gene_name):
        cover_num = 0
        patients_num = len(patients.keys())
        for patient in patients:
            genes = patients[patient]
            for j in actions:
                if list(gene_name)[j] in genes:
                    cover_num += 1
                    break
        return cover_num / patients_num

    def Normalized_minmax(self, feature):
        X_scaler = preprocessing.MinMaxScaler()
        X_train = copy.deepcopy(feature)
        for i in range(len(feature[0])):
            X_train[:, [i]] = X_scaler.fit_transform(feature[:, [i]])
        return X_train

    def Normalized(self, feature):
        X_scaler = preprocessing.StandardScaler()
        X_train = X_scaler.fit_transform(feature)
        return X_train

    def get_feature(self, net, actions, weights, gene_name, gene_final):
        nodes_size = net.shape[0]
        feature = np.zeros((nodes_size, 3))
        i = 0
        for gene in list(gene_name):
            feature[i][0] = np.sum(net[i])
            feature[i][1] = weights[gene]
            if gene in list(gene_final.keys()):
                feature[i][2] = len(gene_final[gene])
            i = i + 1
        feature = self.Normalized_minmax(feature)
        return feature

    def step(self, network, action, gene_num, gene_name, weights):
        actions = self.actions[:]
        node_prop = len(actions)
        weight_sum, patient_num, gene_sta_num = self.get_reward(gene_num, gene_name)
        score_new = self.score_alpha * weight_sum + (1 - self.score_alpha) * patient_num * 3000

        score = self.score_be - score_new
        score_st = self.score_sta - gene_sta_num
        score_pa = self.score_pat - patient_num
        reward = 0
        if score > 0 or (self.score_be == 0 and score == 0):
            reward = reward + score_pa * 50
        if score_st > 0 or (self.score_sta == 0 and score_st == 0):
            reward = reward + 2

        self.reward_all = self.reward_all + reward
        self.score_be = score_new
        self.score_sta = gene_sta_num
        self.score_pat = patient_num
        done = 0
        if len(actions) == 999:
            self.embedding = None
            done = 1
            train_acc = self.getAcc(actions, self.train_patient_data, gene_name)
            self.train_cover.append(train_acc)
            test_acc = self.getAcc(actions, self.test_patient_data, gene_name)
            self.test_cover.append(test_acc)
            self.actions = []
            self.reward_list.append(reward)
            self.reward_all = 0
        return reward, done, actions

    def remember(self, *args):
        self.memory.store_transition(*args)
        self.memory_counter += 1

    def clear_mem(self):
        self.memory.clear()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        # ========== 从经验回放池采样一批经验 ==========
        # 对应：(S, A, R, S', 动作掩码)
        state, action, reward_sum, action_index, sel_action, sample_indices, is_weights = \
            self.memory.sample_buffer(self.batch_size)
        # 清空 Q 网络的梯度（上一步的梯度残留要清掉）
        self.Q.optimizer.zero_grad()
        mu = None
        action_temp = copy.deepcopy(action)
        action_index_new = copy.deepcopy(action_index)
        # 构造「下一状态 S'」的动作掩码（把当前动作 A 设为已选）
        for i in range(self.batch_size):
            action_temp_real = int(action_temp[i])
            action_index_new[i, action_temp_real] = 0

        state = torch.tensor(state, dtype=torch.float32).to(self.Q.device)
        action = torch.LongTensor(action).view(-1, 1).to(self.Q.device)
        reward_sum = torch.tensor(reward_sum, dtype=torch.float32).view(-1, 1).to(self.Q.device)
        reward_sum = reward_sum / 10.0
        new_state = state.clone()

        action_index = torch.LongTensor(action_index).to(self.Q.device)
        action_index_new = torch.LongTensor(action_index_new).to(self.Q.device)
        sel_action = torch.LongTensor(sel_action).to(self.Q.device)
        is_weights = torch.tensor(is_weights, dtype=torch.float32).to(self.Q.device)

                # ========== 计算目标 Q 值 y_t ==========
        # ========== 计算目标 Q 值 y_t (Double DQN 升级版) ==========
        with torch.no_grad():  # 眺望未来不需要计算梯度，省显存+加速
            # 第一步：在线网络 (self.Q) 当“玩家”，评估 S' 的所有动作
            next_q_values_online, _ = self.Q(mu, new_state, action_index_new, batch_flag=True)

            # ⚠️ 极其关键的“掩码(Masking)”操作：
            # action_index_new 中为 0 代表该基因已经入选，不能再挑了。
            # 我们把这些不可选基因的 Q 值强行变成 -1e9（极小值），防止 argmax 选错。
            mask = action_index_new == 0
            next_q_values_online = next_q_values_online.masked_fill(mask, -1e9)

            # 玩家做出决定：找出最高分的动作索引，对应公式里的 argmax a'
            best_next_actions = next_q_values_online.argmax(dim=1, keepdim=True)

            # 第二步：目标网络 (self.Q_target) 当“裁判”，对 S' 进行独立打分
            next_q_values_target, _ = self.Q_target(mu, new_state, action_index_new, batch_flag=True)
            next_q_target = next_q_values_target.gather(1, best_next_actions)

            y_target = reward_sum + self.gamma * next_q_target


        # ========== 计算当前 Q 值 Q(s,a;θ) ==========
        y_pred_all, _ = self.Q(
            mu,
            state,
            action_index,
            batch_flag=True
        )# 用 Q 网络计算当前状态 S 的所有 Q 值
        y_pred = y_pred_all.gather(1, action)# 只取「实际执行的动作 A」对应的 Q 值
        # ========== 计算损失函数 Loss  ==========
        # 对应论文公式：L = (y_t - Q(s,a;θ))²
        td_errors = y_target - y_pred

        elementwise_loss = torch.nn.functional.smooth_l1_loss(
            y_pred,
            y_target,
            reduction="none"
        )

        loss = torch.mean(is_weights * elementwise_loss)
        # ==========  反向传播更新 Q 网络权重 ==========
        loss.backward()
        # 🛡️ 新增：梯度裁剪防弹衣！强行把超过 1.0 的极端梯度削平，防止网络崩溃
        torch.nn.utils.clip_grad_norm_(self.Q.parameters(), max_norm=1.0)
        self.Q.optimizer.step()
        td_errors_np = td_errors.detach().abs().cpu().numpy()
        self.memory.update_priorities(sample_indices, td_errors_np)
        self.cost_his.append(loss.item())
        # 更新 epsilon（探索概率衰减：越往后，探索越少，利用越多）
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon > self.epsilon_min else self.epsilon_min
        self.learn_step_counter += 1
        # ========== 软更新目标 Q 网络 θ⁻ ==========
        # 每次 learn() 都让目标网络向当前 Q 网络平滑逼近一点点
        tau = self.tau # 平滑系数 (通常取 0.001 到 0.005 之间，你也可以把它写进 __init__ 中作为 self.tau)
        for target_param, local_param in zip(self.Q_target.parameters(), self.Q.parameters()):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )


    def save(self, path, cancer):
        torch.save(self.Q.state_dict(), ("{}/agent_" + cancer + ".th").format(path))

    def load(self, path):
        self.Q.load_state_dict(torch.load(path))

    def plot_cost(self, i):
        pass

    def plot_reward(self, i, score_PBRM1, score_MUC4, score_VHL):
        pass

    def plot_cost_finnal(self, i):
        pass