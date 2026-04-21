import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv
from torch_geometric.utils import scatter as scatter_add
from functools import partial


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def change(matrix):
    leng = len(matrix)
    result0 = []
    result1 = []
    for i in range(leng):
        for j in range(leng):
            if matrix[i][j] != 0:
                result0.append(i)
                result1.append(j)
    result = [result0, result1]
    return result


class Q_Fun(nn.Module):
    def __init__(self, in_dim, hid_dim, T, ALPHA, net_old):
        super(Q_Fun, self).__init__()
        self.in_dim = in_dim   # 节点输入特征维度（3维）
        self.hid_dim = hid_dim # 网络隐藏层维度
        # 图卷积层（GCN核心：提取基因的拓扑特征）
        self.conv1 = GCNConv(hid_dim, hid_dim) # 第一层GCN：获取一阶邻居信息
        self.conv2 = GCNConv(hid_dim, hid_dim) # 第二层GCN：获取二阶邻居信息
        self.T = T
        # 全连接层（MLP：特征映射+Q值计算）
        Linear = partial(nn.Linear, bias=True)
        self.lin1 = Linear(3, hid_dim) # 输入：3维原始特征 → 映射到隐藏层
        self.lin2 = Linear(3 * hid_dim, hid_dim) # 拼接3层GCN特征 → 压缩
        self.lin3 = Linear(3, hid_dim)
        self.lin4 = Linear(hid_dim, hid_dim)
        self.lin5 = Linear(hid_dim * 2, hid_dim) # 拼接全局状态+节点特征
        self.lin6 = Linear(hid_dim, hid_dim) # 全局状态特征映射
        self.lin7 = Linear(hid_dim, hid_dim)
        self.lin8 = Linear(hid_dim, 1) # 最终输出：1维Q值
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net_old = torch.tensor(change(net_old)).long().to(self.device)
        self.net_old2 = torch.tensor(net_old).long().to(self.device)
        self.n_actions = self.net_old2.shape[0]
        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=ALPHA)

    def forward(self, mu, x, action_sel, batch_flag=False, test_flag=False):

        if mu is None:
            x_in = x.clone()
            x_1 = self.lin1(x)
            self.dropout = nn.Dropout(p=0.5, inplace=False)
            x_2 = self.conv1(x_1, self.net_old)
            x_2 = x_2.relu()
            self.dropout = nn.Dropout(p=0.5, inplace=False)
            x_3 = self.conv2(x_2, self.net_old)
            x_3 = x_3.relu()
            nodes_vec = self.lin2(torch.cat([x_1, x_2, x_3], dim=-1))
        else:
            nodes_vec = mu
        num_nodes = self.n_actions
        if not batch_flag:
            # 把已经挑出来的基因特征加起来，形成一个代表当前大环境的“全局状态”
            graph_pool2 = scatter_add(nodes_vec, action_sel, dim=-2)[0]
            number = len(action_sel) - torch.sum(action_sel)
            if test_flag:
                number = 1
            graph_pool2 = (graph_pool2).repeat(num_nodes, 1)  # 把全局状态复制N份，每个基因都配一份

        else:
            # 批量训练：一次处理128个样本，给每个样本单独算全局状态
            idx = action_sel.long()
            if idx.dim() == 2:
                idx = idx.unsqueeze(-1)  # 把形状从 [128, 9039] 变成 [128, 9039, 1]
            idx = idx.expand_as(nodes_vec)  # 扩展到 [128, 9039, 64] 和特征维度对齐
            # 批量scatter_add：每个样本的已选基因单独求和
            num_bins = idx.max().item() + 1
            out = torch.zeros(nodes_vec.size(0), num_bins, nodes_vec.size(2), device=nodes_vec.device,
                              dtype=nodes_vec.dtype)
            out.scatter_add_(1, idx, nodes_vec)  # 在维度 1 (也就是 9039 那个维度) 上进行池化求和
            graph_pool2 = out[:, [0]]
            number = num_nodes - torch.sum(action_sel, 1, keepdim=True).unsqueeze(1)
            if test_flag:
                number = 1
            # 同样复制全局状态给每个基因
            graph_pool2 = (graph_pool2).repeat(1, num_nodes, 1)
        # 把全局状态 和 每一个候选基因自己的状态 拼在一起
        Cat = torch.cat((self.lin6(graph_pool2), nodes_vec), dim=-1)
        # 通过最后的神经元，输出最终打分！
        return self.lin8(F.relu(self.lin5(F.relu(Cat)))).squeeze(), nodes_vec

