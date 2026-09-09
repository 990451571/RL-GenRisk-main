from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.utils import scatter as scatter_add


def change(matrix):
    """Convert a dense adjacency matrix to PyG edge_index efficiently."""
    row, col = np.nonzero(np.asarray(matrix))
    return np.vstack([row, col]).astype(np.int64, copy=False)


class Q_Fun(nn.Module):
    def __init__(self, in_dim, hid_dim, T, ALPHA, net_old, preference_dim=0, output_dim=1):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hid_dim = int(hid_dim)
        self.T = T
        self.preference_dim = int(preference_dim)
        if self.preference_dim < 0:
            raise ValueError(f"preference_dim must be non-negative, got {preference_dim!r}")
        self.output_dim = int(output_dim)
        if self.output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim!r}")

        Linear = partial(nn.Linear, bias=True)
        self.lin1 = Linear(self.in_dim, self.hid_dim)
        self.conv1 = GCNConv(self.hid_dim, self.hid_dim)
        self.conv2 = GCNConv(self.hid_dim, self.hid_dim)
        self.lin2 = Linear(3 * self.hid_dim, self.hid_dim)
        self.lin3 = Linear(self.in_dim, self.hid_dim)
        self.lin4 = Linear(self.hid_dim, self.hid_dim)
        self.preference_encoder = (
            Linear(self.preference_dim, self.hid_dim) if self.preference_dim else None
        )
        self.lin5 = Linear(self.hid_dim * (3 if self.preference_dim else 2), self.hid_dim)
        self.lin6 = Linear(self.hid_dim, self.hid_dim)
        self.lin7 = Linear(self.hid_dim, self.hid_dim)
        self.lin8 = Linear(self.hid_dim, self.output_dim)
        self.dropout = nn.Dropout(p=0.2)

        net_array = np.asarray(net_old)
        if net_array.ndim != 2 or net_array.shape[0] != net_array.shape[1]:
            raise ValueError(f"net_old must be a square adjacency matrix, got {net_array.shape}.")
        self.n_actions = int(net_array.shape[0])

        # Register edge_index as a buffer so model.to(device) moves it together
        # with all parameters. This also makes --device cpu safe on CUDA machines.
        edge_index = torch.as_tensor(change(net_array), dtype=torch.long)
        self.register_buffer("net_old", edge_index, persistent=False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=ALPHA)

    def _prepare_preference(self, preference, batch_size, num_nodes, device, dtype):
        if not self.preference_dim:
            return None
        if preference is None:
            raise ValueError("A preference vector is required for preference-conditioned Q_Fun.")
        pref = torch.as_tensor(preference, device=device, dtype=dtype)
        expected = (batch_size, self.preference_dim) if batch_size is not None else (self.preference_dim,)
        if tuple(pref.shape) != expected:
            raise ValueError(f"preference must have shape {expected}, got {tuple(pref.shape)}.")
        if batch_size is None:
            encoded = self.preference_encoder(pref).unsqueeze(0).expand(num_nodes, -1)
        else:
            encoded = self.preference_encoder(pref).unsqueeze(1).expand(-1, num_nodes, -1)
        return encoded

    def forward(self, mu, x, action_sel, batch_flag=False, test_flag=False, preference=None):
        if mu is None:
            x_1 = self.lin1(x)

            x_2 = self.conv1(x_1, self.net_old)
            x_2 = F.relu(x_2)
            x_2 = self.dropout(x_2)

            x_3 = self.conv2(x_2, self.net_old)
            x_3 = F.relu(x_3)
            x_3 = self.dropout(x_3)
            nodes_vec = self.lin2(torch.cat([x_1, x_2, x_3], dim=-1))
        else:
            nodes_vec = mu

        num_nodes = self.n_actions
        if not batch_flag:
            idx = action_sel.long()
            if idx.ndim != 1 or idx.numel() != num_nodes:
                raise ValueError(
                    f"Non-batch action mask must have shape ({num_nodes},), got {tuple(idx.shape)}."
                )
            graph_pool2 = scatter_add(nodes_vec, idx, dim=-2, dim_size=2)[0]
            graph_pool2 = graph_pool2.repeat(num_nodes, 1)
        else:
            idx = action_sel.long()
            if idx.ndim != 2 or idx.shape[1] != num_nodes:
                raise ValueError(
                    f"Batch action mask must have shape (batch, {num_nodes}), got {tuple(idx.shape)}."
                )
            idx_expanded = idx.unsqueeze(-1).expand_as(nodes_vec)
            out = torch.zeros(
                nodes_vec.size(0),
                2,
                nodes_vec.size(2),
                device=nodes_vec.device,
                dtype=nodes_vec.dtype,
            )
            out.scatter_add_(1, idx_expanded, nodes_vec)
            graph_pool2 = out[:, [0], :].repeat(1, num_nodes, 1)

        pref_vec = self._prepare_preference(
            preference,
            nodes_vec.shape[0] if batch_flag else None,
            num_nodes,
            nodes_vec.device,
            nodes_vec.dtype,
        )
        pieces = (self.lin6(graph_pool2), nodes_vec) if pref_vec is None else (self.lin6(graph_pool2), nodes_vec, pref_vec)
        cat = torch.cat(pieces, dim=-1)
        q_values = self.lin8(F.relu(self.lin5(F.relu(cat))))
        if self.output_dim == 1:
            q_values = q_values.squeeze(-1)
        return q_values, nodes_vec

