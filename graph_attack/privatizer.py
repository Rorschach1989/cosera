import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch_geometric.utils as tu
from torch_geometric.data import Data
from torch_geometric.loader import RandomNodeLoader
from torch_geometric.transforms import BaseTransform


class EdgeRR(BaseTransform):
    """Implements randomized response with given (local) privacy level"""

    def __init__(self, eps):
        self.eps = eps
        self.flip_prob = 1 / (1 + np.exp(eps))

    def forward(self, data: Data) -> Data:
        num_nodes = data.num_nodes
        n_2 = num_nodes ** 2
        total_edge_flips = np.random.binomial(
            n_2, self.flip_prob
        )  # Better not be too large
        g = np.random.Generator(np.random.PCG64())
        flip_ids = g.choice(n_2, total_edge_flips, replace=False)
        row, col = np.divmod(flip_ids, num_nodes)
        # Filter the lower triangular
        tril_ids = row > col
        row, col = row[tril_ids], col[tril_ids]
        flip_ids = np.concatenate(
            [
                np.stack([row, col], axis=0),
                np.stack([col, row], axis=0)
            ],
            axis=1
        )
        flip_tensor = torch.sparse_coo_tensor(
            indices=torch.from_numpy(flip_ids).long(),
            values=-torch.ones(flip_ids.shape[1]),
            size=(num_nodes, num_nodes)
        )
        raw_edges = tu.to_torch_coo_tensor(data.edge_index)
        edges = (flip_tensor + raw_edges).coalesce()
        edges = torch.sparse_coo_tensor(
            indices=edges.indices(),
            values=edges.values(),
            size=edges.size()
        )
        edge_index, edge_values = tu.to_edge_index(edges.coalesce())
        edge_index = edge_index.masked_select(edge_values.bool()).view(2, -1)
        out_data = data.clone()
        out_data.edge_index = edge_index
        return out_data


class NoisyEmbedding(nn.Module):
    """Implements noisy embedding without explicitly doing DP-accounting"""

    def __init__(self, gnn: nn.Module, sigma, normalize=True):
        super(NoisyEmbedding, self).__init__()
        self.gnn = gnn
        self.sigma = sigma
        self.normalize = normalize

    def forward(self, x, edge_index):
        if self.normalize:
            x = F.normalize(x, dim=1)
        x = self.gnn(x, edge_index)
        x = x + torch.randn(x.shape, device=x.device) * self.sigma
        return x


class NodeLoaderHelper(RandomNodeLoader):
    """Helper class for data loading with optional EdgeRR transform"""

    def __init__(self, data, num_parts, **kwargs):
        eps = kwargs.pop('eps', None)
        if eps is not None:
            # Requires DP transform
            self.transform = EdgeRR(eps)
        else:
            self.transform = None
        super(NodeLoaderHelper, self).__init__(
            data,
            num_parts,
            **kwargs
        )

    def collate_fn(self, index):
        data = super(NodeLoaderHelper, self).collate_fn(index)
        return data if self.transform is None else self.transform(data)
