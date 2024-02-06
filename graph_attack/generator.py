import torch
import torch_geometric
import numpy as np
import networkx as nx
import torch.nn.functional as F
from torch_geometric.data import Data


class EdgeProbGenerator(object):
    """Construct edge probability P[A_ij = 1] from node features"""

    def __init__(self, node_features: torch.Tensor):
        self.node_features = node_features
        normed_features = F.normalize(self.node_features, dim=1)
        self.node_pair_cos = normed_features @ normed_features.T

    def threshold_generator(self, threshold, prob_above, prob_below):
        cond = (self.node_pair_cos >= threshold).float()
        return prob_above * cond + prob_below * (1 - cond)

    # TODO: enable more generators
    def generate_prob(self, cfg):
        gen_type = cfg.graph.diagonal.f_base
        if gen_type == 'threshold':
            return self.threshold_generator(
                threshold=cfg.graph.diagonal.f_threshold,
                prob_above=cfg.graph.diagonal.f_p_above,
                prob_below=cfg.graph.diagonal.f_p_below
            )
        else:
            raise NotImplementedError


def get_simple_sbm_config(n, k, p, q):
    sizes = [n // k + (1 if x < n % k else 0) for x in range(k)]
    p_matrix = [[0 for _ in range(k)] for _ in range(k)]
    for row in range(k):
        for col in range(k):
            if row == col:
                p_matrix[row][col] = p
            else:
                p_matrix[row][col] = q
    return sizes, p_matrix


def gen_graph(cfg):
    graph_type = cfg.graph.graph_type
    n = cfg.graph.num_nodes
    x = np.random.normal(
        cfg.graph.feature_mean,
        cfg.graph.feature_std,
        size=[n, cfg.graph.num_features],
    )
    if graph_type == 'erdos_renyi':
        nxg = nx.erdos_renyi_graph(
            n=n,
            p=cfg.graph.erdos_renyi.p
        )
    elif graph_type == 'sbm':
        sizes, p = get_simple_sbm_config(
            n=n, k=cfg.graph.sbm.k, p=cfg.graph.sbm.p, q=cfg.graph.sbm.q
        )
        nxg = nx.stochastic_block_model(
            sizes=sizes,
            p=p
        )
    elif graph_type == 'diagonal':
        # Manually construct adjacency matrix
        x = torch.from_numpy(x).float()
        adj_prob = EdgeProbGenerator(node_features=x).generate_prob(cfg)
        adj_matrix = (torch.rand(adj_prob.shape) <= adj_prob)
        # Remove self-loops, make undirected
        adj_matrix = torch.tril(adj_matrix, -1) + torch.tril(adj_matrix, -1).T
        edge_index = torch.stack(torch.where(adj_matrix), dim=0)
        return torch_geometric.data.Data(
            x=x,
            edge_index=edge_index
        )
    else:
        raise NotImplementedError(f'Unknown graph type {graph_type}')
    gdata = torch_geometric.utils.from_networkx(nxg)
    gdata.x = torch.from_numpy(x).float()
    return gdata

