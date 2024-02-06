import torch
import numpy as np
from scipy.special import expit as sigmoid


class EdgeAttacker(object):
    """Interface for an edge-level attack"""

    def __init__(self):
        self._reconstructed_adj = None  # In case we may query some api multiple times and strengthen attack

    def reset_reconstruction(self):
        self._reconstructed_adj = None

    def infer_from_node_embedding(self, emb_matrix: torch.Tensor, **kwargs):
        raise NotImplementedError

    def get_non_diagonal_prediction(self, return_numpy=True):
        assert self._reconstructed_adj is not None
        num_nodes = self._reconstructed_adj.shape[0]
        pred_values = self._reconstructed_adj[np.tril_indices(num_nodes, -1)]
        return sigmoid(pred_values) if not return_numpy else pred_values.cpu().numpy()

