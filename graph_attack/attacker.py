import torch
import torch.nn.functional as F
from .base import EdgeAttacker


class AngleEdgeAttacker(EdgeAttacker):
    """Ad-hoc attack (type-0 attack in He et al)"""

    def compute_threshold(self, num_nodes, **kwargs):
        pass

    def infer_from_node_embedding(self, emb_matrix: torch.Tensor, **kwargs):
        emb_matrix = F.normalize(emb_matrix, dim=-1)
        default_threshold = self.compute_threshold(num_nodes=emb_matrix.shape[0])
        threshold = kwargs.get('threshold', default_threshold)
        angle_matrix = emb_matrix @ emb_matrix.T
        if threshold is None:
            self._reconstructed_adj = angle_matrix
        else:
            self._reconstructed_adj = (angle_matrix > threshold).long()

