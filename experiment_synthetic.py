import torch
import torch_geometric
import numpy as np
from tqdm import tqdm

from graph_attack.utils import HyperParamManager, parse_metrics
from graph_attack.generator import gen_graph
from graph_attack.attacker import AngleEdgeAttacker
from graph_attack.nn import SimpleGraphConv


def run_exp(cfg):
    torch.manual_seed(cfg.exp.seed)
    np.random.seed(cfg.exp.seed)

    def _run():
        # Generate graph data using networkx interface
        n = cfg.graph.num_nodes
        d = cfg.graph.num_features
        gdata = gen_graph(cfg)
        # Produce gnn embeddings
        with torch.no_grad():
            gnn = SimpleGraphConv.from_cfg(cfg)
            emb_matrix = gnn(gdata)
            # Do the reconstruction
            attacker = AngleEdgeAttacker()
            attacker.infer_from_node_embedding(emb_matrix)
            y_score = attacker.get_non_diagonal_prediction(return_numpy=True)
            y_true = torch_geometric.utils.to_dense_adj(
                gdata.edge_index,
                max_num_nodes=n
            ).squeeze(0).cpu().numpy()[np.tril_indices(n, -1)]
            m = parse_metrics(cfg)
        return m(y_score=y_score, y_true=y_true)

    scores = [_run() for _ in range(cfg.exp.num_trials)]
    return np.mean(scores), np.std(scores)


if __name__ == '__main__':
    manager = HyperParamManager()
    manager.register_field(
        key='num_layers',
        value=list(range(1, 11))
    )
    manager.register_field(
        key='num_features',
        value=[2 ** i for i in range(2, 12)]
    )
    manager.register_field(
        key='num_trials',
        value=5
    )
    manager.register_field(
        key='require_weights',
        value=True
    )
    for graph_type in ['sbm']:
        for num_nodes in [100, 500, 1000]:
            m = manager.clone()
            m.register_field(
                key='graph_type',
                value=graph_type
            )
            m.register_field(
                key='num_nodes',
                value=num_nodes
            )
            m.register_field(
                key='metric',
                # value='fp@r100'
                # value='auc'
                value='err_sum'
            )
            for cfg in m.iter_configs():
                mean, std = run_exp(cfg)
                m.record_result(
                    cfg=cfg,
                    result={
                        'mean': mean,
                        'std': std
                    }
                )


