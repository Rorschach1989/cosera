import torch
import torch_geometric
import numpy as np
import torch_geometric.datasets as td
from sklearn.metrics import roc_auc_score

from graph_attack.nn import NoisyGNN
from graph_attack.attacker import AngleEdgeAttacker
from graph_attack.utils import HyperParamManager, parse_metrics, default_device
from graph_attack.privatizer import EdgeRR


def run_exp(cfg):
    torch.manual_seed(cfg.exp.seed)
    dataset = td.Planetoid(root=cfg.dataset.root, name=cfg.dataset.name).data.to(default_device)
    n = dataset.num_nodes

    def _run():
        gnn = NoisyGNN.from_cfg(cfg)
        # Do the training
        optimizer = torch.optim.Adam(gnn.parameters(), lr=cfg.train.lr)
        loss_fn = torch.nn.NLLLoss().to(default_device)
        transform = EdgeRR(cfg.privacy.eps) if cfg.privacy.method == 'edge_rr' else None
        epochs = cfg.train.epochs
        gnn.train()
        for _ in range(epochs):
            # for data in loader:
            data = transform(dataset) if transform is not None else dataset
            optimizer.zero_grad()
            pred, _ = gnn(data)
            loss = loss_fn(
                pred[dataset.train_mask],
                dataset.y[dataset.train_mask]
                # dataset.y[dataset.train_mask]
            )
            loss.backward()
            optimizer.step()
        gnn.eval()
        with torch.no_grad():
            data = transform(dataset) if transform is not None else dataset
            logits, emb_matrix = gnn(data)
            pred = logits[dataset.test_mask].argmax(dim=1)
            correct = (pred == dataset.y[dataset.test_mask]).sum()
            acc = int(correct) / int(dataset.test_mask.sum())
        # Do the attack
        attacker = AngleEdgeAttacker()
        attacker.infer_from_node_embedding(emb_matrix[dataset.test_mask, :])
        y_score = attacker.get_non_diagonal_prediction(return_numpy=True)
        edge_index, _ = torch_geometric.utils.subgraph(
            subset=torch.where(dataset.test_mask)[0],
            edge_index=dataset.edge_index,
            relabel_nodes=True
        )
        n_test = int(dataset.test_mask.sum())
        y_true = torch_geometric.utils.to_dense_adj(
            edge_index,
            max_num_nodes=n_test
        ).squeeze(0).cpu().numpy()[np.tril_indices(n_test, -1)]
        m = [parse_metrics(cfg), roc_auc_score]
        attack_metric = [_m(y_score=y_score, y_true=y_true) for _m in m]
        return acc, attack_metric, gnn.weight_spectrum

    results = [_run() for _ in range(cfg.exp.num_trials)]
    # return np.mean(results, axis=0), np.std(results, axis=0)
    return results


if __name__ == '__main__':
    from graph_attack.utils import setup_cfg_public

    manager = HyperParamManager(cfg_setter=setup_cfg_public)
    # manager.register_field(
    #     key='dataset',
    #     value='Cora'
    # )
    manager.register_field(
        key='num_layers',
        value=2
    )
    manager.register_field(
        key='metric',
        # value='auc'
        value='err_sum'
    )
    manager.register_field(
        key='d_hidden',
        value=[2 ** i for i in range(5, 14)]
    )
    manager.register_field(
        key='defense_method',
        value='noisy_embedding'
        # value='edge_rr'
    )
    manager.register_field(
        key='normalize',
        value=True
    )
    manager.register_field(
        key='epochs',
        value=1000
    )
    manager.register_field(
        key='num_trials',
        value=5
    )
    manager.register_field(
        key='spectral_norm',
        value=True
    )
    for gnn_type in ['kw_gcn', 'mp_gcn', 'gat', 'gin', 'max-sage']:
        for dataset in ['Cora', 'Citeseer', 'Pubmed']:
            for sigma in [0., 0.01, 0.05, 0.1, 0.5, 1.]:
                m = manager.clone()
                m.register_field(
                    key='dataset',
                    value=dataset
                )
                m.register_field(
                    key='gnn_type',
                    value=gnn_type
                )
                m.register_field(
                    key='sigma',
                    value=sigma
                )
                for cfg in m.iter_configs():
                    accs, adv_metrics, spectrum = zip(*run_exp(cfg))
                    m.record_result(
                        cfg,
                        result={
                            'accs': accs,
                            'adv_metrics': adv_metrics,
                            # 'spectrum': spectrum
                        },
                        mode='real'
                    )


