import torch
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

from .utils import default_device
from .privatizer import NoisyEmbedding


class MLP(nn.Module):

    device = default_device

    def __init__(self, num_layers, d_input, d_hidden, d_output):
        super(MLP, self).__init__()
        network = []
        for i in range(num_layers - 1):
            in_features = d_input if i == 0 else d_hidden
            network.append(nn.Linear(in_features, d_hidden))
            if i < num_layers - 1:
                network.append(nn.ReLU())
        self.mlp = nn.Sequential(*network)
        self.out_proj = nn.Linear(d_hidden, d_output)
        self.to(self.device)

    def forward(self, gdata):
        x = self.mlp(gdata.x)
        return F.log_softmax(self.out_proj(x), dim=1), x

    @classmethod
    def from_cfg(cls, cfg):
        return cls(
            num_layers=cfg.gnn.num_layers,
            d_input=cfg.gnn.d_input,
            d_hidden=cfg.gnn.d_hidden,
            d_output=cfg.gnn.d_output,
        )


class SimpleMeanPool(nn.Module):
    """Minimal clean impl of mean-pool GCN"""

    def __init__(self, in_channels, out_channels, require_weights, aggr, spectral_norm):
        super(SimpleMeanPool, self).__init__()
        self.conv = gnn.SimpleConv(aggr=aggr)
        if require_weights:
            lin = nn.Linear(in_channels, out_channels, bias=False)
            if spectral_norm:
                lin = nn.utils.spectral_norm(lin)
            self.lin = lin
        else:
            self.lin = None

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        return self.lin(x) if self.lin is not None else x

    @property
    def weight_op_norm(self):
        if self.lin is None:
            return 1.
        else:
            return torch.linalg.norm(self.lin.weight, ord=2)


class SimpleGraphConv(nn.Module):
    """A simplest setup"""

    device = default_device

    @staticmethod
    def get_rdp_eps(sigma, alpha, weight_op_norm):
        """Compute Renyi DP guarantee under the mean-pooling paradigm"""
        return alpha * (weight_op_norm ** 2) / (sigma ** 2)

    def __init__(self, num_layers, num_features, aggr, require_weights, spectral_norm, self_loop, sigma=0., alpha=1.):
        super(SimpleGraphConv, self).__init__()
        self.require_weights = require_weights
        self.num_layers = num_layers
        self.num_features = num_features
        self.self_loop = self_loop
        self.mp_layers = nn.ModuleList()
        # The eps is evaluated at renyi privacy
        self.eps = None  # By default no privacy configuration
        self.alpha = alpha
        for _ in range(num_layers):
            layer = SimpleMeanPool(
                in_channels=num_features,
                out_channels=num_features,
                require_weights=require_weights,
                aggr=aggr,
                spectral_norm=spectral_norm
            )
            if sigma > 0:
                if self.eps is None:
                    self.eps = self.get_rdp_eps(sigma, alpha, layer.weight_op_norm)
                else:
                    self.eps += self.get_rdp_eps(sigma, alpha, layer.weight_op_norm)
                layer = NoisyEmbedding(layer, sigma)
            self.mp_layers.append(layer)
        self.to(self.device)

    def forward(self, gdata):
        x, edge_index = gdata.x.to(self.device), gdata.edge_index.to(self.device)
        if self.self_loop:
            edge_index, _ = torch_geometric.utils.add_self_loops(edge_index)
        for layer in self.mp_layers:  # Linear configuration, no act required
            x = layer(x, edge_index)
        return x

    @classmethod
    def from_cfg(cls, cfg):
        """Initialize from OmegaConf"""
        return cls(
            num_layers=cfg.gnn.num_layers,
            num_features=cfg.graph.num_features,
            aggr=cfg.gnn.aggr,
            require_weights=cfg.gnn.require_weights,
            spectral_norm=cfg.gnn.spectral_norm,
            self_loop=cfg.gnn.self_loop,
            sigma=cfg.privacy.sigma,  # Allow privacy-preserving perturbation in synthesis mode
            alpha=cfg.privacy.alpha,
        )


class NoisyGNN(nn.Module):
    """GNN with optional noisy perturbation"""

    device = default_device

    @staticmethod
    def _get_gnn(gnn_type, in_channels, out_channels, spectral_norm, **kwargs):
        if gnn_type == 'mp_gcn':
            return SimpleMeanPool(
                in_channels,
                out_channels,
                require_weights=True,
                spectral_norm=spectral_norm,
                aggr='mean'
            )
        elif gnn_type == 'kw_gcn':
            gcn = gnn.GCNConv(in_channels, out_channels)
            if spectral_norm:
                gcn.lin = nn.utils.spectral_norm(gcn.lin)
            return gcn
        elif gnn_type == 'gin':
            return SimpleMeanPool(
                in_channels,
                out_channels,
                require_weights=True,
                spectral_norm=spectral_norm,
                aggr='sum'
            )
        elif gnn_type == 'max-sage':
            return SimpleMeanPool(
                in_channels,
                out_channels,
                require_weights=True,
                spectral_norm=spectral_norm,
                aggr='max'
            )
        elif gnn_type == 'gat':
            n_heads = kwargs.get('n_heads', 1)
            gat = gnn.GATConv(in_channels, out_channels, heads=n_heads)
            if spectral_norm:
                gat.lin_src = nn.utils.spectral_norm(gat.lin_src)
                # gat.lin_dst = nn.utils.spectral_norm(gat.lin_dst)
            return gat
        else:
            raise NotImplementedError

    def __init__(self, gnn_type, num_layers, d_input, d_hidden, d_output, act=None, use_noise=False, **kwargs):
        super(NoisyGNN, self).__init__()
        self.num_layers = num_layers
        self.use_noise = use_noise
        self.act = getattr(F, act) if act is not None else None
        self.normalize = kwargs.pop('normalize', False)
        self.spectral_norm = kwargs.pop('spectral_norm', False)
        self.mp_layers = nn.ModuleList()
        for i in range(self.num_layers):
            in_channels = d_input if i == 0 else d_hidden
            gnn_layer = self._get_gnn(
                gnn_type=gnn_type,
                in_channels=in_channels,
                out_channels=d_hidden,
                spectral_norm=self.spectral_norm
            )
            if use_noise:
                sigma = kwargs.get('sigma', 1.)
                gnn_layer = NoisyEmbedding(gnn=gnn_layer, sigma=sigma, normalize=self.normalize)
            elif self.normalize:
                gnn_layer = NoisyEmbedding(gnn=gnn_layer, sigma=0., normalize=True)
            self.mp_layers.append(gnn_layer)
        self.out_proj = nn.Linear(in_features=d_hidden, out_features=d_output)
        self.to(self.device)

    def forward(self, gdata):
        x, edge_index = gdata.x.to(self.device), gdata.edge_index.to(self.device)
        if self.use_noise and self.normalize:
            x = F.normalize(x, dim=1)
        for i, layer in enumerate(self.mp_layers):
            x = layer(x, edge_index)
            if self.act is not None and i < self.num_layers - 1:
                x = self.act(x)
        hidden_emb = x
        outputs = F.log_softmax(self.out_proj(x), dim=1)
        return outputs, hidden_emb
    
    @property
    @torch.no_grad()
    def weight_spectrum(self):
        def _get_weight(layer):
            if hasattr(layer.gnn, 'lin'):
                return layer.gnn.lin.weight
            else:
                # Assume symmetric GAT
                return layer.gnn.lin_src.weight
        return [
            torch.linalg.svdvals(_get_weight(layer)).cpu().numpy().tolist()
            for layer in self.mp_layers
        ]

    @classmethod
    def from_cfg(cls, cfg):
        return cls(
            gnn_type=cfg.gnn.gnn_type,
            num_layers=cfg.gnn.num_layers,
            d_input=cfg.gnn.d_input,
            d_hidden=cfg.gnn.d_hidden,
            d_output=cfg.gnn.d_output,
            act=cfg.gnn.act,
            use_noise=cfg.privacy.method == 'noisy_embedding',
            sigma=cfg.privacy.sigma,
            normalize=cfg.privacy.normalize,
            spectral_norm=cfg.gnn.spectral_norm
        )
