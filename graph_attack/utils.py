import os
import time
import json
import math
import concurrent.futures as cf
from itertools import product
from collections import OrderedDict
from dataclasses import dataclass

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import ListedColormap
from omegaconf import OmegaConf
from sklearn.metrics import roc_curve

from .metric import fast_numba_auc as roc_auc_score  # Faster than sklearn api

plt.rcParams['text.usetex'] = True


def _get_default_device():
    """cuda > mps > cpu"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


default_device = _get_default_device()


def setup_cfg_synthetic(**kwargs):
    """Setup an experiment config expressed in omegaconf format"""
    cfg = OmegaConf.create()
    cfg.gnn = {}
    cfg.gnn.aggr = kwargs.get('aggr', 'mean')
    cfg.gnn.self_loop = kwargs.get('self_loop', True)
    cfg.gnn.require_weights = kwargs.get('require_weights', False)
    cfg.gnn.num_layers = kwargs.get('num_layers', 1)
    cfg.gnn.spectral_norm = kwargs.get('spectral_norm', False)
    cfg.graph = {}
    cfg.graph.num_nodes = kwargs.get('num_nodes', 100)
    cfg.graph.num_features = kwargs.get('num_features', 4)
    # By default use Gaussian feature
    cfg.graph.feature_mean = kwargs.get('feature_mean', 0.)
    cfg.graph.feature_std = kwargs.get('feature_std', 1.)
    # Graph types
    cfg.graph.graph_type = kwargs.get('graph_type', 'erdos_renyi')  # {erdos_renyi, sbm, diagonal}
    cfg.graph.erdos_renyi = {
        'p': kwargs.get('erdos_renyi_p', math.log(cfg.graph.num_nodes) / cfg.graph.num_nodes)
    }
    cfg.graph.sbm = {
        'k': kwargs.get('sbm_k', 3),
        'p': kwargs.get('sbm_p', 0.3),
        'q': kwargs.get('sbm_q', 0.05)
    }
    cfg.graph.diagonal = {
        'f_base': kwargs.get('diagonal_f', 'threshold'),
        'f_threshold': kwargs.get('diagonal_threshold', .7),
        'f_p_above': kwargs.get('diagonal_p_above', 0.5),
        'f_p_below': kwargs.get(
            'diagonal_p_below', math.log(cfg.graph.num_nodes) / cfg.graph.num_nodes
        )
    }
    cfg.privacy = {}
    cfg.privacy.sigma = kwargs.get('sigma', 0.)
    cfg.privacy.alpha = kwargs.get('alpha', 1.)
    cfg.exp = {}
    cfg.exp.seed = kwargs.get('seed', 7)
    cfg.exp.metric = kwargs.get('metric', 'auc')  # {auc, fp@r[.0-9+]}
    cfg.exp.num_trials = kwargs.get('num_trials', 1)
    return cfg


_PLANETOID_META = {
    'Cora': {
        'num_features': 1433,
        'num_classes': 7
    },
    'Citeseer': {
        'num_features': 3703,
        'num_classes': 6
    },
    'Pubmed': {
        'num_features': 500,
        'num_classes': 3
    }
}


def setup_cfg_public(**kwargs):
    """Setup on public data experiments"""
    cfg = OmegaConf.create()
    cfg.dataset = {}
    cfg.dataset.name = kwargs.get('dataset', 'cora')
    default_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets')
    cfg.dataset.root = kwargs.get('root', default_root)
    cfg.gnn = {}
    cfg.gnn.gnn_type = kwargs.get('gnn_type', 'mp_gcn')
    cfg.gnn.num_layers = kwargs.get('num_layers', 2)
    cfg.gnn.act = kwargs.get('act', 'relu')
    cfg.gnn.d_input = _PLANETOID_META[cfg.dataset.name]['num_features']
    cfg.gnn.d_hidden = kwargs.get('d_hidden', 64)
    cfg.gnn.d_output = _PLANETOID_META[cfg.dataset.name]['num_classes']
    cfg.gnn.spectral_norm = kwargs.get('spectral_norm', False)
    cfg.train = {}
    cfg.train.epochs = kwargs.pop('epochs', 200)
    cfg.train.lr = kwargs.pop('lr', 1e-3)
    cfg.train.num_workers = kwargs.pop('num_workers', 0)
    cfg.privacy = {}
    cfg.privacy.method = kwargs.get('defense_method', None)  # {edge_rr, noisy_embedding, None}
    cfg.privacy.eps = kwargs.get('rr_eps', 4.)  # Better not be two small
    cfg.privacy.sigma = kwargs.get('sigma', 1.)
    cfg.privacy.normalize = kwargs.get('normalize', False)
    cfg.exp = {}
    cfg.exp.seed = kwargs.get('seed', 7)
    cfg.exp.metric = kwargs.get('metric', 'auc')  # {auc, fp@r[.0-9+]}
    cfg.exp.num_trials = kwargs.get('num_trials', 1)
    return cfg


def parse_metrics(cfg):
    if cfg.exp.metric == 'auc':
        return roc_auc_score
    elif cfg.exp.metric == 'err_sum':
        """Compute the sum of type I and type II errors"""

        def _metric(y_true, y_score):
            fprs, tprs, _ = roc_curve(y_true=y_true, y_score=y_score)
            return np.min(fprs + 1 - tprs)
        return _metric
    elif cfg.exp.metric.startswith('fp@r'):
        recall = float(cfg.exp.metric.strip()[4:]) / 100

        def _metric(y_true, y_score):  # Keep signature similar to roc_curve
            fprs, tprs, _ = roc_curve(y_true=y_true, y_score=y_score)
            cond = tprs >= recall
            if cond.any():
                return fprs[cond][0]
            else:
                return 0.
        return _metric
    else:
        raise NotImplementedError


class HyperParamManager(object):

    @staticmethod
    def _default_root(prefix='experiment'):
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        return os.path.join(log_dir, f'{prefix}_{int(time.time())}')

    @staticmethod
    def _gnn_tag_synthetic(cfg):
        return f'd={cfg.graph.num_features}_L={cfg.gnn.num_layers}'
    
    @staticmethod
    def _gnn_tag_real(cfg):
        return f'data={cfg.dataset.name}_d={cfg.gnn.d_hidden}_L={cfg.gnn.num_layers}'

    @staticmethod
    def _gnn_tag_sbm(cfg):
        return f'k={cfg.graph.sbm.k}_p={cfg.graph.sbm.p}_q={cfg.graph.sbm.q}'

    def __init__(self, root_dir=None, cfg_setter=setup_cfg_synthetic):
        if root_dir is None:
            root_dir = HyperParamManager._default_root()
        self._param_store = OrderedDict()
        self._root_dir = root_dir
        if not os.path.exists(root_dir):
            os.mkdir(root_dir)
        self.cfg_setter = cfg_setter

    def register_field(self, key, value):
        if key not in self._param_store:
            self._param_store[key] = []
        if isinstance(value, list):
            self._param_store[key].extend(value)
        else:
            self._param_store[key].append(value)

    def __getitem__(self, item):
        return self._param_store[item]

    def iter_configs(self):
        for cfg_values in product(*self._param_store.values()):
            yield self.cfg_setter(
                **dict(zip(self._param_store.keys(), cfg_values))
            )

    def get_description_string(self, fields):
        descriptions = []
        for field in fields:
            field_val = '-'.join(map(str, self[field]))
            descriptions.append(f'{field}_{field_val}')
        return '+'.join(descriptions)

    def get_tag(self, mode):
        if mode == 'synthetic':
            return self._gnn_tag_synthetic
        elif mode == 'real':
            return self._gnn_tag_real
        else:
            return self._gnn_tag_sbm

    def record_result(self, cfg, result, prefix='task', mode='synthetic'):
        gnn_tag = self.get_tag(mode)
        file_path = os.path.join(
            self._root_dir,
            f'{prefix}_{gnn_tag(cfg)}_{int(time.time())}.json'
        )
        packed_result = {
            'config': OmegaConf.to_object(cfg),
            'result': result
        }
        with open(file_path, mode='w') as fw:
            json.dump(packed_result, fw, indent=4)

    def clone(self):
        new_manager = HyperParamManager(cfg_setter=self.cfg_setter)
        for key, val in self._param_store.items():
            new_manager.register_field(key, val)
        return new_manager


class Analyzer(object):
    """For holding plotting results."""

    @dataclass
    class _Result(object):
        accs: list
        adv_metrics: list
        spectrum: list = None

        def spectrum_summary(self):
            spectrum = list(zip(*self.spectrum))  # swap order to make numpy operation possible

            def _layer_summary(sp):
                op_norms = sp[:, 0]
                cond_numbers = sp[:, 0] / sp[:, -1]
                snrs = np.sqrt(np.square(sp).mean(axis=-1))
                return op_norms, cond_numbers, snrs

            return [_layer_summary(np.asarray(sp)) for sp in spectrum]

    def __init__(self, root_dir, **kwargs):
        self._root_dir = root_dir
        self._meta_data = {}
        results = []

        def _fill(log_path):
            with open(os.path.join(root_dir, log_path)) as fr:
                result_dict = json.load(fr)
            self._update_meta(result_dict['config'])
            result = self._Result(**result_dict['result'])
            results.append((result_dict['config']['gnn']['d_hidden'], result))

        max_workers = kwargs.get('max_workers', 10)
        with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_fill, file_path) for file_path in os.listdir(root_dir)]
            [future.result() for future in cf.as_completed(futures)]

        results.sort(key=lambda x: x[0])
        self.results = [r[1] for r in results]
        self.labels = np.log2(np.asarray([r[0] for r in results])).astype(np.int_)

    def _update_meta(self, config):

        def _maybe_update(key, conf_key):
            if conf_key not in config:
                return
            elif key not in config[conf_key]:
                return
            if key not in self._meta_data:
                self._meta_data[key] = config[conf_key][key]

        _maybe_update('name', 'dataset')
        _maybe_update('gnn_type', 'gnn')
        _maybe_update('num_layers', 'gnn')
        _maybe_update('sigma', 'privacy')
        _maybe_update('epochs', 'train')

    def utility_summary(self):
        accs = [r.accs for r in self.results]
        return np.mean(accs, axis=1), np.std(accs, axis=1)

    def attack_performance_summary(self):
        adv_metrics = [r.adv_metrics for r in self.results]
        return np.mean(adv_metrics, axis=1), np.std(adv_metrics, axis=1)

    def spectrum_summary(self):
        spectrums = np.asarray([r.spectrum_summary() for r in self.results])
        return spectrums.mean(axis=-1), spectrums.std(axis=-1)


class SBMAnalyzer(object):

    @dataclass
    class _Result(object):
        mean: float
        std: float

    def __init__(self, root_dir, **kwargs):
        self._root_dir = root_dir
        self._meta_data = {}
        results = []

        def _fill(log_path):
            with open(os.path.join(root_dir, log_path)) as fr:
                result_dict = json.load(fr)
            self._update_meta(result_dict['config'])
            result = self._Result(**result_dict['result'])
            results.append(
                (result_dict['config']['graph']['sbm']['k'],
                 result_dict['config']['graph']['sbm']['p'],
                 result)
            )

        max_workers = kwargs.get('max_workers', 10)
        with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_fill, file_path) for file_path in os.listdir(root_dir)]
            [future.result() for future in cf.as_completed(futures)]

        ks = sorted(set(r[0] for r in results))
        k_map = {k: i for i, k in enumerate(ks)}
        ps = sorted(set(r[1] for r in results))
        p_map = {p: i for i, p in enumerate(ps)}
        self.results_mean = np.zeros([len(ks), len(ps)])
        self.results_std = np.zeros([len(ks), len(ps)])
        for k, p, r in results:
            self.results_mean[k_map[k], p_map[p]] = r.mean
            self.results_std[k_map[k], p_map[p]] = r.std
        self.k_labels = ks
        self.p_labels = ps

    def _update_meta(self, config):

        def _maybe_update(key, conf_key):
            if conf_key not in config:
                return
            elif key not in config[conf_key]:
                return
            if key not in self._meta_data:
                self._meta_data[key] = config[conf_key][key]

        _maybe_update('num_layers', 'gnn')
        _maybe_update('metric', 'exp')


class MatrixAnalyzer(object):
    """Helper class for visualizing results, note that we allow instantiations of this class
    even when the original manager no longer resides in memory"""

    _ERR_CM = 'Reds'
    _AUC_CM = 'Blues'
    _FP_CM = 'Browns'

    @staticmethod
    def _default_mapper(row, col):
        return (
            int(row) - 1,
            int(math.log2(float(col))) - 2,
        )

    def __init__(self, root_dir, shape=(10, 10), **kwargs):
        self._root_dir = root_dir
        self._result = np.zeros(shape)
        self._row_labels = set()
        self._col_labels = set()
        self._meta_data = {}
        mapper = kwargs.get('mapper', self._default_mapper)

        def _fill(log_path):
            with open(os.path.join(root_dir, log_path)) as fr:
                result_dict = json.load(fr)
            _, col_, row_, _ = log_path.split('_')
            row_, col_ = int(row_.split('=')[1]), int(col_.split('=')[1])
            row, col = mapper(row_, col_)
            self._row_labels.add(row_)
            self._col_labels.add(int(math.log2(col_)))  # log scale is better for display
            self._result[row, col] = result_dict['result']['mean']
            self._update_meta(result_dict['config'])

        max_workers = kwargs.get('max_workers', 10)
        with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_fill, file_path) for file_path in os.listdir(root_dir)]
            [future.result() for future in cf.as_completed(futures)]

        self.row_labels = sorted(self._row_labels)
        self.col_labels = sorted(self._col_labels)

    @property
    def result(self):
        return self._result

    @property
    def meta_data(self):
        return self._meta_data

    @property
    def root_dir(self):
        return self._root_dir

    def _update_meta(self, config):

        def _maybe_update(key, conf_key):
            if conf_key not in config:
                return
            elif key not in config[conf_key]:
                return
            if key not in self._meta_data:
                self._meta_data[key] = config[conf_key][key]

        _maybe_update('graph_type', 'graph')
        _maybe_update('num_nodes', 'graph')
        _maybe_update('metric', 'exp')

    def get_cmap(self, **kwargs):
        # Get color maps
        if self._meta_data['metric'] == 'auc':
            cm = self._AUC_CM
            tick_range = np.linspace(0.5, 1, 6)
        elif self._meta_data['metric'] == 'err_sum':
            cm = self._ERR_CM
            tick_range = np.linspace(0., 1., 11)
        elif self._meta_data['metric'].startswith('fp'):
            cm = self._FP_CM
            tick_range = np.linspace(0., 1., 11)
        else:
            raise NotImplementedError('Unknown metric type')
        num_colors = kwargs.get('num_colors', 100)
        return ListedColormap(colormaps[cm](np.linspace(0, 1, num_colors))), tick_range

    @property
    def plot_title(self):
        metric = self.meta_data['metric']
        graph_type = self.meta_data['graph_type']
        num_nodes = self.meta_data['num_nodes']
        return f'{metric} plot with graph type {graph_type}, num_nodes {num_nodes}'

    @classmethod
    def plot_v1(cls, analyzer, save_fig=True, **kwargs):
        fig_size = kwargs.get('fig_size', [10, 10])
        cmap, tick_range = analyzer.get_cmap(**kwargs)
        fig, ax = plt.subplots(figsize=fig_size)
        im = ax.imshow(analyzer.result, cmap=cmap)
        ax.set_title(analyzer.title)
        ax.set_xlabel('num_features d')
        ax.set_ylabel('num_layers L')
        ax.set_yticks(range(len(analyzer.row_labels)), labels=analyzer.row_labels)
        ax.set_xticks(range(len(analyzer.col_labels)), labels=analyzer.col_labels)
        fig.colorbar(im, ax=ax, ticks=tick_range)
        if save_fig:
            title = kwargs.get('title', analyzer.title)
            plt.savefig(os.path.join(analyzer.root_dir, f'{title}.png'))
        plt.show()

    @classmethod
    def plot_multiple_v1(cls, analyzers, save_fig=True, **kwargs):
        analyzers.sort(key=lambda p: p.meta_data['num_nodes'])
        fig_size = kwargs.get('fig_size', [10, 10])
        cmap, tick_range = analyzers[0].get_cmap(**kwargs)  # All plotters shall use the same metric
        fig, axes = plt.subplots(1, len(analyzers), figsize=fig_size)
        for analyzer, ax in zip(analyzers, axes):
            im = ax.imshow(analyzer.result, cmap=cmap, vmin=tick_range[0], vmax=tick_range[-1])
            num_nodes = analyzer.meta_data['num_nodes']
            ax.set_title(fr'$n={num_nodes}$')
            ax.set_xlabel(r'$\log_2 d$')
            ax.set_ylabel(r'$L$')
            ax.set_yticks(range(len(analyzer.row_labels)), labels=analyzer.row_labels)
            ax.set_xticks(range(len(analyzer.col_labels)), labels=analyzer.col_labels)
        fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=1.0,
                            wspace=0.2, hspace=0.02)
        cb_ax = fig.add_axes([1.03, 0.35, 0.01, 0.3])
        cbar = fig.colorbar(im, cax=cb_ax, ticks=tick_range)
        if save_fig:
            metric = analyzer.meta_data['metric']
            graph_type = analyzer.meta_data['graph_type']
            title = fr'{metric} plot with graph type {graph_type}'
            plt.savefig(f'{title}.pdf', bbox_inches="tight")
        plt.show()

