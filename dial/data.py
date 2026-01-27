"""
Data loading and preprocessing utilities for the DIAL project.

Reference: https://github.com/dmlc/dgl/blob/master/examples/core/Graphormer/dataset.py

Provided helpers:
- ABCDDataset: dataset wrapper that precomputes DGL graph features
- PPMIDataset: dataset wrapper for pre-split PPMI pickle files
- load_data: load pickled raw data
- preprocess_labels: task-specific label processing
- balance_dataset: down-sample majority class to balance ratio
- split_dataset: stratified train/test split
"""

import pickle
import numpy as np
import torch
import torch.nn.functional as F
import dgl
from dgl import shortest_dist
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple


def normalize_sc(sc_array: np.ndarray) -> torch.Tensor:
    """Scale structural connectivity values to 0-1 with log compression."""
    sc = np.asarray(sc_array, dtype=np.float32)
    sc = np.maximum(sc, 0.0)
    sc_log = np.log1p(sc)
    min_val = sc_log.min()
    max_val = sc_log.max()
    if max_val > min_val:
        sc_norm = (sc_log - min_val) / (max_val - min_val)
    else:
        sc_norm = sc_log - min_val
    return torch.from_numpy(sc_norm)


def filter_fc(fc_array: np.ndarray) -> torch.Tensor:
    """Keep only the top 30% the strongest functional connections by absolute value."""
    fc = np.asarray(fc_array, dtype=np.float32)
    abs_vals = np.abs(fc)
    threshold = np.quantile(abs_vals, 0.7)
    mask = abs_vals >= threshold
    filtered = np.where(mask, fc, 0.0)
    return torch.from_numpy(filtered)


def build_graph(S: torch.Tensor, node_feat: torch.Tensor):
    num_nodes = S.shape[0]
    mask = S > 0
    src, dst = mask.nonzero(as_tuple=True)
    graph = dgl.graph((src, dst), num_nodes=num_nodes)
    edge_weights = S[src, dst].unsqueeze(-1)
    graph.edata['feat'] = edge_weights
    graph.ndata['feat'] = node_feat
    return graph


class BrainDatasetBase(Dataset):
    """Base dataset with shared graph preprocessing and collation."""

    def __init__(
            self,
            samples: List[Dict],
            device: str = 'cpu',
            max_degree: int = 511,
            max_path_len: int = 5
    ):
        self.device = device
        self.max_degree = max_degree
        self.max_path_len = max(1, max_path_len)
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        return {
            'S': sample['S'].to(self.device),
            'F': sample['F'].to(self.device),
            'label': sample['label'].to(self.device),
            'name': sample['name'],
            'graph': sample['graph']
        }

    def collate(self, samples: List[Dict]):
        """Graphormer-style batching with padding."""
        graphs = [s['graph'] for s in samples]
        labels = torch.stack([s['label'] for s in samples]).reshape(len(samples), -1)
        num_graphs = len(graphs)
        num_nodes = [g.num_nodes() for g in graphs]
        max_num_nodes = max(num_nodes)

        # Graphormer adds a virtual node to the graph, which is connected to all other nodes
        # and supposed to represent the graph embedding. So here +1 is for the virtual node.
        attn_mask = torch.zeros(num_graphs, max_num_nodes + 1, max_num_nodes + 1, dtype=torch.bool)
        node_feat = []
        in_degree, out_degree = [], []
        path_data = []
        # Since shortest_dist returns -1 for unreachable node pairs and padded
        # nodes are unreachable to others, distance relevant to padded nodes use -1 padding as well.
        dist = -torch.ones((num_graphs, max_num_nodes, max_num_nodes), dtype=torch.long)

        for i, g in enumerate(graphs):
            attn_mask[i, :, num_nodes[i] + 1:] = True
            node_feat.append(g.ndata['feat'])
            in_degree.append(torch.clamp(g.in_degrees() + 1, min=0, max=self.max_degree))
            out_degree.append(torch.clamp(g.out_degrees() + 1, min=0, max=self.max_degree))

            path = g.ndata['path']
            path_len = path.shape[2]
            if path_len >= self.max_path_len:
                shortest_path = path[:, :, :self.max_path_len]
            else:
                shortest_path = F.pad(path, (0, self.max_path_len - path_len), 'constant', -1)

            pad_nodes = max_num_nodes - num_nodes[i]
            if pad_nodes > 0:
                shortest_path = F.pad(shortest_path, (0, 0, 0, pad_nodes, 0, pad_nodes), 'constant', -1)

            edge_feat = g.edata['feat'] if 'feat' in g.edata else torch.zeros((g.num_edges(), 1), dtype=torch.float32)
            if edge_feat.dim() == 1:
                edge_feat = edge_feat.unsqueeze(-1)
            edge_feat = edge_feat + 1
            edge_feat = torch.cat([edge_feat, torch.zeros(1, edge_feat.shape[1])], dim=0)
            path_data.append(edge_feat[shortest_path])

            dist[i, :num_nodes[i], :num_nodes[i]] = g.ndata['spd']

        node_feat = pad_sequence(node_feat, batch_first=True)
        in_degree = pad_sequence(in_degree, batch_first=True)
        out_degree = pad_sequence(out_degree, batch_first=True)
        path_data = torch.stack(path_data)
        S_batch = torch.stack([s['S'] for s in samples])
        F_batch = torch.stack([s['F'] for s in samples])
        names = [s['name'] for s in samples]

        return {
            'labels': labels,
            'attn_mask': attn_mask,
            'node_feat': node_feat,
            'in_degree': in_degree,
            'out_degree': out_degree,
            'path_data': path_data,
            'dist': dist,
            'S': S_batch,
            'F': F_batch,
            'names': names
        }


class ABCDDataset(BrainDatasetBase):
    """Brain-connectivity dataset with DGL-based preprocessing."""

    def __init__(
            self,
            data_list: List[Dict],
            device: str = 'cpu',
            max_degree: int = 511,
            max_path_len: int = 5
    ):
        samples = [
            self._prepare_graph_sample(item, max_path_len=max_path_len)
            for item in tqdm(data_list, desc='Preparing ABCD samples')
        ]
        super().__init__(
            samples=samples,
            device=device,
            max_degree=max_degree,
            max_path_len=max_path_len
        )

    @staticmethod
    def _prepare_graph_sample(item: Dict, max_path_len: int = 5) -> Dict:
        S = normalize_sc(item['SC'])
        F_mat = filter_fc(item['FC'])
        label = torch.tensor(item['label'], dtype=torch.long)
        name = item['name']
        graph = build_graph(S, F_mat)
        spd, path = shortest_dist(graph, root=None, return_paths=True)
        graph.ndata['spd'] = spd
        graph.ndata['path'] = path
        return {'S': S, 'F': F_mat, 'label': label, 'name': name, 'graph': graph}


class PPMIDataset(BrainDatasetBase):
    """Dataset wrapper for PPMI train/test pickle splits.

    Expects a pickle file shaped like
    {'scn': scn_train, 'fcn': fcn_train, 'labels': y_train}
    where the first dimension aligns across entries.
    """

    def __init__(
            self,
            data_path: str,
            device: str = 'cpu',
            max_degree: int = 511,
            max_path_len: int = 5,
            name_prefix: str = 'ppmi'
    ):
        data_list = self._load_ppmi_split(data_path, name_prefix=name_prefix)
        samples = [
            self._prepare_graph_sample(item, max_path_len=max_path_len)
            for item in tqdm(data_list, desc='Preparing PPMI samples')
        ]
        super().__init__(
            samples=samples,
            device=device,
            max_degree=max_degree,
            max_path_len=max_path_len
        )

    @staticmethod
    def _prepare_graph_sample(item: Dict, max_path_len: int = 5) -> Dict:
        # S = normalize_sc(item['SC'])
        # F_mat = filter_fc(item['FC'])
        S = torch.from_numpy(item['SC']).float()
        F_mat = torch.from_numpy(item['FC']).float()
        label = torch.tensor(item['label'], dtype=torch.long)
        name = item['name']
        graph = build_graph(S, F_mat)
        spd, path = shortest_dist(graph, root=None, return_paths=True)
        graph.ndata['spd'] = spd
        graph.ndata['path'] = path
        return {'S': S, 'F': F_mat, 'label': label, 'name': name, 'graph': graph}

    @staticmethod
    def _load_ppmi_split(data_path: str, name_prefix: str = 'ppmi') -> List[Dict]:
        print(f"[PPMI] loading from {data_path} ...")
        with open(data_path, 'rb') as f:
            raw = pickle.load(f)

        required_keys = ('scn', 'fcn', 'labels')
        missing = [k for k in required_keys if k not in raw]
        if missing:
            raise KeyError(f"PPMI pickle missing keys: {missing}")

        scn = np.asarray(raw['scn'])
        fcn = np.asarray(raw['fcn'])
        labels = np.asarray(raw['labels'])

        if scn.shape[0] != fcn.shape[0] or scn.shape[0] != labels.shape[0]:
            raise ValueError(
                f"PPMI data mismatch - scn: {scn.shape}, fcn: {fcn.shape}, labels: {labels.shape}"
            )

        samples: List[Dict] = []
        for idx, (sc, fc, label) in enumerate(zip(scn, fcn, labels)):
            samples.append({
                'SC': sc,
                'FC': fc,
                'label': int(label),
                'name': f"{name_prefix}_{idx}"
            })

        print(f"[PPMI] loaded {len(samples)} samples")
        return samples


def load_data(data_path: str) -> Dict:
    """Load the raw pickle file."""
    print(f"[data] loading from {data_path} ...")
    with open(data_path, 'rb') as f:
        data_dict = pickle.load(f)
    print(f"[data] loaded {len(data_dict)} records")
    return data_dict


def preprocess_labels(data_dict: Dict, task: str = 'OCD') -> Dict:
    """Task-specific label preprocessing."""
    label_names = ['Dep', 'Bip', 'DMDD', 'Schi', 'Anx', 'OCD', 'Eat', 'ADHD', 'ODD', 'Cond', 'PTSD', 'ADHD_ODD_Cond']
    label_idx = {name: i for i, name in enumerate(label_names)}
    processed_dict: Dict[str, Dict] = {}

    if task == 'ADHD_ODD_Cond':
        adhd_idx = label_idx['ADHD']
        odd_idx = label_idx['ODD']
        cond_idx = label_idx['Cond']
        print(f"[labels] ADHD/ODD/Cond task uses indices {adhd_idx}, {odd_idx}, {cond_idx}")
        for key, value in data_dict.items():
            labels = value['label']
            merged = int(labels[adhd_idx] == 1 or labels[odd_idx] == 1 or labels[cond_idx] == 1)
            processed_dict[key] = {
                'SC': value['SC'],
                'FC': value['FC'],
                'label': merged,
                'name': value['name'],
                'original_labels': value['label']
            }
    elif task in label_idx:
        target_idx = label_idx[task]
        print(f"[labels] {task} task uses index {target_idx}")
        for key, value in data_dict.items():
            label = value['label'][target_idx]
            processed_dict[key] = {
                'SC': value['SC'],
                'FC': value['FC'],
                'label': label,
                'name': value['name'],
                'original_labels': value['label']
            }
    else:
        raise ValueError(f'Unsupported task type: {task}')

    return processed_dict


def balance_dataset(data_dict: Dict, ratio: float = 1.0, random_seed: int = 42) -> Dict:
    """Down-sample the majority class to reach the desired positive/negative ratio."""
    positive_samples = {k: v for k, v in data_dict.items() if v['label'] == 1}
    negative_samples = {k: v for k, v in data_dict.items() if v['label'] == 0}

    n_pos = len(positive_samples)
    n_neg = len(negative_samples)
    print(f"[balance] original distribution - pos: {n_pos}, neg: {n_neg}")

    if n_pos > n_neg:
        target_pos = int(n_neg * ratio)
        rng = np.random.default_rng(random_seed)
        selected = rng.choice(list(positive_samples.keys()), target_pos, replace=False)
        balanced = {k: positive_samples[k] for k in selected}
        balanced.update(negative_samples)
    else:
        target_neg = int(max(n_pos / ratio, 1))
        rng = np.random.default_rng(random_seed)
        selected = rng.choice(list(negative_samples.keys()), target_neg, replace=False)
        balanced = {k: negative_samples[k] for k in selected}
        balanced.update(positive_samples)

    n_pos_new = sum(1 for v in balanced.values() if v['label'] == 1)
    n_neg_new = sum(1 for v in balanced.values() if v['label'] == 0)
    print(f"[balance] balanced distribution - pos: {n_pos_new}, neg: {n_neg_new}")
    return balanced


def split_dataset(data_dict: Dict, test_size: float = 0.3, random_seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """Stratified train/test split."""
    data_list = list(data_dict.values())
    labels = [item['label'] for item in data_list]

    train_data, test_data = train_test_split(
        data_list,
        test_size=test_size,
        random_state=random_seed,
        stratify=labels
    )

    train_pos = sum(1 for item in train_data if item['label'] == 1)
    train_neg = len(train_data) - train_pos
    test_pos = sum(1 for item in test_data if item['label'] == 1)
    test_neg = len(test_data) - test_pos

    print(f"[split] train: {len(train_data)} (pos {train_pos}, neg {train_neg})")
    print(f"[split] test:  {len(test_data)} (pos {test_pos}, neg {test_neg})")

    return train_data, test_data
