import os
import sys
import random
import pickle
import logging
import argparse
from datetime import datetime
from typing import Dict, List

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl import shortest_dist
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dial.data import (
    ABCDDataset,
    PPMIDataset,
    load_data,
    preprocess_labels,
    balance_dataset,
    split_dataset
)
from baselines.models import (
    MLPBaseline,
    GCNBaseline,
    GATBaseline,
    GraphSAGEBaseline,
    GINBaseline,
    GraphormerBaseline
)

LOGGER_NAME = "dial_baselines"


def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def build_model(model_name: str, num_nodes: int, args: argparse.Namespace) -> nn.Module:
    model_name = model_name.lower().strip()

    if model_name == "mlp":
        return MLPBaseline(
            num_nodes=num_nodes,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_classes=2,
            dropout=args.dropout
        )
    if model_name == "gcn":
        return GCNBaseline(
            num_nodes=num_nodes,
            hidden_dim=args.hidden_dim,
            embed_dim=args.embed_dim,
            num_classes=2,
            dropout=args.dropout
        )
    if model_name == "gat":
        return GATBaseline(
            num_nodes=num_nodes,
            hidden_dim=args.hidden_dim,
            embed_dim=args.embed_dim,
            heads=args.gat_heads,
            num_classes=2,
            dropout=args.dropout
        )
    if model_name == "sage" or model_name == "graphsage":
        return GraphSAGEBaseline(
            num_nodes=num_nodes,
            hidden_dim=args.hidden_dim,
            embed_dim=args.embed_dim,
            num_classes=2,
            dropout=args.dropout
        )
    if model_name == "gin":
        return GINBaseline(
            num_nodes=num_nodes,
            hidden_dim=args.hidden_dim,
            embed_dim=args.embed_dim,
            num_classes=2,
            dropout=args.dropout
        )
    if model_name == "graphormer":
        return GraphormerBaseline(
            num_nodes=num_nodes,
            d_model=args.d_model,
            nhead=args.graph_nhead,
            num_layers=args.graph_layers,
            dim_feedforward=args.graph_ffn,
            num_classes=2,
            dropout=args.dropout,
            max_degree=args.max_degree,
            max_path_len=args.max_path_len,
        )
    raise ValueError(f"Unknown model: {model_name}")


def _graphormer_inputs_from_adj(
        adj_batch: torch.Tensor,
        max_degree: int,
        max_path_len: int
) -> Dict[str, torch.Tensor]:
    """
    Build Graphormer inputs from adjacency.
    """
    device = adj_batch.device
    compute_device = torch.device("cpu")
    batch_size, num_nodes, _ = adj_batch.shape
    attn_mask = torch.zeros(batch_size, num_nodes + 1, num_nodes + 1, dtype=torch.bool, device=device)
    node_feat_list = []
    in_degree_list = []
    out_degree_list = []
    path_data_list = []
    dist = -torch.ones((batch_size, num_nodes, num_nodes), dtype=torch.long, device=device)

    for idx in range(batch_size):
        adj_cpu = adj_batch[idx].to(compute_device)
        mask = adj_cpu != 0
        src, dst = mask.nonzero(as_tuple=True)

        if src.numel() == 0:
            src = dst = torch.arange(num_nodes, device=compute_device)

        g = dgl.graph((src, dst), num_nodes=num_nodes, device=compute_device)
        edge_feat = adj_cpu[src, dst].unsqueeze(-1)
        if edge_feat.numel() == 0:
            edge_feat = torch.ones((num_nodes, 1), device=compute_device, dtype=adj_cpu.dtype)

        g.edata['feat'] = edge_feat
        g.ndata['feat'] = adj_cpu

        spd, path = shortest_dist(g, root=None, return_paths=True)
        spd = spd.long().to(device)

        node_feat_list.append(g.ndata['feat'].to(device))
        in_degree_list.append(torch.clamp(g.in_degrees() + 1, min=0, max=max_degree).to(device))
        out_degree_list.append(torch.clamp(g.out_degrees() + 1, min=0, max=max_degree).to(device))
        dist[idx, :num_nodes, :num_nodes] = spd

        path_len = path.shape[2]
        if path_len >= max_path_len:
            shortest_path = path[:, :, :max_path_len]
        else:
            shortest_path = F.pad(path, (0, max_path_len - path_len), value=-1)

        edge_feat = edge_feat + 1
        edge_feat = torch.cat(
            [edge_feat, torch.zeros(1, edge_feat.shape[1], device=compute_device, dtype=edge_feat.dtype)],
            dim=0
        )
        path_data_list.append(edge_feat[shortest_path].to(device))

    node_feat = torch.stack(node_feat_list)
    in_degree = torch.stack(in_degree_list)
    out_degree = torch.stack(out_degree_list)
    path_data = torch.stack(path_data_list)

    return {
        'node_feat': node_feat,
        'in_degree': in_degree,
        'out_degree': out_degree,
        'path_data': path_data,
        'dist': dist,
        'attn_mask': attn_mask
    }


def forward_batch(model: nn.Module, batch: Dict, device: str, args: argparse.Namespace) -> torch.Tensor:
    if getattr(model, "expects_graphormer_inputs", False):
        fc_adj = batch['F'].to(device)
        sc_adj = batch['S'].to(device)

        fc_inputs = _graphormer_inputs_from_adj(fc_adj, max_degree=args.max_degree, max_path_len=args.max_path_len)
        sc_inputs = _graphormer_inputs_from_adj(sc_adj, max_degree=args.max_degree, max_path_len=args.max_path_len)

        logits = model(
            fc_inputs['node_feat'],
            fc_inputs['in_degree'],
            fc_inputs['out_degree'],
            fc_inputs['path_data'],
            fc_inputs['dist'],
            fc_inputs['attn_mask'],
            sc_inputs['node_feat'],
            sc_inputs['in_degree'],
            sc_inputs['out_degree'],
            sc_inputs['path_data'],
            sc_inputs['dist'],
            sc_inputs['attn_mask'],
        )
    else:
        sc = batch['S'].to(device)
        fc = batch['F'].to(device)
        # Ensure data is safe
        sc = torch.nan_to_num(sc, nan=0.0)
        fc = torch.nan_to_num(fc, nan=0.0)
        logits = model(fc, sc)
    return logits


def train_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        device: str,
        epoch: int,
        num_epochs: int,
        args: argparse.Namespace,
        show_progress: bool = True
) -> Dict:
    model.train()

    total_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []
    sample_count = 0

    batch_iterable = dataloader
    if show_progress:
        batch_iterable = tqdm(
            dataloader,
            desc=f"Train {epoch + 1}/{num_epochs}",
            leave=False
        )

    for batch in batch_iterable:
        labels = batch['labels'].to(device).squeeze(-1).long()
        batch_size = labels.shape[0]

        optimizer.zero_grad()
        logits = forward_batch(model, batch, device, args)
        loss = nn.functional.cross_entropy(logits, labels)

        # Check for NaN loss
        if torch.isnan(loss):
            print(f"[Warning] NaN loss detected at epoch {epoch}")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1).detach().cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.detach().cpu().tolist())
        sample_count += batch_size

    if show_progress and hasattr(batch_iterable, 'close'):
        batch_iterable.close()

    avg_loss = total_loss / max(sample_count, 1)
    accuracy = accuracy_score(all_labels, all_preds)

    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }


def evaluate(model: nn.Module, dataloader: DataLoader, device: str, args: argparse.Namespace) -> Dict:
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []
    all_names = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch['labels'].to(device).squeeze(-1).long()
            names = batch['names']

            logits = forward_batch(model, batch, device, args)
            probs = torch.softmax(logits, dim=1)

            all_preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
            all_probs.extend(probs[:, 1].detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())
            all_names.extend(names)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.0

    cm = confusion_matrix(all_labels, all_preds)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels,
        'names': all_names
    }

    return metrics


def print_metrics(metrics: Dict, logger: logging.Logger, prefix: str = ""):
    logger.info("%sAccuracy:  %.4f", prefix, metrics['accuracy'])
    logger.info("%sPrecision: %.4f", prefix, metrics['precision'])
    logger.info("%sRecall:    %.4f", prefix, metrics['recall'])
    logger.info("%sF1-Score:  %.4f", prefix, metrics['f1'])
    logger.info("%sAUC:       %.4f", prefix, metrics['auc'])
    if 'confusion_matrix' in metrics:
        logger.info("%sConfusion Matrix:\n%s", prefix, metrics['confusion_matrix'])


def run_single_model(
        model_name: str,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: str,
        num_nodes: int,
        args: argparse.Namespace,
        base_dir: str
) -> Dict:
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = os.path.join(base_dir, model_name, run_id)
    os.makedirs(model_dir, exist_ok=True)

    logger = setup_logger(os.path.join(model_dir, f"{model_name}.log"))
    logger.info("=" * 80)
    logger.info(f"Running baseline: {model_name}")
    logger.info("=" * 80)
    logger.info(f"Args: {args}")
    logger.info("Seeds -> train/init: %d | data split/balance: %d", args.random_seed, args.data_seed)

    model = build_model(model_name, num_nodes, args).to(device)
    logger.info(f"Model: {model}")
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Number of parameters: {num_params}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_auc = 0.0
    best_epoch = 0
    train_history = []
    test_history = []

    for epoch in range(args.num_epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch, args.num_epochs, args)
        test_metrics = evaluate(model, test_loader, device, args)

        train_history.append(train_metrics)
        test_history.append(test_metrics)

        # Use AUC or F1 for scheduler? Standard is often Loss or AUC/F1. Using F1 here as in previous code.
        scheduler.step(test_metrics['f1'])

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info("Epoch %d/%d", epoch + 1, args.num_epochs)
            logger.info("  Train - Loss: %.4f, Acc: %.4f", train_metrics['loss'], train_metrics['accuracy'])
            logger.info(
                "  Test  - Acc: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f, AUC: %.4f",
                test_metrics['accuracy'],
                test_metrics['precision'],
                test_metrics['recall'],
                test_metrics['f1'],
                test_metrics['auc']
            )

        if test_metrics['auc'] >= best_auc:
            best_auc = test_metrics['auc']
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pth"))
            logger.info("  *** Saved new best model (AUC=%.4f) ***", best_auc)

    logger.info(f"Best epoch: {best_epoch + 1}")
    logger.info("Final train evaluation:")
    train_final = evaluate(model, train_loader, device, args)
    print_metrics(train_final, logger, prefix="  ")

    logger.info("Final test evaluation:")
    test_final = evaluate(model, test_loader, device, args)
    print_metrics(test_final, logger, prefix="  ")

    results = {
        'model': model_name,
        'best_epoch': best_epoch,
        'train_final': train_final,
        'test_final': test_final,
        'train_history': train_history,
        'test_history': test_history,
    }

    with open(os.path.join(model_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    with open(os.path.join(model_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"{model_name} baseline summary - {args.task}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Best epoch: {best_epoch + 1}\n\n")
        f.write("Test results:\n")
        f.write(classification_report(
            test_final['labels'],
            test_final['predictions'],
            target_names=['Negative', 'Positive']
        ))

    logger.info("Completed baseline: %s", model_name)
    return results


def load_datasets(args: argparse.Namespace):
    if args.task == 'PPMI':
        train_dataset = PPMIDataset(args.ppmi_train_path, device=args.device)
        test_dataset = PPMIDataset(args.ppmi_test_path, device=args.device)
    else:
        data_dict = load_data(args.data_path)
        processed_dict = preprocess_labels(data_dict, task=args.task)
        balanced_dict = balance_dataset(processed_dict, ratio=args.balance_ratio, random_seed=args.data_seed)
        train_data, test_data = split_dataset(balanced_dict, test_size=args.test_size, random_seed=args.data_seed)
        train_dataset = ABCDDataset(train_data, device=args.device, max_path_len=args.max_path_len,
                                    max_degree=args.max_degree)
        test_dataset = ABCDDataset(test_data, device=args.device, max_path_len=args.max_path_len,
                                   max_degree=args.max_degree)
    return train_dataset, test_dataset


def run(args: argparse.Namespace):
    set_seed(args.random_seed)

    os.makedirs(args.output_dir, exist_ok=True)
    task_root = os.path.join(args.output_dir, "baselines", args.task)
    os.makedirs(task_root, exist_ok=True)

    train_dataset, test_dataset = load_datasets(args)
    if len(train_dataset) == 0:
        raise ValueError("Empty training dataset detected.")

    sample = train_dataset[0]
    num_nodes = sample['S'].shape[0]

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=test_dataset.collate
    )

    model_names = [m.strip().lower() for m in args.models.split(',') if m.strip()]
    all_results = {}
    for model_name in model_names:
        results = run_single_model(
            model_name=model_name,
            train_loader=train_loader,
            test_loader=test_loader,
            device=args.device,
            num_nodes=num_nodes,
            args=args,
            base_dir=task_root
        )
        all_results[model_name] = results

    return all_results


def build_arg_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Baseline experiments for FC/SC using MLP, GCN, GAT, GraphSAGE, GIN, and Graphormer.",
        add_help=add_help
    )

    # Data
    parser.add_argument('--data_path', type=str, default=r"W:\Brain Analysis\data\ABCD\processed\data_dict.pkl")
    parser.add_argument('--task', type=str, default='PPMI',
                        choices=['Dep', 'Bip', 'DMDD', 'Schi', 'Anx', 'OCD', 'Eat', 'ADHD', 'ODD',
                                 'Cond', 'PTSD', 'ADHD_ODD_Cond', 'PPMI'], help='Task name')
    parser.add_argument('--ppmi_train_path', type=str, default=r"../data/PPMI/train_data.pkl",
                        help='PPMI train pickle path')
    parser.add_argument('--ppmi_test_path', type=str, default=r"../data/PPMI/test_data.pkl",
                        help='PPMI test pickle path')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--test_size', type=float, default=0.3, help='Hold-out test fraction')
    parser.add_argument('--balance_ratio', type=float, default=1.0, help='Negative-to-positive balance ratio')
    parser.add_argument('--data_seed', type=int, default=20010920, help='Seed for data balancing/splitting')
    parser.add_argument('--random_seed', type=int, default=20010920, help='Training/init seed')

    # Model/baseline
    parser.add_argument('--models', type=str,
                        default='mlp,gcn,gat,gin,sage,graphormer',
                        help='Comma-separated baseline names to run')
    parser.add_argument('--embed_dim', type=int, default=128, help='Embedding dimension for outputs')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension for baselines')
    parser.add_argument('--gat_heads', type=int, default=4, help='Number of heads for GAT')
    parser.add_argument('--dropout', type=float, default=0.3)

    # Graphormer-specific
    parser.add_argument('--d_model', type=int, default=64, help='Graphormer hidden dimension')
    parser.add_argument('--graph_nhead', type=int, default=4, help='Graphormer attention heads')
    parser.add_argument('--graph_layers', type=int, default=2, help='Graphormer encoder layers')
    parser.add_argument('--graph_ffn', type=int, default=128, help='Graphormer feedforward dimension')
    parser.add_argument('--max_degree', type=int, default=511, help='Max degree bucket for Graphormer encoders')
    parser.add_argument('--max_path_len', type=int, default=5, help='Max path length for Graphormer encoders')

    # Training
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay factor')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

    # Device
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cpu/cuda)')
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    if not torch.cuda.is_available():
        args.device = 'cpu'

    run(args)