import os
import random
import pickle
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from typing import Dict, List
from datetime import datetime
from tqdm import tqdm

from dial.model import DIALModel
from dial.data import (
    ABCDDataset,
    PPMIDataset,
    load_data,
    preprocess_labels,
    balance_dataset,
    split_dataset
)

LOGGER_NAME = "dial_experiment"


def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def setup_logger(log_path: str) -> logging.Logger:
    """Configure file and console logging for the current run."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Remove old handlers when reconfiguring (important for repeated runs in same process)
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


def train_epoch(model: nn.Module,
                dataloader: DataLoader,
                optimizer: optim.Optimizer,
                device: str = 'cpu',
                epoch: int = 0,
                num_epochs: int = 1,
                show_progress: bool = True) -> Dict:
    model.train()

    total_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_probs: List[float] = []
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
        node_feat = batch['node_feat'].to(device)
        in_degree = batch['in_degree'].to(device)
        out_degree = batch['out_degree'].to(device)
        path_data = batch['path_data'].to(device)
        dist = batch['dist'].to(device)
        attn_mask = batch['attn_mask'].to(device)
        S = batch['S'].to(device)
        F = batch['F'].to(device)

        optimizer.zero_grad()
        y_pred, loss = model(
            node_feat=node_feat,
            in_degree=in_degree,
            out_degree=out_degree,
            path_data=path_data,
            dist=dist,
            attn_mask=attn_mask,
            S=S,
            F=F,
            y=labels,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * batch_size
        probs = torch.softmax(y_pred, dim=1)
        preds = probs.argmax(dim=1).detach().cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.detach().cpu().tolist())
        all_probs.extend(probs[:, 1].detach().cpu().tolist())
        sample_count += batch_size

    if show_progress and hasattr(batch_iterable, 'close'):
        batch_iterable.close()

    avg_loss = total_loss / max(sample_count, 1)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }


def evaluate(model: nn.Module,
             dataloader: DataLoader,
             device: str = 'cpu') -> Dict:
    """
    Evaluate the model on the provided dataloader and compute metrics.

    Args:
        model: Model to evaluate.
        dataloader: DataLoader that yields mini-batches.
        device: Device identifier.

    Returns:
        Dictionary with accuracy, precision, recall, F1, AUC, raw predictions, and labels.
    """
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []
    all_names = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch['labels'].to(device).squeeze(-1).long()
            names = batch['names']
            y_pred, _ = model.inference(
                node_feat=batch['node_feat'].to(device),
                in_degree=batch['in_degree'].to(device),
                out_degree=batch['out_degree'].to(device),
                path_data=batch['path_data'].to(device),
                dist=batch['dist'].to(device),
                attn_mask=batch['attn_mask'].to(device),
                S=batch['S'].to(device),
                F=batch['F'].to(device)
            )
            probs = torch.softmax(y_pred, dim=1)

            all_preds.extend(y_pred.argmax(dim=1).detach().cpu().tolist())
            all_probs.extend(probs[:, 1].detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())
            all_names.extend(names)

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
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


def print_metrics(metrics: Dict, prefix: str = ""):
    """Log formatted metrics with an optional text prefix."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.info("%sAccuracy:  %.4f", prefix, metrics['accuracy'])
    logger.info("%sPrecision: %.4f", prefix, metrics['precision'])
    logger.info("%sRecall:    %.4f", prefix, metrics['recall'])
    logger.info("%sF1-Score:  %.4f", prefix, metrics['f1'])
    logger.info("%sAUC:       %.4f", prefix, metrics['auc'])
    if 'confusion_matrix' in metrics:
        logger.info("%sConfusion Matrix:\n%s", prefix, metrics['confusion_matrix'])


def main(args: argparse.Namespace):
    train_seed = args.random_seed
    data_seed = getattr(args, "data_seed", train_seed)
    set_seed(train_seed)

    os.makedirs(args.output_dir, exist_ok=True)
    task_root = os.path.join(args.output_dir, args.task)
    os.makedirs(task_root, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    task_dir = os.path.join(task_root, run_id)
    os.makedirs(task_dir, exist_ok=True)

    logger = setup_logger(os.path.join(task_dir, 'experiment.log'))
    logger.info("=" * 80)
    logger.info(f"DIAL experiment - task: {args.task} (run {run_id})")
    logger.info("=" * 80)

    logger.info(f"Args: {args}")
    logger.info("Seeds -> train/init: %d | data split/balance: %d", train_seed, data_seed)

    if args.task == 'PPMI':
        logger.info("[Step 1] Load PPMI pre-split data")
        full_train_dataset = PPMIDataset(args.ppmi_train_path, device=args.device)
        test_dataset = PPMIDataset(args.ppmi_test_path, device=args.device)

        train_indices = list(range(len(full_train_dataset)))

        train_labels = [full_train_dataset[i]['label'].cpu().item() for i in train_indices]

        train_idx, val_idx = train_test_split(
            train_indices,
            test_size=args.val_size,
            random_state=data_seed,
            stratify=train_labels
        )
        train_dataset = Subset(full_train_dataset, train_idx)
        val_dataset = Subset(full_train_dataset, val_idx)
    else:
        logger.info("[Step 1] Load data")
        data_dict = load_data(args.data_path)

        logger.info(f"[Step 2] Label preprocessing - {args.task}")
        processed_dict = preprocess_labels(data_dict, task=args.task)

        logger.info(f"[Step 3] Balance dataset (ratio {args.balance_ratio}:1)")
        balanced_dict = balance_dataset(
            processed_dict,
            ratio=args.balance_ratio,
            random_seed=data_seed
        )

        logger.info(f"[Step 4] Split dataset (test size {args.test_size})")
        train_data, test_data = split_dataset(
            balanced_dict,
            test_size=args.test_size,
            random_seed=data_seed
        )

        logger.info(f"[Step 5] Split train into train/val (val size {args.val_size})")

        train_labels = [sample['label'] for sample in train_data]

        train_data, val_data = train_test_split(
            train_data,
            test_size=args.val_size,
            random_state=data_seed,
            stratify=train_labels
        )

        logger.info("[Step 6] Build dataset objects")
        train_dataset = ABCDDataset(train_data, device=args.device)
        val_dataset = ABCDDataset(val_data, device=args.device)
        test_dataset = ABCDDataset(test_data, device=args.device)

    logger.info("Dataset sizes -> Train: %d | Val: %d | Test: %d",
                len(train_dataset), len(val_dataset), len(test_dataset))

    train_collate = train_dataset.dataset.collate if isinstance(train_dataset, Subset) else train_dataset.collate
    val_collate = val_dataset.dataset.collate if isinstance(val_dataset, Subset) else val_dataset.collate
    test_collate = test_dataset.dataset.collate if isinstance(test_dataset, Subset) else test_dataset.collate

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_collate
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=val_collate
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=test_collate
    )

    if len(train_dataset) == 0:
        raise ValueError("Empty training dataset detected.")
    sample = train_dataset[0]
    N = sample['S'].shape[0]
    logger.info(f"Number of nodes: {N}")

    logger.info("[Step 6] Build DIAL model")
    model = DIALModel(
        N=N,
        d_model=args.d_model,
        num_classes=2,
        task='classification',
        num_node_layers=args.num_node_layers,
        num_graph_layers=args.num_graph_layers,
        dropout=args.dropout,
    ).to(args.device)
    logger.info(f"Model Architecture: {model}")

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Number of parameters: {num_params}")

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=1e-6
    )

    logger.info(f"[Step 7] Train for {args.num_epochs} epochs")
    logger.info("-" * 80)

    best_auc = 0.0
    best_epoch = 0
    train_history = []
    val_history = []
    test_history = []

    for epoch in range(args.num_epochs):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device=args.device,
            epoch=epoch,
            num_epochs=args.num_epochs
        )

        val_metrics = evaluate(model, val_loader, args.device)
        test_metrics = evaluate(model, test_loader, args.device)

        train_history.append(train_metrics)
        val_history.append(val_metrics)
        test_history.append(test_metrics)

        scheduler.step()

        logger.info("Epoch %d/%d", epoch + 1, args.num_epochs)
        logger.info(
            "  Train - Loss: %.4f, Acc: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f, AUC: %.4f",
            train_metrics['loss'],
            train_metrics['accuracy'],
            train_metrics['precision'],
            train_metrics['recall'],
            train_metrics['f1'],
            train_metrics['auc']
        )
        logger.info(
            "  Val   - Acc: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f, AUC: %.4f",
            val_metrics['accuracy'],
            val_metrics['precision'],
            val_metrics['recall'],
            val_metrics['f1'],
            val_metrics['auc']
        )
        logger.info(
            "  Test  - Acc: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f, AUC: %.4f",
            test_metrics['accuracy'],
            test_metrics['precision'],
            test_metrics['recall'],
            test_metrics['f1'],
            test_metrics['auc']
        )

        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_epoch = epoch
            model_path = os.path.join(task_dir, 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            logger.info("  *** Saved new best model (val AUC=%.4f) ***", best_auc)

    logger.info(f"Final evaluation (best epoch: {best_epoch + 1})")
    logger.info("-" * 80)

    model.load_state_dict(torch.load(os.path.join(task_dir, 'best_model.pth')))

    logger.info("Train results:")
    train_final = evaluate(model, train_loader, args.device)
    print_metrics(train_final, prefix="  ")

    logger.info("Validation results:")
    val_final = evaluate(model, val_loader, args.device)
    print_metrics(val_final, prefix="  ")

    logger.info("Test results:")
    test_final = evaluate(model, test_loader, args.device)
    print_metrics(test_final, prefix="  ")

    logger.info(f"[Step 9] Save artifacts to {task_dir}")

    results = {
        'task': args.task,
        'best_epoch': best_epoch,
        'train_final': train_final,
        'val_final': val_final,
        'test_final': test_final,
        'train_history': train_history,
        'val_history': val_history,
        'test_history': test_history,
        'config': {
            'N': N,
            'd_model': args.d_model,
            'num_epochs': args.num_epochs,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'test_size': args.test_size,
            'val_size': args.val_size,
            'balance_ratio': args.balance_ratio,
            'run_id': run_id,
            'train_seed': train_seed,
            'data_seed': data_seed,
        }
    }

    with open(os.path.join(task_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    with open(os.path.join(task_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"DIAL experiment summary - {args.task}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Best epoch: {best_epoch + 1}\n\n")
        f.write("Test results:\n")
        f.write(classification_report(
            test_final['labels'],
            test_final['predictions'],
            target_names=['Negative', 'Positive']
        ))

    logger.info("=" * 80)
    logger.info("Experiment complete!")
    logger.info("=" * 80)

    return results


def build_arg_parser(add_help: bool = True) -> argparse.ArgumentParser:
    """Argument parser shared by single-run and multi-run entry points."""
    parser = argparse.ArgumentParser(
        description='DIAL brain disorder classification experiment',
        add_help=add_help
    )

    # Data
    parser.add_argument('--data_path', type=str, default=r"/data/tianhao/DIAL/data/data_dict.pkl")
    parser.add_argument('--task', type=str, default='OCD',
                        choices=['Dep', 'Bip', 'DMDD', 'Schi', 'Anx', 'OCD', 'Eat', 'ADHD', 'ODD',
                                 'Cond', 'PTSD', 'ADHD_ODD_Cond', 'PPMI'], help='Task name')
    parser.add_argument('--ppmi_train_path', type=str, default=r"./data/PPMI/train_data.pkl",
                        help='PPMI train pickle path')
    parser.add_argument('--ppmi_test_path', type=str, default=r"./data/PPMI/test_data.pkl",
                        help='PPMI test pickle path')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--test_size', type=float, default=0.3, help='Hold-out test fraction')
    parser.add_argument('--val_size', type=float, default=0.1, help='Validation fraction from training split')
    parser.add_argument('--balance_ratio', type=float, default=1.0, help='Negative-to-positive balance ratio')
    parser.add_argument('--data_seed', type=int, default=20010920,
                        help='Seed for data balancing/splitting (kept fixed across runs)')
    parser.add_argument('--random_seed', type=int, default=20010920, help='Training/init seed')

    # Model
    parser.add_argument('--d_model', type=int, default=64, help='Transformer hidden dimension')
    parser.add_argument('--num_node_layers', type=int, default=2, help='Number of node encoder layers')
    parser.add_argument('--num_graph_layers', type=int, default=2, help='Number of graph Transformer layers')
    parser.add_argument('--dropout', type=float, default=0.3)

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

    main(args)
