"""Train a Graph Attention Network on the ENZYMES dataset.

This script reuses the ENZYMES parsing and batching utilities from
gcn_enzymes.py so the raw TU-format dataset is interpreted exactly the same way.
The main difference is the neural architecture: this file defines a true Graph
Attention Network (GAT) with multiple attention layers and multiple graph
pooling stages, then evaluates it with stratified 6-fold cross-validation.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader

from gcn_enzymes import (
    EnzymesDataset,
    GraphSample,
    collate_graphs,
    fit_attribute_scaler,
    load_enzymes_graphs,
    print_misclassification_matrices,
    save_learning_curve_figures,
    save_misclassification_figures,
    set_seed,
)


class GATLayer(nn.Module):
    """One dense multi-head graph attention layer for padded graph batches."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        heads: int,
        dropout: float,
        concat: bool = True,
        negative_slope: float = 0.2,
    ) -> None:
        """Initialize the feature projection and attention parameters.

        Each head learns:
        1. a linear projection from input features to output features, and
        2. an attention mechanism that scores every valid neighbor pair.
        """

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat

        self.linear = nn.Linear(in_features, out_features * heads, bias=False)
        self.attn_src = nn.Parameter(torch.empty(heads, out_features))
        self.attn_dst = nn.Parameter(torch.empty(heads, out_features))
        self.attn_dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset all learnable weights with Xavier initialization."""

        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply masked multi-head attention over the neighbors of each node.

        The input uses padded dense tensors:
        - x has shape (batch, nodes, features)
        - adj has shape (batch, nodes, nodes)
        - mask marks which node positions are real versus padding

        Attention is only computed for valid edges plus self-loops.
        """

        batch_size, num_nodes, _ = x.shape

        # Remove any padded-node interactions and add self-loops for every real node.
        valid_pair_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
        adj_with_loops = ((adj > 0).float() + torch.diag_embed(mask)).clamp(max=1.0)
        adj_with_loops = adj_with_loops * valid_pair_mask

        # Project node features once, then split them into multiple attention heads.
        h = self.linear(x).view(batch_size, num_nodes, self.heads, self.out_features)
        h = h.permute(0, 2, 1, 3)

        # Compute additive attention logits for every source-target node pair.
        src_scores = (h * self.attn_src.view(1, self.heads, 1, self.out_features)).sum(dim=-1)
        dst_scores = (h * self.attn_dst.view(1, self.heads, 1, self.out_features)).sum(dim=-1)
        attention_logits = self.leaky_relu(
            src_scores.unsqueeze(-1) + dst_scores.unsqueeze(-2)
        )

        # Mask out non-edges so softmax only normalizes across true neighbors.
        edge_mask = adj_with_loops.unsqueeze(1).bool()
        attention_logits = attention_logits.masked_fill(~edge_mask, -1e9)
        attention_weights = F.softmax(attention_logits, dim=-1)
        attention_weights = self.attn_dropout(attention_weights)

        # Aggregate neighbor information using the learned attention weights.
        output = torch.matmul(attention_weights, h)
        output = output * mask.unsqueeze(1).unsqueeze(-1)

        if self.concat:
            output = output.permute(0, 2, 1, 3).reshape(
                batch_size,
                num_nodes,
                self.heads * self.out_features,
            )
        else:
            output = output.mean(dim=1)

        return output * mask.unsqueeze(-1)


class GATClassifier(nn.Module):
    """A deeper graph attention model with multi-stage graph pooling."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        heads: int,
        dropout: float,
    ) -> None:
        """Build four stacked GAT blocks plus graph-level readout layers."""

        super().__init__()
        gat_width = hidden_dim * heads

        # Attention Layer 1
        self.gat1 = GATLayer(input_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
        # Attention Layer 2
        self.gat2 = GATLayer(gat_width, hidden_dim, heads=heads, dropout=dropout, concat=True)
        # Attention Layer 3
        self.gat3 = GATLayer(gat_width, hidden_dim, heads=heads, dropout=dropout, concat=True)
        # Attention Layer 4
        self.gat4 = GATLayer(gat_width, hidden_dim, heads=heads, dropout=dropout, concat=True)

        # Feature Dropout Layer
        self.dropout = nn.Dropout(dropout)
        # Dense Readout Layer
        self.readout = nn.Linear(gat_width * 8, gat_width * 2)
        # Output Classification Layer
        self.classifier = nn.Linear(gat_width * 2, num_classes)

    def graph_pool(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Pool node embeddings into one graph embedding using mean and max pooling."""

        expanded_mask = mask.unsqueeze(-1)

        # Pooling Layer: global mean pooling.
        mean_pool = (x * expanded_mask).sum(dim=1)
        mean_pool = mean_pool / mask.sum(dim=1, keepdim=True).clamp(min=1.0)

        # Pooling Layer: global max pooling.
        masked_x = x.masked_fill(expanded_mask == 0, float("-inf"))
        max_pool = masked_x.max(dim=1).values
        max_pool[max_pool == float("-inf")] = 0.0

        return torch.cat([mean_pool, max_pool], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run the batch through four attention blocks and concatenate all readouts."""

        pooled_representations: list[torch.Tensor] = []

        # Attention Layer 1
        x = self.gat1(x, adj, mask)
        x = F.elu(x)
        x = self.dropout(x)
        # Pooling Layer 1
        pooled_representations.append(self.graph_pool(x, mask))

        # Attention Layer 2
        x = self.gat2(x, adj, mask)
        x = F.elu(x)
        x = self.dropout(x)
        # Pooling Layer 2
        pooled_representations.append(self.graph_pool(x, mask))

        # Attention Layer 3
        x = self.gat3(x, adj, mask)
        x = F.elu(x)
        x = self.dropout(x)
        # Pooling Layer 3
        pooled_representations.append(self.graph_pool(x, mask))

        # Attention Layer 4
        x = self.gat4(x, adj, mask)
        x = F.elu(x)
        x = self.dropout(x)
        # Pooling Layer 4
        pooled_representations.append(self.graph_pool(x, mask))

        # Combine graph summaries from all depths before classification.
        pooled = torch.cat(pooled_representations, dim=1)
        pooled = F.elu(self.readout(pooled))
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


def run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float, list[int], list[int]]:
    """Run one train or evaluation pass over a graph data split."""

    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    predictions: list[int] = []
    targets: list[int] = []

    for features, adjacency, mask, labels in data_loader:
        # Move the current padded batch onto the active device.
        features = features.to(device)
        adjacency = adjacency.to(device)
        mask = mask.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(is_training):
            # Forward pass through the GAT classifier.
            logits = model(features, adjacency, mask)
            loss = F.cross_entropy(logits, labels)

            if is_training:
                # Standard gradient-based parameter update.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())

    average_loss = total_loss / len(data_loader.dataset)
    accuracy = accuracy_score(targets, predictions)
    return average_loss, accuracy, predictions, targets


def train_one_fold(
    graphs: list[GraphSample],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    attr_dim: int,
    device: torch.device,
    epochs: int,
    patience: int,
    batch_size: int,
    hidden_dim: int,
    heads: int,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    seed: int,
    validation_fraction: float,
    verbose: bool,
    evaluate_test: bool = True,
) -> dict[str, object]:
    """Train one GAT model for one outer cross-validation fold."""

    train_labels = np.asarray([graphs[idx].y for idx in train_indices], dtype=np.int64)
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=validation_fraction,
        random_state=seed,
    )
    subtrain_idx, val_idx = next(splitter.split(train_indices, train_labels))
    subtrain_indices = train_indices[subtrain_idx]
    val_indices = train_indices[val_idx]

    # Standardize continuous node attributes using only the training subset.
    scaler = fit_attribute_scaler(graphs, subtrain_indices, attr_dim)

    train_dataset = EnzymesDataset(graphs, subtrain_indices, scaler, attr_dim)
    val_dataset = EnzymesDataset(graphs, val_indices, scaler, attr_dim)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_graphs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_graphs,
    )
    test_loader = None
    if evaluate_test:
        test_dataset = EnzymesDataset(graphs, test_indices, scaler, attr_dim)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_graphs,
        )

    input_dim = graphs[0].x.shape[1]
    num_classes = len({graph.y for graph in graphs})
    model = GATClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        heads=heads,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    patience_counter = 0
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(1, epochs + 1):
        # Fit the model on the sub-training split.
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, device, optimizer)
        # Use validation loss for early stopping.
        val_loss, val_acc, _, _ = run_epoch(model, val_loader, device, optimizer=None)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
            print(
                f"    Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Evaluate the best checkpoint once on the validation and test splits.
    model.load_state_dict(best_state)
    val_loss, val_acc, val_predictions, val_targets = run_epoch(
        model,
        val_loader,
        device,
        optimizer=None,
    )
    val_macro_f1 = f1_score(val_targets, val_predictions, average="macro")
    metrics: dict[str, object] = {
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_macro_f1": val_macro_f1,
        "history": history,
    }

    if evaluate_test and test_loader is not None:
        test_loss, test_acc, test_predictions, test_targets = run_epoch(
            model,
            test_loader,
            device,
            optimizer=None,
        )
        macro_f1 = f1_score(test_targets, test_predictions, average="macro")
        metrics.update(
            {
                "test_loss": test_loss,
                "test_accuracy": test_acc,
                "macro_f1": macro_f1,
                "predictions": test_predictions,
                "targets": test_targets,
            }
        )
    else:
        metrics.update(
            {
                "test_loss": None,
                "test_accuracy": None,
                "macro_f1": None,
                "predictions": [],
                "targets": [],
            }
        )

    return metrics


def evaluate_candidate(
    graphs: list[GraphSample],
    labels: np.ndarray,
    attr_dim: int,
    device: torch.device,
    candidate: dict[str, float | int],
    seed: int,
    sweep_epochs: int,
    patience: int,
    validation_fraction: float,
    verbose: bool,
    search_fold_count: int,
) -> dict[str, float]:
    """Score one GAT hyperparameter candidate using validation metrics only."""

    cv = StratifiedKFold(n_splits=search_fold_count, shuffle=True, random_state=seed)
    fold_val_macro_f1: list[float] = []
    fold_val_accuracy: list[float] = []

    for train_indices, test_indices in cv.split(np.zeros(len(graphs)), labels):
        metrics = train_one_fold(
            graphs=graphs,
            train_indices=train_indices,
            test_indices=test_indices,
            attr_dim=attr_dim,
            device=device,
            epochs=sweep_epochs,
            patience=patience,
            batch_size=int(candidate["batch_size"]),
            hidden_dim=int(candidate["hidden_dim"]),
            heads=int(candidate["heads"]),
            learning_rate=float(candidate["learning_rate"]),
            weight_decay=float(candidate["weight_decay"]),
            dropout=float(candidate["dropout"]),
            seed=seed,
            validation_fraction=validation_fraction,
            verbose=verbose,
            evaluate_test=False,
        )
        fold_val_accuracy.append(float(metrics["val_accuracy"]))
        fold_val_macro_f1.append(float(metrics["val_macro_f1"]))

    return {
        "mean_val_accuracy": float(np.mean(fold_val_accuracy)),
        "mean_val_macro_f1": float(np.mean(fold_val_macro_f1)),
    }


def select_hyperparameters(
    graphs: list[GraphSample],
    labels: np.ndarray,
    attr_dim: int,
    device: torch.device,
    seed: int,
    sweep_epochs: int,
    patience: int,
    validation_fraction: float,
    verbose: bool,
    max_search_trials: int,
    search_fold_count: int,
) -> dict[str, float | int]:
    """Choose GAT hyperparameters using Optuna."""

    print("=== GAT Hyperparameter Sweep ===")
    print(f"Search folds used during tuning: {search_fold_count}")
    print(f"Maximum search trials: {max_search_trials}")
    print("Search backend: optuna")
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial) -> float:
        candidate = {
            "hidden_dim": trial.suggest_categorical("hidden_dim", [8, 12, 16, 24]),
            "heads": trial.suggest_categorical("heads", [2, 4, 6, 8]),
            "dropout": trial.suggest_categorical("dropout", [0.25, 0.35, 0.45, 0.55]),
            "learning_rate": trial.suggest_categorical("learning_rate", [3e-4, 5e-4, 1e-3]),
            "batch_size": trial.suggest_categorical("batch_size", [8, 12, 16, 24]),
            "weight_decay": trial.suggest_categorical("weight_decay", [1e-5, 1e-4, 5e-4, 1e-3]),
        }
        summary = evaluate_candidate(
            graphs=graphs,
            labels=labels,
            attr_dim=attr_dim,
            device=device,
            candidate=candidate,
            seed=seed,
            sweep_epochs=sweep_epochs,
            patience=patience,
            validation_fraction=validation_fraction,
            verbose=verbose,
            search_fold_count=search_fold_count,
        )
        trial.set_user_attr("mean_val_accuracy", summary["mean_val_accuracy"])
        return summary["mean_val_macro_f1"]

    study.optimize(objective, n_trials=max_search_trials)
    best_config = {
        "hidden_dim": int(study.best_params["hidden_dim"]),
        "heads": int(study.best_params["heads"]),
        "dropout": float(study.best_params["dropout"]),
        "learning_rate": float(study.best_params["learning_rate"]),
        "batch_size": int(study.best_params["batch_size"]),
        "weight_decay": float(study.best_params["weight_decay"]),
        "mean_validation_macro_f1": float(study.best_value),
        "mean_validation_accuracy": float(study.best_trial.user_attrs["mean_val_accuracy"]),
    }

    print("Chosen GAT hyperparameters:")
    for key, value in best_config.items():
        print(f"  {key}: {value}")
    print()
    return best_config


def main() -> None:
    """Run 6-fold cross-validation for the ENZYMES GAT model."""

    dataset_dir = Path("ENZYMES").resolve()
    sweep_epochs = 30
    final_epochs = 120
    patience = 20
    seed = 42
    validation_fraction = 0.1
    max_search_trials = 12
    search_fold_count = 4
    use_node_attributes = True
    use_node_labels = True
    verbose = False

    set_seed(seed)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    graphs, original_label_values, attr_dim = load_enzymes_graphs(
        dataset_dir=dataset_dir,
        use_node_attributes=use_node_attributes,
        use_node_labels=use_node_labels,
    )

    feature_dim = graphs[0].x.shape[1]
    num_graphs = len(graphs)
    node_counts = np.asarray([graph.x.shape[0] for graph in graphs], dtype=np.int64)
    labels = np.asarray([graph.y for graph in graphs], dtype=np.int64)

    print("=== ENZYMES Dataset Summary ===")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Graphs: {num_graphs}")
    print(f"Node count range: {node_counts.min()} - {node_counts.max()}")
    print(f"Average nodes per graph: {node_counts.mean():.2f}")
    print(f"Node feature dimension: {feature_dim}")
    print(f"Continuous attribute dimension: {attr_dim}")
    print(f"Graph classes (EC labels): {original_label_values}")
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print()

    best_hyperparameters = select_hyperparameters(
        graphs=graphs,
        labels=labels,
        attr_dim=attr_dim,
        device=device,
        seed=seed,
        sweep_epochs=sweep_epochs,
        patience=patience,
        validation_fraction=validation_fraction,
        verbose=verbose,
        max_search_trials=max_search_trials,
        search_fold_count=search_fold_count,
    )

    cv = StratifiedKFold(n_splits=6, shuffle=True, random_state=seed)
    fold_accuracies: list[float] = []
    fold_macro_f1: list[float] = []
    all_targets: list[int] = []
    all_predictions: list[int] = []
    all_histories: list[dict[str, list[float]]] = []

    print("=== Final GAT Evaluation With Chosen Hyperparameters ===")
    for key, value in best_hyperparameters.items():
        print(f"{key}: {value}")
    print()

    for fold_id, (train_indices, test_indices) in enumerate(
        cv.split(np.zeros(num_graphs), labels),
        start=1,
    ):
        # Train and evaluate one independent model for the current fold.
        print(f"=== Fold {fold_id} / 6 ===")
        metrics = train_one_fold(
            graphs=graphs,
            train_indices=train_indices,
            test_indices=test_indices,
            attr_dim=attr_dim,
            device=device,
            epochs=final_epochs,
            patience=patience,
            batch_size=int(best_hyperparameters["batch_size"]),
            hidden_dim=int(best_hyperparameters["hidden_dim"]),
            heads=int(best_hyperparameters["heads"]),
            learning_rate=float(best_hyperparameters["learning_rate"]),
            weight_decay=float(best_hyperparameters["weight_decay"]),
            dropout=float(best_hyperparameters["dropout"]),
            seed=seed,
            validation_fraction=validation_fraction,
            verbose=verbose,
            evaluate_test=True,
        )

        fold_accuracies.append(float(metrics["test_accuracy"]))
        fold_macro_f1.append(float(metrics["macro_f1"]))
        all_targets.extend(original_label_values[idx] for idx in metrics["targets"])
        all_predictions.extend(original_label_values[idx] for idx in metrics["predictions"])
        all_histories.append(metrics["history"])

        print(
            f"Fold {fold_id} test_loss={metrics['test_loss']:.4f} "
            f"test_acc={metrics['test_accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )
        print()

    # Report aggregate performance over all held-out test folds.
    print("=== 6-Fold Cross-Validation Summary ===")
    print(f"Mean accuracy: {np.mean(fold_accuracies):.4f} +/- {np.std(fold_accuracies):.4f}")
    print(f"Mean macro F1: {np.mean(fold_macro_f1):.4f} +/- {np.std(fold_macro_f1):.4f}")
    print("Chosen hyperparameters:")
    for key, value in best_hyperparameters.items():
        print(f"  {key}: {value}")
    print()
    print("=== Combined Classification Report Across All Test Folds ===")
    print(
        classification_report(
            all_targets,
            all_predictions,
            labels=original_label_values,
            zero_division=0,
        )
    )
    print_misclassification_matrices(
        targets=all_targets,
        predictions=all_predictions,
        class_labels=original_label_values,
    )
    save_misclassification_figures(
        targets=all_targets,
        predictions=all_predictions,
        class_labels=original_label_values,
        model_name="GAT",
    )
    save_learning_curve_figures(
        histories=all_histories,
        model_name="GAT",
    )


if __name__ == "__main__":
    main()
