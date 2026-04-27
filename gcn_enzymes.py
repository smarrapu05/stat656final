"""Train a Graph Convolutional Network on the ENZYMES dataset.

The ENZYMES folder uses the TU dataset text format described in README.txt:

- ENZYMES_A.txt stores the sparse adjacency list for every graph.
- ENZYMES_graph_indicator.txt maps each global node id to a graph id.
- ENZYMES_graph_labels.txt stores the graph-level EC class labels.
- ENZYMES_node_attributes.txt stores continuous node features.
- ENZYMES_node_labels.txt stores categorical node labels.

This script reconstructs each protein graph, builds node features from the
optional node attributes and node labels, and trains a true Graph Convolutional
Network (GCN) with stratified 6-fold cross-validation.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


@dataclass
class GraphSample:
    """Container for one graph's node features, adjacency matrix, and class label."""

    x: np.ndarray
    adj: np.ndarray
    y: int
    graph_id: int


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch so cross-validation runs are reproducible."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_int_lines(path: Path) -> list[int]:
    """Read a text file containing one integer per line and return the parsed values."""

    with path.open("r", encoding="utf-8") as handle:
        return [int(line.strip()) for line in handle if line.strip()]


def build_node_features(
    dataset_dir: Path,
    use_node_attributes: bool,
    use_node_labels: bool,
) -> tuple[np.ndarray, int]:
    """Assemble the node feature matrix used by the GCN.
    The ENZYMES dataset provides two optional node-level feature sources:
    continuous attributes and discrete node labels. This function concatenates
    them into one feature vector per node. If both are disabled or missing, it
    falls back to a single constant feature so the GCN can still run.
    """

    feature_parts: list[np.ndarray] = []
    attr_dim = 0

    attr_path = dataset_dir / "ENZYMES_node_attributes.txt"
    if use_node_attributes and attr_path.exists():
        # Continuous biochemical descriptors for each node.
        node_attributes = np.loadtxt(attr_path, delimiter=",", dtype=np.float32)
        if node_attributes.ndim == 1:
            node_attributes = node_attributes.reshape(-1, 1)
        feature_parts.append(node_attributes)
        attr_dim = node_attributes.shape[1]

    label_path = dataset_dir / "ENZYMES_node_labels.txt"
    if use_node_labels and label_path.exists():
        # Convert categorical node labels into one-hot vectors for the GCN.
        node_labels = np.asarray(read_int_lines(label_path), dtype=np.int64)
        label_min = int(node_labels.min())
        label_max = int(node_labels.max())
        num_label_values = label_max - label_min + 1
        label_indices = node_labels - label_min
        one_hot_labels = np.eye(num_label_values, dtype=np.float32)[label_indices]
        feature_parts.append(one_hot_labels)

    if not feature_parts:
        # Final fallback: give every node the same scalar feature.
        graph_indicator = read_int_lines(dataset_dir / "ENZYMES_graph_indicator.txt")
        fallback_features = np.ones((len(graph_indicator), 1), dtype=np.float32)
        feature_parts.append(fallback_features)

    features = np.concatenate(feature_parts, axis=1).astype(np.float32)
    return features, attr_dim


def load_enzymes_graphs(
    dataset_dir: Path,
    use_node_attributes: bool = True,
    use_node_labels: bool = True,
) -> tuple[list[GraphSample], list[int], int]:
    """Parse the raw TU-format ENZYMES files into individual graph objects.

    The raw files use a single global node index space across all 600 graphs.
    This function rebuilds each graph by:
    1. reading graph membership from ENZYMES_graph_indicator.txt,
    2. slicing the node feature matrix for each graph,
    3. rebuilding a local adjacency matrix from ENZYMES_A.txt, and
    4. encoding the graph labels into zero-based class indices for PyTorch.
    """

    graph_indicator = np.asarray(
        read_int_lines(dataset_dir / "ENZYMES_graph_indicator.txt"),
        dtype=np.int64,
    )
    graph_labels_raw = read_int_lines(dataset_dir / "ENZYMES_graph_labels.txt")
    node_features, attr_dim = build_node_features(
        dataset_dir,
        use_node_attributes=use_node_attributes,
        use_node_labels=use_node_labels,
    )

    num_nodes = len(graph_indicator)
    num_graphs = len(graph_labels_raw)
    nodes_per_graph = np.bincount(graph_indicator, minlength=num_graphs + 1)[1:]

    # Map each global node id from the raw files to its node index inside its own graph.
    local_node_index = np.zeros(num_nodes + 1, dtype=np.int64)
    running_counts = np.zeros(num_graphs + 1, dtype=np.int64)
    for global_node_id, graph_id in enumerate(graph_indicator, start=1):
        local_node_index[global_node_id] = running_counts[graph_id]
        running_counts[graph_id] += 1

    graph_features: list[np.ndarray] = []
    graph_adjs: list[np.ndarray] = []
    graph_node_positions: list[list[int]] = [[] for _ in range(num_graphs)]
    for global_idx, graph_id in enumerate(graph_indicator):
        # Collect the row positions of every node that belongs to each graph.
        graph_node_positions[graph_id - 1].append(global_idx)

    for graph_id in range(1, num_graphs + 1):
        node_positions = graph_node_positions[graph_id - 1]
        x = node_features[node_positions]
        node_count = int(nodes_per_graph[graph_id - 1])
        # Each graph gets its own dense adjacency matrix for batching simplicity.
        adj = np.zeros((node_count, node_count), dtype=np.float32)
        graph_features.append(x)
        graph_adjs.append(adj)

    with (dataset_dir / "ENZYMES_A.txt").open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            src_text, dst_text = line.split(",")
            src = int(src_text.strip())
            dst = int(dst_text.strip())
            graph_id = int(graph_indicator[src - 1])
            if graph_id != int(graph_indicator[dst - 1]):
                raise ValueError(
                    f"Edge ({src}, {dst}) crosses graph boundaries, which violates "
                    "the TU-format block-diagonal assumption."
                )

            src_local = int(local_node_index[src])
            dst_local = int(local_node_index[dst])
            # The ENZYMES graphs are treated as undirected for this GCN.
            graph_adjs[graph_id - 1][src_local, dst_local] = 1.0
            graph_adjs[graph_id - 1][dst_local, src_local] = 1.0

    # Convert original EC labels (1..6) into zero-based class ids.
    unique_labels = sorted(set(graph_labels_raw))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    encoded_labels = [label_to_index[label] for label in graph_labels_raw]

    graphs: list[GraphSample] = []
    for graph_idx in range(num_graphs):
        graphs.append(
            GraphSample(
                x=graph_features[graph_idx],
                adj=graph_adjs[graph_idx],
                y=encoded_labels[graph_idx],
                graph_id=graph_idx + 1,
            )
        )

    return graphs, unique_labels, attr_dim


def fit_attribute_scaler(
    graphs: list[GraphSample],
    train_indices: Iterable[int],
    attr_dim: int,
) -> StandardScaler | None:
    """Fit a scaler on the continuous node attributes from the training split only."""

    if attr_dim <= 0:
        return None

    stacked_attributes = np.vstack([graphs[idx].x[:, :attr_dim] for idx in train_indices])
    scaler = StandardScaler()
    scaler.fit(stacked_attributes)
    return scaler


class EnzymesDataset(Dataset):
    """Dataset wrapper that applies fold-specific preprocessing to selected graphs."""

    def __init__(
        self,
        graphs: list[GraphSample],
        indices: Iterable[int],
        scaler: StandardScaler | None,
        attr_dim: int,
    ) -> None:
        """Build a subset of graphs and standardize only the continuous attributes."""

        self.samples: list[GraphSample] = []
        for idx in indices:
            graph = graphs[idx]
            x = graph.x.copy()
            if scaler is not None and attr_dim > 0:
                # Only the first attr_dim columns are continuous attributes.
                x[:, :attr_dim] = scaler.transform(x[:, :attr_dim])
            self.samples.append(
                GraphSample(
                    x=x.astype(np.float32),
                    adj=graph.adj.astype(np.float32),
                    y=graph.y,
                    graph_id=graph.graph_id,
                )
            )

    def __len__(self) -> int:
        """Return the number of graphs in this subset."""

        return len(self.samples)

    def __getitem__(self, index: int) -> GraphSample:
        """Fetch one graph sample for DataLoader batching."""

        return self.samples[index]


def collate_graphs(batch: list[GraphSample]) -> tuple[torch.Tensor, ...]:
    """Pad variable-sized graphs so they can be processed together in one mini-batch.
    Each batch is padded up to the largest graph in that batch. The mask records
    which node positions are real and which are padding, so the GCN and pooling
    layers can ignore padded entries.
    """

    batch_size = len(batch)
    max_nodes = max(sample.x.shape[0] for sample in batch)
    feature_dim = batch[0].x.shape[1]

    features = np.zeros((batch_size, max_nodes, feature_dim), dtype=np.float32)
    adjacency = np.zeros((batch_size, max_nodes, max_nodes), dtype=np.float32)
    mask = np.zeros((batch_size, max_nodes), dtype=np.float32)
    labels = np.zeros(batch_size, dtype=np.int64)

    for batch_idx, sample in enumerate(batch):
        node_count = sample.x.shape[0]
        # Copy each graph into the top-left corner of the padded batch tensors.
        features[batch_idx, :node_count] = sample.x
        adjacency[batch_idx, :node_count, :node_count] = sample.adj
        mask[batch_idx, :node_count] = 1.0
        labels[batch_idx] = sample.y

    return (
        torch.from_numpy(features),
        torch.from_numpy(adjacency),
        torch.from_numpy(mask),
        torch.from_numpy(labels),
    )


class GCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        """Create one graph convolution layer using the normalized GCN update rule."""

        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one graph convolution step to a padded batch of graphs.

        The operation is:
        H' = D^(-1/2) (A + I) D^(-1/2) H W
        where the mask prevents padded nodes from contributing to the result.
        """

        # Remove any accidental connectivity involving padded nodes.
        valid_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
        masked_adj = adj * valid_mask
        # Add self-loops so each node keeps its own information.
        self_loops = torch.diag_embed(mask)
        adj_hat = masked_adj + self_loops

        # Build the symmetric normalized adjacency matrix used by standard GCNs.
        degree = adj_hat.sum(dim=-1).clamp(min=1.0)
        degree_inv_sqrt = degree.pow(-0.5)
        normalized_adj = (
            degree_inv_sqrt.unsqueeze(-1) * adj_hat * degree_inv_sqrt.unsqueeze(-2)
        )

        # First apply a learned linear projection, then aggregate neighbors.
        support = self.linear(x)
        output = torch.matmul(normalized_adj, support)
        return output * mask.unsqueeze(-1)


class GCNClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        """Construct a deeper multi-scale GCN for graph classification.

        This version uses four graph convolution layers. After every convolution,
        the model performs graph-level mean pooling and max pooling. The pooled
        summaries from all four stages are concatenated so the classifier can use
        both shallow and deep structural information.
        """

        super().__init__()
        # GCN Layer 1
        self.gcn1 = GCNLayer(input_dim, hidden_dim)
        # GCN Layer 2
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        # GCN Layer 3
        self.gcn3 = GCNLayer(hidden_dim, hidden_dim)
        # GCN Layer 4
        self.gcn4 = GCNLayer(hidden_dim, hidden_dim)
        # Dropout Layer
        self.dropout = nn.Dropout(dropout)
        # Dense Readout Layer
        self.readout = nn.Linear(hidden_dim * 8, hidden_dim * 2)
        # Output Classification Layer
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def graph_pool(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Summarize node embeddings with both mean pooling and max pooling.

        Mean pooling captures the average structural signal in a graph, while
        max pooling keeps the strongest node-level activation. Concatenating both
        gives the classifier a richer graph representation.
        """

        expanded_mask = mask.unsqueeze(-1)

        # Pooling Layer: global mean pooling over the valid nodes.
        mean_pool = (x * expanded_mask).sum(dim=1)
        mean_pool = mean_pool / mask.sum(dim=1, keepdim=True).clamp(min=1.0)

        # Pooling Layer: global max pooling over the valid nodes.
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
        """Run the padded batch through four GCN blocks and multi-stage pooling."""

        pooled_representations: list[torch.Tensor] = []

        # GCN Layer 1
        x = self.gcn1(x, adj, mask)
        x = F.relu(x)
        x = self.dropout(x)
        # Pooling Layer 1
        pooled_representations.append(self.graph_pool(x, mask))

        # GCN Layer 2
        x = self.gcn2(x, adj, mask)
        x = F.relu(x)
        x = self.dropout(x)
        # Pooling Layer 2
        pooled_representations.append(self.graph_pool(x, mask))

        # GCN Layer 3
        x = self.gcn3(x, adj, mask)
        x = F.relu(x)
        x = self.dropout(x)
        # Pooling Layer 3
        pooled_representations.append(self.graph_pool(x, mask))

        # GCN Layer 4
        x = self.gcn4(x, adj, mask)
        x = F.relu(x)
        x = self.dropout(x)
        # Pooling Layer 4
        pooled_representations.append(self.graph_pool(x, mask))

        # Concatenate graph summaries from every depth of the network.
        pooled = torch.cat(pooled_representations, dim=1)
        pooled = F.relu(self.readout(pooled))
        pooled = self.dropout(pooled)

        logits = self.classifier(pooled)
        return logits


def run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float, list[int], list[int]]:
    """Run one full pass over a data split and return loss plus predictions."""

    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    predictions: list[int] = []
    targets: list[int] = []

    for features, adjacency, mask, labels in data_loader:
        # Move the entire padded batch to CPU or GPU.
        features = features.to(device)
        adjacency = adjacency.to(device)
        mask = mask.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(is_training):
            # Forward pass through the GCN classifier.
            logits = model(features, adjacency, mask)
            loss = F.cross_entropy(logits, labels)

            if is_training:
                # Standard optimization step during training mode.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())

    average_loss = total_loss / len(data_loader.dataset)
    accuracy = accuracy_score(targets, predictions)
    return average_loss, accuracy, predictions, targets


def format_count_matrix(matrix: np.ndarray, class_labels: list[int]) -> str:
    """Format a count matrix so it prints cleanly in the terminal."""

    row_label_width = max(len("true\\pred"), max(len(str(label)) for label in class_labels))
    cell_width = max(8, max(len(str(int(value))) for value in matrix.flatten()))
    header = " " * (row_label_width + 2) + "".join(
        f"{str(label):>{cell_width}}" for label in class_labels
    )

    lines = [header]
    for row_label, row_values in zip(class_labels, matrix):
        row_text = f"{str(row_label):>{row_label_width}}  " + "".join(
            f"{int(value):>{cell_width}}" for value in row_values
        )
        lines.append(row_text)
    return "\n".join(lines)


def build_confusion_matrices(
    targets: list[int],
    predictions: list[int],
    class_labels: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Build the full confusion matrix and the off-diagonal-only error matrix."""

    confusion = confusion_matrix(targets, predictions, labels=class_labels)
    misclassification = confusion.copy()
    np.fill_diagonal(misclassification, 0)
    return confusion, misclassification


def save_matrix_figure(
    matrix: np.ndarray,
    class_labels: list[int],
    title: str,
    output_path: Path,
    cmap: str,
) -> None:
    """Save a seaborn heatmap that is clean enough for report or paper use."""

    plt.figure(figsize=(8, 6), dpi=300)
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap=cmap,
        linewidths=0.5,
        linecolor="white",
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar_kws={"label": "Count"},
    )
    plt.title(title, fontsize=14)
    plt.xlabel("Predicted EC Class", fontsize=12)
    plt.ylabel("True EC Class", fontsize=12)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def print_misclassification_matrices(
    targets: list[int],
    predictions: list[int],
    class_labels: list[int],
) -> None:
    """Print a confusion matrix and an off-diagonal misclassification matrix."""

    confusion, misclassification = build_confusion_matrices(
        targets=targets,
        predictions=predictions,
        class_labels=class_labels,
    )

    print("=== Confusion Matrix Across All Test Folds ===")
    print(format_count_matrix(confusion, class_labels))
    print()
    print("=== Misclassification Matrix Across All Test Folds ===")
    print(format_count_matrix(misclassification, class_labels))
    print()


def save_misclassification_figures(
    targets: list[int],
    predictions: list[int],
    class_labels: list[int],
    model_name: str,
) -> None:
    """Save publication-style confusion and misclassification heatmap figures."""

    confusion, misclassification = build_confusion_matrices(
        targets=targets,
        predictions=predictions,
        class_labels=class_labels,
    )

    confusion_path = Path(f"{model_name.lower()}_confusion_matrix.png").resolve()
    misclassification_path = Path(f"{model_name.lower()}_misclassification_matrix.png").resolve()

    save_matrix_figure(
        matrix=confusion,
        class_labels=class_labels,
        title=f"{model_name} Confusion Matrix",
        output_path=confusion_path,
        cmap="Blues",
    )
    save_matrix_figure(
        matrix=misclassification,
        class_labels=class_labels,
        title=f"{model_name} Misclassification Matrix",
        output_path=misclassification_path,
        cmap="Reds",
    )

    print(f"Saved figure: {confusion_path}")
    print(f"Saved figure: {misclassification_path}")
    print()


def pad_history(values: list[float], target_len: int) -> np.ndarray:
    """Extend a fold history to a common length by repeating its final value."""

    history_array = np.array(values, dtype=float)
    if len(history_array) < target_len:
        history_array = np.concatenate(
            [history_array, np.full(target_len - len(history_array), history_array[-1])]
        )
    return history_array


def save_learning_curve_figures(
    histories: list[dict[str, list[float]]],
    model_name: str,
) -> None:
    """Save mean +- standard deviation learning curves across folds."""

    if not histories:
        return

    max_epochs_run = max(len(history["train_loss"]) for history in histories)
    epoch_axis = np.arange(1, max_epochs_run + 1)
    figure_specs = [
        ("train_loss", "val_loss", "Loss", Path(f"{model_name.lower()}_learning_curve_loss.png").resolve()),
        (
            "train_acc",
            "val_acc",
            "Accuracy",
            Path(f"{model_name.lower()}_learning_curve_accuracy.png").resolve(),
        ),
    ]

    for train_key, val_key, ylabel, output_path in figure_specs:
        train_matrix = np.vstack([pad_history(history[train_key], max_epochs_run) for history in histories])
        val_matrix = np.vstack([pad_history(history[val_key], max_epochs_run) for history in histories])

        train_mean, train_std = train_matrix.mean(axis=0), train_matrix.std(axis=0)
        val_mean, val_std = val_matrix.mean(axis=0), val_matrix.std(axis=0)

        plt.figure(figsize=(9, 5), dpi=300)
        plt.plot(epoch_axis, train_mean, label="Train", color="#1f77b4", linewidth=2)
        plt.fill_between(epoch_axis, train_mean - train_std, train_mean + train_std, color="#1f77b4", alpha=0.2)
        plt.plot(epoch_axis, val_mean, label="Validation", color="#d62728", linewidth=2)
        plt.fill_between(epoch_axis, val_mean - val_std, val_mean + val_std, color="#d62728", alpha=0.2)
        plt.title(f"{model_name} {ylabel} Learning Curve (mean +- 1 std across folds)", fontsize=14)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved figure: {output_path}")
    print()


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
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    seed: int,
    validation_fraction: float,
    verbose: bool,
    evaluate_test: bool = True,
) -> dict[str, object]:
    """Train and evaluate the GCN on one cross-validation fold.

    Each outer fold is split again into a sub-training portion and a validation
    portion. The validation split is used only for early stopping and model
    selection. The held-out test fold is used once at the end.
    """

    train_labels = np.asarray([graphs[idx].y for idx in train_indices], dtype=np.int64)
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=validation_fraction,
        random_state=seed,
    )
    subtrain_idx, val_idx = next(splitter.split(train_indices, train_labels))
    subtrain_indices = train_indices[subtrain_idx]
    val_indices = train_indices[val_idx]

    # Fit preprocessing only on the training portion to avoid data leakage.
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
    model = GCNClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
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
        # Train on the sub-training split.
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, device, optimizer)
        # Monitor validation loss for early stopping.
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

    # Reload the best validation checkpoint before touching the held-out test fold.
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
    """Score one hyperparameter candidate using validation metrics only."""

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
    """Choose GCN hyperparameters using Optuna."""

    print("=== GCN Hyperparameter Sweep ===")
    print(f"Search folds used during tuning: {search_fold_count}")
    print(f"Maximum search trials: {max_search_trials}")
    print("Search backend: optuna")
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial) -> float:
        candidate = {
            "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 48, 64, 96, 128]),
            "dropout": trial.suggest_categorical("dropout", [0.20, 0.30, 0.40, 0.50]),
            "learning_rate": trial.suggest_categorical("learning_rate", [3e-4, 5e-4, 1e-3, 2e-3]),
            "batch_size": trial.suggest_categorical("batch_size", [16, 24, 32]),
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
        "dropout": float(study.best_params["dropout"]),
        "learning_rate": float(study.best_params["learning_rate"]),
        "batch_size": int(study.best_params["batch_size"]),
        "weight_decay": float(study.best_params["weight_decay"]),
        "mean_validation_macro_f1": float(study.best_value),
        "mean_validation_accuracy": float(study.best_trial.user_attrs["mean_val_accuracy"]),
    }

    print("Chosen GCN hyperparameters:")
    for key, value in best_config.items():
        print(f"  {key}: {value}")
    print()
    return best_config


def main() -> None:
    """Run the full ENZYMES experiment from raw files through 6-fold evaluation."""

    dataset_dir = Path("ENZYMES").resolve()
    sweep_epochs = 40
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

    print("=== Final GCN Evaluation With Chosen Hyperparameters ===")
    for key, value in best_hyperparameters.items():
        print(f"{key}: {value}")
    print()

    for fold_id, (train_indices, test_indices) in enumerate(cv.split(np.zeros(num_graphs), labels), start=1):
        # Train one model per fold and keep its held-out test predictions.
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

    # Summarize the cross-validation performance over all six held-out folds.
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
        model_name="GCN",
    )
    save_learning_curve_figures(
        histories=all_histories,
        model_name="GCN",
    )


if __name__ == "__main__":
    main()
