from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch_geometric.utils import negative_sampling

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.graph_pyg.models import EdgeMLPDecoder, GATEncoder

DEFAULT_PYG_DIR = PROJECT_ROOT / "data" / "processed" / "supplychain_pps5000" / "pyg"
CAPITAL_NAME_CANDIDATES = ("注册资本_raw", "注册资本", "注册资金", "capital")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_capital_column(
    feature_names: list[str] | None,
    fallback_col: str | None,
    fallback_idx: int | None,
) -> int:
    if feature_names:
        for candidate in CAPITAL_NAME_CANDIDATES:
            if candidate in feature_names:
                return int(feature_names.index(candidate))
        if fallback_col and fallback_col in feature_names:
            return int(feature_names.index(fallback_col))
    if fallback_idx is not None:
        return int(fallback_idx)
    raise ValueError(
        "Cannot resolve capital column. Provide --capital-col-name or --capital-col-idx, "
        "or export feature_names containing one of: "
        + ", ".join(CAPITAL_NAME_CANDIDATES)
    )


def _minmax_1d(values: np.ndarray) -> np.ndarray:
    scaler = MinMaxScaler()
    return scaler.fit_transform(values.reshape(-1, 1)).astype(np.float32)


def compute_bidirectional_pagerank_feature(num_nodes: int, train_pos_directed: np.ndarray) -> np.ndarray:
    edges = [(int(u), int(v)) for u, v in train_pos_directed.T.tolist()]
    g_forward = nx.DiGraph()
    g_reverse = nx.DiGraph()
    g_forward.add_nodes_from(range(num_nodes))
    g_reverse.add_nodes_from(range(num_nodes))
    g_forward.add_edges_from(edges)
    g_reverse.add_edges_from([(v, u) for u, v in edges])

    pr_forward = nx.pagerank(g_forward)
    pr_reverse = nx.pagerank(g_reverse)

    pr_mean = np.array(
        [(float(pr_forward[i]) + float(pr_reverse[i])) * 0.5 for i in range(num_nodes)],
        dtype=np.float32,
    )
    return _minmax_1d(pr_mean)


def build_x_new(
    x_raw: torch.Tensor,
    train_pos_directed: torch.Tensor,
    *,
    feature_names: list[str] | None,
    capital_col_name: str | None,
    capital_col_idx: int | None,
) -> torch.Tensor:
    x_np = x_raw.detach().cpu().numpy().astype(np.float32, copy=True)
    if x_np.ndim != 2 or x_np.shape[1] != 616:
        raise ValueError(f"Expected raw x shape [N, 616], got {tuple(x_np.shape)}")

    capital_idx = resolve_capital_column(feature_names, capital_col_name, capital_col_idx)
    if not (0 <= capital_idx < x_np.shape[1]):
        raise ValueError(f"Capital column index out of range: {capital_idx}")

    capital_raw = x_np[:, capital_idx]
    one_hot_rest = np.delete(x_np, capital_idx, axis=1)
    if one_hot_rest.shape[1] != 615:
        raise ValueError(f"Expected one_hot_rest dim 615 after slicing, got {one_hot_rest.shape[1]}")

    capital_norm = _minmax_1d(np.log1p(np.clip(capital_raw, a_min=0.0, a_max=None)).astype(np.float32).reshape(-1))
    pr_norm = compute_bidirectional_pagerank_feature(
        num_nodes=x_np.shape[0],
        train_pos_directed=train_pos_directed.detach().cpu().numpy().astype(np.int64, copy=False),
    )
    x_new = np.concatenate([capital_norm, pr_norm, one_hot_rest.astype(np.float32)], axis=1).astype(np.float32)
    if x_new.shape[1] != 617:
        raise ValueError(f"Expected X_new dim 617, got {x_new.shape[1]}")
    return torch.from_numpy(x_new)


def sample_train_negatives(train_pos_directed: torch.Tensor, num_nodes: int, seed: int) -> torch.Tensor:
    torch.manual_seed(int(seed))
    neg = negative_sampling(
        edge_index=train_pos_directed,
        num_nodes=num_nodes,
        num_neg_samples=int(train_pos_directed.size(1)),
        method="sparse",
        force_undirected=False,
    )
    return neg.to(torch.long)


def to_directed(edge_undirected: torch.Tensor) -> torch.Tensor:
    rev = edge_undirected[[1, 0], :]
    return torch.cat([edge_undirected, rev], dim=1).to(torch.long)


def check_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"Detected NaN/Inf in {name}")


def edge_logits_and_labels(
    decoder: nn.Module,
    z: torch.Tensor,
    pos_edges: torch.Tensor,
    neg_edges: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pos_logits = decoder(z, pos_edges)
    neg_logits = decoder(z, neg_edges)
    logits = torch.cat([pos_logits, neg_logits], dim=0)
    labels = torch.cat(
        [
            torch.ones(pos_logits.size(0), device=logits.device),
            torch.zeros(neg_logits.size(0), device=logits.device),
        ],
        dim=0,
    )
    return logits, labels


def compute_metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    # BCEWithLogitsLoss consumes raw logits; metrics consume probabilities.
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy()
    auc = float(roc_auc_score(y_true, probs))
    ap = float(average_precision_score(y_true, probs))
    acc = float(((probs >= 0.5).astype(np.float32) == y_true).mean())
    return {"auc": auc, "ap": ap, "acc": acc}


def main() -> int:
    ap = argparse.ArgumentParser(description="Train GAT encoder + MLP decoder for link prediction.")
    ap.add_argument("--pyg-dir", default=str(DEFAULT_PYG_DIR))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--gat-dropout", type=float, default=0.2)
    ap.add_argument("--activation", default="elu", choices=["elu", "leaky_relu"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--capital-col-name", default=None)
    ap.add_argument("--capital-col-idx", type=int, default=None)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument(
        "--save-best-path",
        default=str(PROJECT_ROOT / "outputs" / "models" / "gat_lp_best.pt"),
    )
    args = ap.parse_args()

    set_seed(int(args.seed))
    pyg_dir = Path(args.pyg_dir)
    data_path = pyg_dir / "supplychain_data.pt"
    split_path = pyg_dir / "split_edges.pt"
    if not data_path.exists() or not split_path.exists():
        raise FileNotFoundError(f"Missing required files in {pyg_dir}: supplychain_data.pt / split_edges.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    split = torch.load(split_path, map_location="cpu", weights_only=False)

    if "train_pos_edge_index_directed" not in split:
        raise KeyError("split_edges.pt missing train_pos_edge_index_directed. Re-run export script first.")

    feature_names = list(getattr(data, "feature_names", [])) or None
    x_new = build_x_new(
        data.x,
        split["train_pos_edge_index_directed"],
        feature_names=feature_names,
        capital_col_name=args.capital_col_name,
        capital_col_idx=args.capital_col_idx,
    )
    check_finite("X_new", x_new)

    mp_edge_index = split["train_message_passing_edge_index"].to(torch.long).to(device)
    train_pos_directed = split["train_pos_edge_index_directed"].to(torch.long).to(device)
    val_pos_directed = to_directed(split["val_pos_edge_index_undirected"].to(torch.long)).to(device)
    val_neg_directed = to_directed(split["val_neg_edge_index_undirected"].to(torch.long)).to(device)
    test_pos_directed = to_directed(split["test_pos_edge_index_undirected"].to(torch.long)).to(device)
    test_neg_directed = to_directed(split["test_neg_edge_index_undirected"].to(torch.long)).to(device)

    x_new = x_new.to(device)
    encoder = GATEncoder(
        in_channels=617,
        hidden_channels=64,
        out_channels=16,
        heads=4,
        gat_dropout=float(args.gat_dropout),
        activation=args.activation,
    ).to(device)
    decoder = EdgeMLPDecoder(node_dim=16).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    criterion = nn.BCEWithLogitsLoss()
    best_val_auc = float("-inf")
    best_epoch = 0
    patience_counter = 0
    save_best_path = Path(args.save_best_path)
    save_best_path.parent.mkdir(parents=True, exist_ok=True)

    num_nodes = int(split["num_nodes"])
    for epoch in range(1, int(args.epochs) + 1):
        encoder.train()
        decoder.train()
        optimizer.zero_grad()

        train_neg_directed = sample_train_negatives(
            train_pos_directed=train_pos_directed,
            num_nodes=num_nodes,
            seed=int(args.seed + epoch),
        ).to(device)

        z = encoder(x_new, mp_edge_index)
        check_finite("z", z)
        logits, labels = edge_logits_and_labels(decoder, z, train_pos_directed, train_neg_directed)
        check_finite("train_logits", logits)
        loss = criterion(logits, labels)
        check_finite("train_loss", loss)
        loss.backward()
        nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), max_norm=2.0)
        optimizer.step()

        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            z_eval = encoder(x_new, mp_edge_index)
            val_logits, val_labels = edge_logits_and_labels(decoder, z_eval, val_pos_directed, val_neg_directed)
            val_metrics = compute_metrics_from_logits(val_logits, val_labels)

        improvement = float(val_metrics["auc"] - best_val_auc)
        if improvement > float(args.min_delta):
            best_val_auc = float(val_metrics["auc"])
            best_epoch = int(epoch)
            patience_counter = 0
            torch.save(
                {
                    "encoder_state_dict": encoder.state_dict(),
                    "decoder_state_dict": decoder.state_dict(),
                    "best_epoch": int(best_epoch),
                    "best_val_auc": float(best_val_auc),
                    "best_val_ap": float(val_metrics["ap"]),
                    "best_val_acc": float(val_metrics["acc"]),
                    "args": vars(args),
                },
                save_best_path,
            )
        else:
            patience_counter += 1

        print(
            f"epoch={epoch:03d} "
            f"train_loss={loss.item():.6f} "
            f"val_auc={val_metrics['auc']:.4f} "
            f"val_ap={val_metrics['ap']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} "
            f"best_val_auc={best_val_auc:.4f} "
            f"patience_counter={patience_counter}"
        )
        if patience_counter >= int(args.patience):
            print(
                f"early_stop epoch={epoch:03d} "
                f"best_epoch={best_epoch:03d} "
                f"best_val_auc={best_val_auc:.4f}"
            )
            break

    if best_epoch <= 0 or not save_best_path.exists():
        raise RuntimeError("No best checkpoint saved. Cannot run final test evaluation.")

    best_ckpt = torch.load(save_best_path, map_location=device, weights_only=False)
    encoder.load_state_dict(best_ckpt["encoder_state_dict"])
    decoder.load_state_dict(best_ckpt["decoder_state_dict"])
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        z_final = encoder(x_new, mp_edge_index)
        test_logits, test_labels = edge_logits_and_labels(decoder, z_final, test_pos_directed, test_neg_directed)
        test_metrics = compute_metrics_from_logits(test_logits, test_labels)

    print(
        f"final_test best_epoch={best_ckpt['best_epoch']} "
        f"test_auc={test_metrics['auc']:.4f} "
        f"test_ap={test_metrics['ap']:.4f} "
        f"test_acc={test_metrics['acc']:.4f} "
        f"checkpoint={save_best_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
