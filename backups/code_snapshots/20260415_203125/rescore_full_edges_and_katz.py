from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_edges import read_csv_robust
from scripts.graph_pyg.export_pyg_supplychain_with_fringe import build_edges
from scripts.graph_pyg.models import EdgeMLPDecoder, GATEncoder
from scripts.graph_pyg.train_link_prediction import build_x_new


DEFAULT_BASE_DIR = PROJECT_ROOT / "data" / "processed" / "supplychain_pps5000"
DEFAULT_PYG_DIR = DEFAULT_BASE_DIR / "pyg"
DEFAULT_CKPT = PROJECT_ROOT / "outputs" / "models" / "gat_lp_best_smoke.pt"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "analysis"


def compute_bidirectional_pagerank_from_raw_directed(
    num_nodes: int, edge_index_directed_raw: np.ndarray
) -> np.ndarray:
    # Build PR on the original directed business graph (before PyG bidirectionalization).
    edges = [(int(u), int(v)) for u, v in edge_index_directed_raw.T.tolist()]
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
    return pr_mean


def infer_safe_alpha(g: nx.DiGraph, alpha_cap: float) -> tuple[float, float]:
    if g.number_of_edges() == 0:
        return float(alpha_cap), 0.0
    out_strength = dict(g.out_degree(weight="weight"))
    in_strength = dict(g.in_degree(weight="weight"))
    rho_upper = float(max(max(out_strength.values(), default=0.0), max(in_strength.values(), default=0.0)))
    if rho_upper <= 0.0:
        return float(alpha_cap), rho_upper
    alpha = min(float(alpha_cap), 0.9 / rho_upper)
    return float(alpha), rho_upper


def katz_with_backoff(
    g: nx.DiGraph,
    *,
    alpha_init: float,
    beta: float,
    weight: str,
    max_retries: int = 6,
) -> tuple[dict[int, float], float]:
    alpha = float(alpha_init)
    last_err: Exception | None = None
    for _ in range(max_retries):
        try:
            vals = nx.katz_centrality(g, alpha=alpha, beta=beta, weight=weight, max_iter=3000, tol=1e-8)
            return {int(k): float(v) for k, v in vals.items()}, alpha
        except Exception as e:  # noqa: BLE001
            last_err = e
            alpha *= 0.5
    raise RuntimeError(f"Katz failed after retries; last_error={last_err}") from last_err


def main() -> int:
    ap = argparse.ArgumentParser(description="Rescore full directed business edges and compute in/out Katz centrality.")
    ap.add_argument("--base-dir", default=str(DEFAULT_BASE_DIR))
    ap.add_argument("--pyg-dir", default=str(DEFAULT_PYG_DIR))
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--alpha-cap", type=float, default=0.1)
    ap.add_argument("--katz-beta", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--capital-col-name", default=None)
    ap.add_argument("--capital-col-idx", type=int, default=None)
    args = ap.parse_args()

    t0 = time.time()
    base_dir = Path(args.base_dir)
    pyg_dir = Path(args.pyg_dir)
    out_dir = Path(args.out_dir)
    ckpt_path = Path(args.checkpoint)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = pyg_dir / "supplychain_data.pt"
    prejoin_path = pyg_dir / "node_feature_prejoin.csv"
    edge_csv_path = base_dir / "edge_supplychain.csv"
    for p in (data_path, prejoin_path, edge_csv_path, ckpt_path):
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    data = torch.load(data_path, map_location="cpu", weights_only=False)
    prejoin = read_csv_robust(prejoin_path).copy()
    edges_df = read_csv_robust(edge_csv_path).copy()

    node_ids = list(getattr(data, "node_id", []))
    if not node_ids:
        raise ValueError("supplychain_data.pt does not contain node_id.")
    id_to_idx = {str(nid): i for i, nid in enumerate(node_ids)}
    num_nodes = len(node_ids)
    if len(prejoin) != num_nodes:
        raise ValueError(f"node_feature_prejoin.csv row mismatch: {len(prejoin)} != {num_nodes}")

    # Build both edge sets:
    # - edge_index_full_bidir: for GAT message passing
    # - edge_index_full_directed_raw: for PR features + business scoring
    edge_index_full_bidir, edge_index_full_directed_raw, edge_stats = build_edges(edges_df, id_to_idx)
    if edge_index_full_bidir.shape[1] != 100740:
        raise RuntimeError(f"Expected 100740 full bidirectional edges, got {edge_index_full_bidir.shape[1]}")

    feature_names = list(getattr(data, "feature_names", [])) or None
    x_new = build_x_new(
        data.x,
        torch.from_numpy(edge_index_full_directed_raw).to(torch.long),
        feature_names=feature_names,
        capital_col_name=args.capital_col_name,
        capital_col_idx=args.capital_col_idx,
    )

    # Runtime assert: PR is computed from the original raw directed edges.
    _ = compute_bidirectional_pagerank_from_raw_directed(num_nodes=num_nodes, edge_index_directed_raw=edge_index_full_directed_raw)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {})
    encoder = GATEncoder(
        in_channels=617,
        hidden_channels=64,
        out_channels=16,
        heads=4,
        gat_dropout=float(ckpt_args.get("gat_dropout", 0.2)),
        activation=str(ckpt_args.get("activation", "elu")),
    )
    decoder = EdgeMLPDecoder(node_dim=16)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    decoder.load_state_dict(ckpt["decoder_state_dict"])

    # Explicit eval() is required to disable GAT dropout during inference.
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        z = encoder(x_new, torch.from_numpy(edge_index_full_bidir).to(torch.long))
        logits = decoder(z, torch.from_numpy(edge_index_full_directed_raw).to(torch.long))
        # Explicit sigmoid is required to convert decoder logits into probabilities.
        probs = torch.sigmoid(logits)

    probs_np = probs.detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(probs_np).all():
        raise RuntimeError("Detected NaN/Inf in edge probabilities.")
    if float(probs_np.min()) < 0.0 or float(probs_np.max()) > 1.0:
        raise RuntimeError("Probability out of [0,1] range.")

    src = edge_index_full_directed_raw[0].astype(np.int64, copy=False)
    dst = edge_index_full_directed_raw[1].astype(np.int64, copy=False)
    edge_probs = pd.DataFrame(
        {
            "src_idx": src,
            "dst_idx": dst,
            "src_node_id": [node_ids[i] for i in src.tolist()],
            "dst_node_id": [node_ids[i] for i in dst.tolist()],
            "ai_prob": probs_np,
        }
    )
    edge_probs_path = out_dir / "full_edge_ai_probs.csv"
    edge_probs.to_csv(edge_probs_path, index=False, encoding="utf-8-sig")

    g = nx.DiGraph()
    g.add_nodes_from(range(num_nodes))
    g.add_weighted_edges_from([(int(u), int(v), float(w)) for u, v, w in zip(src.tolist(), dst.tolist(), probs_np.tolist())])
    alpha_init, rho_upper = infer_safe_alpha(g, alpha_cap=float(args.alpha_cap))

    out_katz_map, alpha_out = katz_with_backoff(
        g, alpha_init=alpha_init, beta=float(args.katz_beta), weight="weight"
    )
    in_katz_map, alpha_in = katz_with_backoff(
        g.reverse(copy=False), alpha_init=alpha_init, beta=float(args.katz_beta), weight="weight"
    )

    centrality_df = pd.DataFrame(
        {
            "node_idx": np.arange(num_nodes, dtype=np.int64),
            "node_id": node_ids,
            "in_katz": [in_katz_map.get(i, 0.0) for i in range(num_nodes)],
            "out_katz": [out_katz_map.get(i, 0.0) for i in range(num_nodes)],
        }
    )
    prejoin = prejoin.copy()
    prejoin["node_idx"] = np.arange(len(prejoin), dtype=np.int64)
    topk = int(args.topk)
    top_df = (
        centrality_df.merge(prejoin, on=["node_idx", "node_id"], how="left")
        .sort_values(by=["in_katz", "out_katz"], ascending=[False, False])
        .head(topk)
        .reset_index(drop=True)
    )

    top_path = out_dir / "katz_top20_chain_owners.csv"
    top_df.to_csv(top_path, index=False, encoding="utf-8-sig")

    meta = {
        "checkpoint": str(ckpt_path),
        "data_path": str(data_path),
        "edge_csv_path": str(edge_csv_path),
        "num_nodes": int(num_nodes),
        "num_edges_full_bidir_message_passing": int(edge_index_full_bidir.shape[1]),
        "num_edges_full_directed_raw_scoring": int(edge_index_full_directed_raw.shape[1]),
        "edge_stats": edge_stats,
        "rho_upper_bound": float(rho_upper),
        "alpha_init": float(alpha_init),
        "alpha_in_final": float(alpha_in),
        "alpha_out_final": float(alpha_out),
        "katz_beta": float(args.katz_beta),
        "topk": int(topk),
        "prob_min": float(probs_np.min()) if probs_np.size else 0.0,
        "prob_max": float(probs_np.max()) if probs_np.size else 0.0,
        "runtime_seconds": float(time.time() - t0),
    }
    meta_path = out_dir / "katz_run_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote: {edge_probs_path}")
    print(f"Wrote: {top_path}")
    print(f"Wrote: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
