from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SplitResult:
    train_pos_undirected: np.ndarray
    val_pos_undirected: np.ndarray
    test_pos_undirected: np.ndarray
    train_neg_undirected: np.ndarray
    val_neg_undirected: np.ndarray
    test_neg_undirected: np.ndarray
    train_edge_index_bidirectional: np.ndarray
    stats: dict[str, object]


def _as_undirected_unique(edge_index: np.ndarray) -> np.ndarray:
    if edge_index.size == 0:
        return np.empty((2, 0), dtype=np.int64)
    src = edge_index[0].astype(np.int64, copy=False)
    dst = edge_index[1].astype(np.int64, copy=False)
    mask = src != dst
    src = src[mask]
    dst = dst[mask]
    if src.size == 0:
        return np.empty((2, 0), dtype=np.int64)
    lo = np.minimum(src, dst)
    hi = np.maximum(src, dst)
    undirected = np.stack([lo, hi], axis=1)
    undirected = np.unique(undirected, axis=0)
    return undirected.T.astype(np.int64, copy=False)


def _to_edge_set(edges_undirected: np.ndarray) -> set[tuple[int, int]]:
    if edges_undirected.size == 0:
        return set()
    return {(int(u), int(v)) for u, v in edges_undirected.T.tolist()}


def _set_to_edge_index(edges: set[tuple[int, int]]) -> np.ndarray:
    if not edges:
        return np.empty((2, 0), dtype=np.int64)
    arr = np.array(sorted(edges), dtype=np.int64)
    return arr.T


def _build_components_and_forest(num_nodes: int, undirected_edges: np.ndarray) -> tuple[list[list[int]], set[tuple[int, int]]]:
    adj: dict[int, list[int]] = defaultdict(list)
    active_nodes: set[int] = set()
    for u, v in undirected_edges.T.tolist():
        ui = int(u)
        vi = int(v)
        adj[ui].append(vi)
        adj[vi].append(ui)
        active_nodes.add(ui)
        active_nodes.add(vi)

    visited: set[int] = set()
    components: list[list[int]] = []
    forest_edges: set[tuple[int, int]] = set()

    for start in sorted(active_nodes):
        if start in visited:
            continue
        comp: list[int] = []
        q: deque[int] = deque([start])
        visited.add(start)
        while q:
            cur = q.popleft()
            comp.append(cur)
            for nxt in adj[cur]:
                if nxt in visited:
                    continue
                visited.add(nxt)
                q.append(nxt)
                a, b = (cur, nxt) if cur <= nxt else (nxt, cur)
                forest_edges.add((a, b))
        components.append(comp)

    # Nodes that never appear in any edge are not part of link prediction splits.
    _ = num_nodes
    return components, forest_edges


def _to_bidirectional(edge_undirected: np.ndarray) -> np.ndarray:
    if edge_undirected.size == 0:
        return np.empty((2, 0), dtype=np.int64)
    rev = edge_undirected[[1, 0], :]
    return np.concatenate([edge_undirected, rev], axis=1)


def _sample_negatives_with_pyg(
    *,
    num_nodes: int,
    positive_forbidden_undirected: np.ndarray,
    num_samples: int,
    seed: int,
) -> np.ndarray:
    if num_samples == 0:
        return np.empty((2, 0), dtype=np.int64)
    try:
        import torch
        from torch_geometric.utils import negative_sampling
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Missing dependencies for negative sampling. Please install torch and torch_geometric."
        ) from e

    full_forbidden_bidir = _to_bidirectional(positive_forbidden_undirected)
    edge_index = torch.from_numpy(full_forbidden_bidir).to(torch.long)
    torch.manual_seed(int(seed))

    collected: set[tuple[int, int]] = set()
    max_rounds = 20
    for round_idx in range(max_rounds):
        needed = int(num_samples - len(collected))
        if needed <= 0:
            break
        request = max(needed * 2, needed)
        neg = negative_sampling(
            edge_index=edge_index,
            num_nodes=num_nodes,
            num_neg_samples=int(request),
            method="sparse",
            force_undirected=True,
        )
        neg_np = _as_undirected_unique(neg.detach().cpu().numpy().astype(np.int64, copy=False))
        for u, v in neg_np.T.tolist():
            collected.add((int(u), int(v)))
            if len(collected) >= num_samples:
                break
        if round_idx > 3 and len(collected) < num_samples and needed == (num_samples - len(collected)):
            # Sampling stagnated, likely because candidate space is small.
            break

    if len(collected) < num_samples:
        raise RuntimeError(
            f"Failed to sample enough negative edges: required={num_samples}, got={len(collected)}."
        )
    arr = np.array(sorted(collected)[:num_samples], dtype=np.int64)
    return arr.T


def _calc_degree(num_nodes: int, undirected_edges: np.ndarray) -> np.ndarray:
    deg = np.zeros(num_nodes, dtype=np.int64)
    if undirected_edges.size == 0:
        return deg
    src = undirected_edges[0]
    dst = undirected_edges[1]
    np.add.at(deg, src, 1)
    np.add.at(deg, dst, 1)
    return deg


def _validate_components_connected(
    components: list[list[int]],
    train_undirected: np.ndarray,
) -> bool:
    if train_undirected.size == 0:
        return len(components) == 0
    adj: dict[int, list[int]] = defaultdict(list)
    for u, v in train_undirected.T.tolist():
        ui = int(u)
        vi = int(v)
        adj[ui].append(vi)
        adj[vi].append(ui)
    for comp in components:
        if len(comp) <= 1:
            continue
        start = comp[0]
        seen: set[int] = {start}
        q: deque[int] = deque([start])
        comp_set = set(comp)
        while q:
            cur = q.popleft()
            for nxt in adj[cur]:
                if nxt in comp_set and nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        if len(seen) != len(comp):
            return False
    return True


def safe_link_prediction_split(
    *,
    edge_index: np.ndarray,
    num_nodes: int,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> SplitResult:
    if val_ratio < 0 or test_ratio < 0:
        raise ValueError("val_ratio and test_ratio must be non-negative")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    undirected = _as_undirected_unique(edge_index)
    all_edges = _to_edge_set(undirected)
    components, must_keep_set = _build_components_and_forest(num_nodes, undirected)

    extra_edges = sorted(all_edges - must_keep_set)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(extra_edges)

    n_extra = len(extra_edges)
    n_test = int(round(n_extra * test_ratio))
    n_val = int(round(n_extra * val_ratio))
    if n_test + n_val > n_extra:
        overflow = n_test + n_val - n_extra
        n_val = max(0, n_val - overflow)

    test_set = set(extra_edges[:n_test])
    val_set = set(extra_edges[n_test : n_test + n_val])
    train_extra_set = set(extra_edges[n_test + n_val :])
    train_set = must_keep_set | train_extra_set

    train_pos = _set_to_edge_index(train_set)
    val_pos = _set_to_edge_index(val_set)
    test_pos = _set_to_edge_index(test_set)

    train_neg = _sample_negatives_with_pyg(
        num_nodes=num_nodes,
        positive_forbidden_undirected=undirected,
        num_samples=train_pos.shape[1],
        seed=seed + 11,
    )
    val_neg = _sample_negatives_with_pyg(
        num_nodes=num_nodes,
        positive_forbidden_undirected=undirected,
        num_samples=val_pos.shape[1],
        seed=seed + 23,
    )
    test_neg = _sample_negatives_with_pyg(
        num_nodes=num_nodes,
        positive_forbidden_undirected=undirected,
        num_samples=test_pos.shape[1],
        seed=seed + 37,
    )

    # Validate split disjointness and negative safety.
    train_set_check = _to_edge_set(train_pos)
    val_set_check = _to_edge_set(val_pos)
    test_set_check = _to_edge_set(test_pos)
    if (train_set_check & val_set_check) or (train_set_check & test_set_check) or (val_set_check & test_set_check):
        raise RuntimeError("Positive edge split overlap detected.")

    all_real = _to_edge_set(undirected)
    for name, neg_edges in [("train", train_neg), ("val", val_neg), ("test", test_neg)]:
        if _to_edge_set(neg_edges) & all_real:
            raise RuntimeError(f"Negative overlap with real edges detected in {name}.")

    deg_all = _calc_degree(num_nodes, undirected)
    deg_train = _calc_degree(num_nodes, train_pos)
    active_nodes = np.where(deg_all > 0)[0]
    has_island = bool(np.any(deg_train[active_nodes] == 0)) if active_nodes.size > 0 else False
    comp_ok = _validate_components_connected(components, train_pos)
    if has_island:
        raise RuntimeError("Train graph contains isolated active nodes.")
    if not comp_ok:
        raise RuntimeError("Connected components are not preserved in train graph.")

    train_bidir = _to_bidirectional(train_pos)
    stats = {
        "seed": int(seed),
        "num_nodes": int(num_nodes),
        "full_undirected_edges": int(undirected.shape[1]),
        "must_keep_edges": int(len(must_keep_set)),
        "extra_edges": int(n_extra),
        "train_pos_undirected": int(train_pos.shape[1]),
        "val_pos_undirected": int(val_pos.shape[1]),
        "test_pos_undirected": int(test_pos.shape[1]),
        "train_neg_undirected": int(train_neg.shape[1]),
        "val_neg_undirected": int(val_neg.shape[1]),
        "test_neg_undirected": int(test_neg.shape[1]),
        "active_nodes": int(active_nodes.size),
        "deg1_nodes_full_graph": int(np.sum(deg_all == 1)),
        "deg0_nodes_train_among_active": int(np.sum(deg_train[active_nodes] == 0)) if active_nodes.size > 0 else 0,
        "component_count_full_graph": int(len(components)),
        "components_preserved_connected": bool(comp_ok),
    }

    return SplitResult(
        train_pos_undirected=train_pos,
        val_pos_undirected=val_pos,
        test_pos_undirected=test_pos,
        train_neg_undirected=train_neg,
        val_neg_undirected=val_neg,
        test_neg_undirected=test_neg,
        train_edge_index_bidirectional=train_bidir,
        stats=stats,
    )
