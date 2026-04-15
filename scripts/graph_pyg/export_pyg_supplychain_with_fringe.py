import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_edges import clean_cell, make_id, normalize_name, read_csv_robust
from scripts.graph_pyg.split_link_prediction_safe import safe_link_prediction_split

DEFAULT_BASE_DIR = PROJECT_ROOT / "data" / "processed" / "supplychain_pps5000"
DEFAULT_OUT_DIR = DEFAULT_BASE_DIR / "pyg"
DEFAULT_REGISTRY_DIR = PROJECT_ROOT / "data" / "raw" / "registry_cn_parquet"

CATEGORICAL_COLS = ["所属省份", "所属城市", "国标行业大类", "企业机构类型大类"]
NUMERIC_COLS = ["注册资本"]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS
UNKNOWN_TOKEN = "Unknown"

REGISTRY_ALIASES = {
    "统一社会信用代码": ["统一社会信用代码"],
    "系统匹配企业名称": ["系统匹配企业名称", "企业名称"],
    "注册资本": ["注册资本"],
    "所属省份": ["所属省份"],
    "所属城市": ["所属城市"],
    "所属区县": ["所属区县"],
    "国标行业大类": ["国标行业大类", "所属行业"],
    "企业机构类型大类": ["企业机构类型大类", "企业类型"],
}


def log_step(msg: str) -> None:
    print(f"[STEP] {msg}", flush=True)


def parse_capital_to_number(v: object) -> float:
    if v is None or pd.isna(v):
        return 0.0
    s = str(v).strip()
    if not s:
        return 0.0
    s = s.replace(",", "").replace("，", "")
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", s)
    if not m:
        return 0.0
    base = float(m.group(1))
    if "亿" in s:
        return base * 1e8
    if "万" in s:
        return base * 1e4
    return base


def normalize_for_match(v: object) -> str | None:
    n = normalize_name(v)
    if n is None or pd.isna(n):
        return None
    return str(n)


def build_node_id_from_row(row: pd.Series) -> str | None:
    code = clean_cell(row.get("统一社会信用代码"))
    if code is not None and not pd.isna(code):
        return make_id("corp", str(code))
    name = normalize_name(row.get("系统匹配企业名称"))
    if name is not None and not pd.isna(name):
        return make_id("name", str(name))
    return None


def extract_credit_code(node_id: str) -> str | None:
    if isinstance(node_id, str) and node_id.startswith("corp:"):
        code = node_id.split(":", 1)[1].strip()
        return code or None
    return None


def collect_registry_parquet_files(registry_dir: Path) -> list[Path]:
    files = sorted(registry_dir.glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under: {registry_dir}")
    return files


def try_query_registry_with_duckdb(
    registry_dir: Path,
    *,
    target_codes: set[str] | None = None,
    target_names: set[str] | None = None,
) -> pd.DataFrame | None:
    target_codes = target_codes or set()
    target_names = target_names or set()
    if not target_codes and not target_names:
        return pd.DataFrame(columns=["统一社会信用代码", "系统匹配企业名称", *FEATURE_COLS])
    try:
        import duckdb
    except Exception:  # noqa: BLE001
        return None

    def q(v: str) -> str:
        return "'" + v.replace("'", "''") + "'"

    con = duckdb.connect(database=":memory:")
    try:
        parquet_glob = str((registry_dir / "**" / "*.parquet").as_posix())
        where_parts: list[str] = []
        if target_codes:
            codes_sql = ", ".join([q(c) for c in sorted(target_codes)])
            where_parts.append(f"统一社会信用代码 IN ({codes_sql})")
        if target_names:
            names_sql = ", ".join([q(n) for n in sorted(target_names)])
            where_parts.append(f"企业名称 IN ({names_sql})")
        where_sql = " OR ".join(where_parts) if where_parts else "FALSE"
        sql = f"""
            SELECT
                统一社会信用代码,
                企业名称 AS 系统匹配企业名称,
                注册资本,
                所属省份,
                所属城市,
                所属区县,
                所属行业 AS 国标行业大类,
                企业类型 AS 企业机构类型大类
            FROM read_parquet('{parquet_glob}')
            WHERE {where_sql}
        """
        log_step("duckdb query registry parquet (predicate pushdown)")
        return con.execute(sql).df()
    finally:
        con.close()


def unify_registry_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for target, aliases in REGISTRY_ALIASES.items():
        if target in out.columns:
            continue
        src = next((a for a in aliases if a in out.columns), None)
        out[target] = out[src] if src is not None else pd.NA
    keep = ["统一社会信用代码", "系统匹配企业名称", *FEATURE_COLS]
    for c in keep:
        if c not in out.columns:
            out[c] = pd.NA
    return out[keep].copy()


def read_registry_selected_cols(
    registry_dir: Path,
    files: list[Path],
    *,
    target_codes: set[str] | None = None,
    target_names: set[str] | None = None,
) -> pd.DataFrame:
    duckdb_out = try_query_registry_with_duckdb(registry_dir, target_codes=target_codes, target_names=target_names)
    if duckdb_out is not None:
        return unify_registry_columns(duckdb_out)

    needed_aliases: list[str] = sorted({c for vals in REGISTRY_ALIASES.values() for c in vals})
    chunks: list[pd.DataFrame] = []
    target_codes = target_codes or set()
    target_names = target_names or set()
    has_filters = bool(target_codes or target_names)
    for i, p in enumerate(files, start=1):
        log_step(f"registry load {i}/{len(files)}: {p.name}")
        try:
            df = pd.read_parquet(p, columns=needed_aliases)
        except Exception:
            df = pd.read_parquet(p)
            keep = [c for c in needed_aliases if c in df.columns]
            df = df[keep]
        df = unify_registry_columns(df)
        if has_filters:
            code_col = df["统一社会信用代码"].map(clean_cell).map(lambda x: str(x) if pd.notna(x) else None)
            name_col = df["系统匹配企业名称"].map(normalize_for_match)
            mask = pd.Series(False, index=df.index)
            if target_codes:
                mask = mask | code_col.isin(target_codes)
            if target_names:
                mask = mask | name_col.isin(target_names)
            df = df[mask].copy()
        if not df.empty:
            chunks.append(df)
    needed = ["统一社会信用代码", "系统匹配企业名称", *FEATURE_COLS]
    out = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=needed)
    for c in needed:
        if c not in out.columns:
            out[c] = pd.NA
    return out[needed].copy()


def build_active_nodes(node_df: pd.DataFrame, fringe_df: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    active_ids = pd.Index(pd.unique(pd.concat([edges["src_id"], edges["dst_id"]], ignore_index=True)))
    active_ids = active_ids[active_ids.notna()]
    active_set = set(active_ids.astype(str).tolist())

    node_work = node_df.copy()
    node_work["node_id"] = node_work.apply(build_node_id_from_row, axis=1)
    node_work = node_work[node_work["node_id"].notna()].copy()
    node_work["node_id"] = node_work["node_id"].astype(str)
    node_work = node_work[node_work["node_id"].isin(active_set)].copy()
    node_work["node_origin"] = "node"

    required_base = ["node_id", "系统匹配企业名称", "统一社会信用代码", *FEATURE_COLS, "node_origin"]
    for c in required_base:
        if c not in node_work.columns:
            node_work[c] = pd.NA
    node_base = node_work[required_base].copy()

    fringe_work = fringe_df.copy()
    if "node_id" not in fringe_work.columns:
        raise ValueError("node_fringe.csv missing required column: node_id")
    fringe_work["node_id"] = fringe_work["node_id"].astype(str)
    fringe_work = fringe_work[fringe_work["node_id"].isin(active_set)].copy()
    fringe_work = fringe_work[~fringe_work["node_id"].isin(set(node_base["node_id"].astype(str)))].copy()
    fringe_work["系统匹配企业名称"] = fringe_work.get("node_name", pd.Series([pd.NA] * len(fringe_work))).map(clean_cell)
    fringe_work["统一社会信用代码"] = fringe_work["node_id"].map(extract_credit_code)
    fringe_work["node_origin"] = "fringe"
    for c in FEATURE_COLS:
        fringe_work[c] = pd.NA
    fringe_base = fringe_work[required_base].copy()

    out = pd.concat([node_base, fringe_base], ignore_index=True)
    out["norm_name"] = out["系统匹配企业名称"].map(normalize_for_match)
    out["credit_code"] = out["统一社会信用代码"].map(
        lambda x: str(clean_cell(x)) if clean_cell(x) is not pd.NA and not pd.isna(clean_cell(x)) else None
    )
    return out


def encode_features(df: pd.DataFrame) -> tuple[np.ndarray, dict[str, object], pd.DataFrame]:
    work = df.copy()
    work["注册资本_num"] = work["注册资本"].map(parse_capital_to_number).astype(np.float32)
    cat_parts: list[pd.DataFrame] = []
    cat_feature_names: list[str] = []
    category_sizes: dict[str, int] = {}
    for c in CATEGORICAL_COLS:
        s = work[c].map(clean_cell)
        s = s.map(lambda v: UNKNOWN_TOKEN if v is None or pd.isna(v) else str(v))
        onehot = pd.get_dummies(s, prefix=c, dtype=np.float32)
        category_sizes[c] = int(onehot.shape[1])
        cat_parts.append(onehot)
        cat_feature_names.extend(onehot.columns.tolist())

    x_num = work[["注册资本_num"]].to_numpy(dtype=np.float32)
    x_cat = pd.concat(cat_parts, axis=1).to_numpy(dtype=np.float32) if cat_parts else np.zeros((len(work), 0), dtype=np.float32)
    x = np.concatenate([x_num, x_cat], axis=1).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    feature_names = ["注册资本_raw"] + cat_feature_names
    meta = {
        "numeric_features": ["注册资本_raw"],
        "categorical_features": CATEGORICAL_COLS,
        "categorical_onehot_sizes": category_sizes,
        "feature_names": feature_names,
        "unknown_token": UNKNOWN_TOKEN,
        "dtype": "float32",
    }
    return x, meta, work


def build_edges(edges_df: pd.DataFrame, id_to_idx: dict[str, int]) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    work = edges_df.copy()
    work = work[work["layer"] == "supplychain"].copy() if "layer" in work.columns else work
    work = work[work["src_id"].notna() & work["dst_id"].notna()].copy()
    work["src_id"] = work["src_id"].astype(str)
    work["dst_id"] = work["dst_id"].astype(str)
    work = work[work["src_id"].isin(id_to_idx) & work["dst_id"].isin(id_to_idx)].copy()
    work["src_idx"] = work["src_id"].map(id_to_idx)
    work["dst_idx"] = work["dst_id"].map(id_to_idx)

    before = len(work)
    work = work[work["src_idx"] != work["dst_idx"]].copy()
    self_loop_removed = before - len(work)
    work = work.drop_duplicates(subset=["src_idx", "dst_idx"], keep="first").copy()
    directed_unique = work[["src_idx", "dst_idx"]].to_numpy(dtype=np.int64).T
    dedup_forward_rows = len(work)
    rev = work.rename(columns={"src_idx": "dst_idx", "dst_idx": "src_idx"})
    merged = pd.concat([work[["src_idx", "dst_idx"]], rev[["src_idx", "dst_idx"]]], ignore_index=True)
    merged = merged.drop_duplicates(subset=["src_idx", "dst_idx"], keep="first").copy()
    edge_index = merged[["src_idx", "dst_idx"]].to_numpy(dtype=np.int64).T

    stats = {
        "input_edge_rows": int(before),
        "self_loop_removed": int(self_loop_removed),
        "dedup_forward_rows": int(dedup_forward_rows),
        "output_edge_rows_bidirectional": int(merged.shape[0]),
    }
    return edge_index, directed_unique, stats


def build_train_pos_directed(
    train_pos_undirected: np.ndarray,
    directed_edge_index: np.ndarray,
) -> np.ndarray:
    if train_pos_undirected.size == 0 or directed_edge_index.size == 0:
        return np.empty((2, 0), dtype=np.int64)
    train_set = {(int(u), int(v)) for u, v in train_pos_undirected.T.tolist()}
    keep: list[tuple[int, int]] = []
    for u, v in directed_edge_index.T.tolist():
        a, b = (int(u), int(v))
        lo, hi = (a, b) if a <= b else (b, a)
        if (lo, hi) in train_set:
            keep.append((a, b))
    if not keep:
        return np.empty((2, 0), dtype=np.int64)
    return np.array(keep, dtype=np.int64).T


def save_checkpoint(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> int:
    ap = argparse.ArgumentParser(description="Export supplychain single-graph PyG Data with safe link prediction split.")
    ap.add_argument("--base-dir", default=str(DEFAULT_BASE_DIR))
    ap.add_argument("--registry-dir", default=str(DEFAULT_REGISTRY_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--resume-from", default="none", choices=["none", "active_nodes", "code_match", "name_match", "prejoin_done"])
    ap.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio over extra edges.")
    ap.add_argument("--test-ratio", type=float, default=0.1, help="Test ratio over extra edges.")
    ap.add_argument("--split-seed", type=int, default=42, help="Random seed for safe split and negative sampling.")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    registry_dir = Path(args.registry_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    node_path = base_dir / "node.csv"
    fringe_path = base_dir / "node_fringe.csv"
    edge_path = base_dir / "edge_supplychain.csv"
    for p in (node_path, fringe_path, edge_path):
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    ck_active = out_dir / "active_nodes_base.csv"
    ck_code = out_dir / "registry_code_match.csv"
    ck_name = out_dir / "registry_name_match.csv"
    ck_prejoin = out_dir / "node_feature_prejoin.csv"
    node_map_path = out_dir / "node_id_map.csv"
    pt_path = out_dir / "supplychain_data.pt"
    split_path = out_dir / "split_edges.pt"
    stats_path = out_dir / "pyg_build_stats.json"
    split_stats_path = out_dir / "split_stats.json"

    node_df = read_csv_robust(node_path)
    fringe_df = read_csv_robust(fringe_path)
    edges_df = read_csv_robust(edge_path)

    if args.resume_from in {"active_nodes", "code_match", "name_match", "prejoin_done"} and ck_active.exists():
        log_step("resume active nodes from checkpoint")
        active_nodes = read_csv_robust(ck_active)
    else:
        log_step("1/5 build active nodes base")
        active_nodes = build_active_nodes(node_df, fringe_df, edges_df)
        save_checkpoint(active_nodes, ck_active)

    if args.resume_from in {"code_match", "name_match", "prejoin_done"} and ck_code.exists():
        log_step("resume code-match checkpoint")
        code_df = read_csv_robust(ck_code)
        reg_df_for_name: pd.DataFrame | None = None
    else:
        log_step("2/5 registry code match")
        target_codes = {str(x) for x in active_nodes["credit_code"].dropna().astype(str).tolist() if str(x).strip()}
        target_names_all = {str(x) for x in active_nodes["norm_name"].dropna().astype(str).tolist() if str(x).strip()}
        files = collect_registry_parquet_files(registry_dir)
        reg_df = read_registry_selected_cols(registry_dir, files, target_codes=target_codes, target_names=target_names_all)
        reg_df["credit_code"] = reg_df["统一社会信用代码"].map(
            lambda x: str(clean_cell(x)) if clean_cell(x) is not pd.NA and not pd.isna(clean_cell(x)) else None
        )
        reg_code = reg_df[reg_df["credit_code"].notna()].drop_duplicates(subset=["credit_code"], keep="first")
        code_df = active_nodes.merge(reg_code[["credit_code", *FEATURE_COLS]], on="credit_code", how="left", suffixes=("", "_regcode"))
        reg_df_for_name = reg_df
        save_checkpoint(code_df, ck_code)

    if args.resume_from in {"name_match", "prejoin_done"} and ck_name.exists():
        log_step("resume name-match checkpoint")
        name_df = read_csv_robust(ck_name)
    else:
        log_step("3/5 registry name match")
        pending = code_df.copy()
        for c in FEATURE_COLS:
            if f"{c}_regcode" in pending.columns:
                pending.rename(columns={f"{c}_regcode": f"reg_code_{c}"}, inplace=True)
            elif c in pending.columns and f"reg_code_{c}" not in pending.columns:
                pending[f"reg_code_{c}"] = pending[c]
                pending.drop(columns=[c], inplace=True)
        missing_mask = pd.Series(False, index=pending.index)
        for c in FEATURE_COLS:
            missing_mask = missing_mask | pending[f"reg_code_{c}"].isna()
        need_name = pending[missing_mask].copy()
        target_names = {str(x) for x in need_name["norm_name"].dropna().astype(str).tolist() if str(x).strip()}
        if reg_df_for_name is not None:
            reg_df = reg_df_for_name.copy()
            if target_names:
                reg_df = reg_df[reg_df["系统匹配企业名称"].map(normalize_for_match).isin(target_names)].copy()
        else:
            files = collect_registry_parquet_files(registry_dir)
            reg_df = read_registry_selected_cols(registry_dir, files, target_names=target_names)
        reg_df["norm_name"] = reg_df["系统匹配企业名称"].map(normalize_for_match)
        reg_name = reg_df[reg_df["norm_name"].notna()].drop_duplicates(subset=["norm_name"], keep="first")
        name_map = need_name.merge(reg_name[["norm_name", *FEATURE_COLS]], on="norm_name", how="left", suffixes=("", "_regname"))
        name_df = pending.merge(name_map[["node_id", *[f"{c}_regname" for c in FEATURE_COLS]]], on="node_id", how="left")
        save_checkpoint(name_df, ck_name)

    if args.resume_from == "prejoin_done" and ck_prejoin.exists():
        log_step("resume prejoin checkpoint")
        prejoin = read_csv_robust(ck_prejoin)
    else:
        log_step("4/5 feature merge prejoin")
        rows = []
        for r in name_df.itertuples(index=False):
            rec = {
                "node_id": getattr(r, "node_id"),
                "系统匹配企业名称": getattr(r, "系统匹配企业名称", pd.NA),
                "统一社会信用代码": getattr(r, "统一社会信用代码", pd.NA),
                "node_origin": getattr(r, "node_origin", pd.NA),
                "credit_code": getattr(r, "credit_code", pd.NA),
                "norm_name": getattr(r, "norm_name", pd.NA),
            }
            has_code = False
            has_name = False
            for c in FEATURE_COLS:
                code_v = getattr(r, f"reg_code_{c}", pd.NA)
                name_v = getattr(r, f"{c}_regname", pd.NA)
                node_v = getattr(r, c, pd.NA)
                if pd.notna(code_v):
                    has_code = True
                if pd.notna(name_v):
                    has_name = True
                rec[c] = node_v if rec["node_origin"] == "node" and pd.notna(node_v) else (code_v if pd.notna(code_v) else name_v)
            if rec["node_origin"] == "node":
                rec["match_source"] = "node_primary+registry_code" if has_code else ("node_primary+registry_name" if has_name else "node_primary_only")
            else:
                rec["match_source"] = "registry_code" if has_code else ("registry_name" if has_name else "unmatched")
            rows.append(rec)
        prejoin = pd.DataFrame(rows)
        save_checkpoint(prejoin, ck_prejoin)

    prejoin = prejoin.drop_duplicates(subset=["node_id"], keep="first").copy()
    prejoin["node_id"] = prejoin["node_id"].astype(str)
    prejoin = prejoin.sort_values(by=["node_id"]).reset_index(drop=True)
    id_to_idx = {nid: i for i, nid in enumerate(prejoin["node_id"].tolist())}
    pd.DataFrame({"node_id": list(id_to_idx.keys()), "node_idx": list(id_to_idx.values())}).to_csv(
        node_map_path, index=False, encoding="utf-8-sig"
    )

    x, feature_meta, feature_work = encode_features(prejoin)
    edge_index, directed_edge_index, edge_stats = build_edges(edges_df, id_to_idx)
    split = safe_link_prediction_split(
        edge_index=edge_index,
        num_nodes=int(x.shape[0]),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        seed=int(args.split_seed),
    )

    try:
        import torch
        from torch_geometric.data import Data
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Missing required dependencies for PyG export. Please install torch and torch_geometric first.") from e

    data = Data(
        x=torch.from_numpy(x).to(torch.float32),
        edge_index=torch.from_numpy(split.train_edge_index_bidirectional).to(torch.long),
        num_nodes=int(x.shape[0]),
    )
    data.node_id = prejoin["node_id"].tolist()
    data.feature_names = feature_meta["feature_names"]
    data.edge_relation = "supplychain_link"
    data.split_seed = int(args.split_seed)
    torch.save(data, pt_path)

    train_pos_directed = build_train_pos_directed(split.train_pos_undirected, directed_edge_index)
    split_payload = {
        "train_pos_edge_index_undirected": torch.from_numpy(split.train_pos_undirected).to(torch.long),
        "train_pos_edge_index_directed": torch.from_numpy(train_pos_directed).to(torch.long),
        "val_pos_edge_index_undirected": torch.from_numpy(split.val_pos_undirected).to(torch.long),
        "test_pos_edge_index_undirected": torch.from_numpy(split.test_pos_undirected).to(torch.long),
        "train_neg_edge_index_undirected": torch.from_numpy(split.train_neg_undirected).to(torch.long),
        "val_neg_edge_index_undirected": torch.from_numpy(split.val_neg_undirected).to(torch.long),
        "test_neg_edge_index_undirected": torch.from_numpy(split.test_neg_undirected).to(torch.long),
        "train_message_passing_edge_index": torch.from_numpy(split.train_edge_index_bidirectional).to(torch.long),
        "num_nodes": int(x.shape[0]),
        "num_features_raw": int(x.shape[1]),
        "seed": int(args.split_seed),
        "val_ratio": float(args.val_ratio),
        "test_ratio": float(args.test_ratio),
    }
    torch.save(split_payload, split_path)

    fringe_mask = prejoin["node_origin"].astype(str).eq("fringe")
    fringe_all = prejoin[fringe_mask].copy()
    fringe_count = int(len(fringe_all))
    if fringe_count > 0:
        fringe_missing_matrix = fringe_all[FEATURE_COLS].isna()
        fringe_missing_rate = float(fringe_missing_matrix.mean().mean())
        fringe_cap_zero_ratio = float((feature_work.loc[fringe_mask, "注册资本_num"] == 0.0).mean())
    else:
        fringe_missing_rate = 0.0
        fringe_cap_zero_ratio = 0.0

    quality_alert = fringe_missing_rate > 0.40
    stats = {
        "num_nodes": int(x.shape[0]),
        "num_edges_train_message_passing": int(split.train_edge_index_bidirectional.shape[1]),
        "num_edges_train_pos_directed": int(train_pos_directed.shape[1]),
        "num_features": int(x.shape[1]),
        "feature_meta": feature_meta,
        "edge_stats": edge_stats,
        "split_stats": split.stats,
        "match_source_counts": prejoin["match_source"].value_counts(dropna=False).to_dict(),
        "fringe_nodes": fringe_count,
        "fringe_feature_missing_rate": fringe_missing_rate,
        "fringe_registered_capital_zero_ratio": fringe_cap_zero_ratio,
        "quality_alert_fringe_missing_gt_40pct": quality_alert,
        "no_nan_in_x": bool(np.isfinite(x).all()),
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    split_stats_path.write_text(json.dumps(split.stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote: {node_map_path}")
    print(f"Wrote: {ck_prejoin}")
    print(f"Wrote: {pt_path}")
    print(f"Wrote: {split_path}")
    print(f"Wrote: {stats_path}")
    print(f"Wrote: {split_stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
