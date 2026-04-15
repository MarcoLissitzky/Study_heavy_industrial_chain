import argparse
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path so `scripts.*` imports work
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_edges import (  # noqa: E402
    build_node_index,
    clean_cell,
    make_id,
    normalize_name,
    read_csv_robust,
)


def compute_main_node_id(df_node: pd.DataFrame) -> pd.Series:
    def mk(r: pd.Series) -> str:
        code = clean_cell(r.get("统一社会信用代码"))
        if code is not None and not pd.isna(code):
            return make_id("corp", str(code))
        nm = normalize_name(r.get("系统匹配企业名称"))
        return make_id("name", str(nm))

    return df_node.apply(mk, axis=1)


def main() -> int:
    ap = argparse.ArgumentParser(description="Report nodes with supplier_count>20 or customer_count>20 (supplychain layer).")
    ap.add_argument("--base-dir", required=True, help="Directory containing node.csv, edge_supplychain.csv, viz/supplychain_node_metrics.csv")
    ap.add_argument("--threshold", type=int, default=20, help="Threshold for supplier/customer count")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    node_csv = base_dir / "node.csv"
    edge_supply = base_dir / "edge_supplychain.csv"
    metrics_csv = base_dir / "viz" / "supplychain_node_metrics.csv"

    nodes = read_csv_robust(node_csv)
    metrics = read_csv_robust(metrics_csv)
    edges = read_csv_robust(edge_supply)

    nodes["node_id"] = compute_main_node_id(nodes)

    # main list
    t = int(args.threshold)
    sel = metrics[(pd.to_numeric(metrics["supplier_count"], errors="coerce").fillna(0) > t) | (pd.to_numeric(metrics["customer_count"], errors="coerce").fillna(0) > t)].copy()

    # join node info
    merged = sel.merge(
        nodes[
            [
                "node_id",
                "系统匹配企业名称",
                "统一社会信用代码",
                "注册资本",
                "所属省份",
                "所属城市",
                "所属区县",
                "国标行业大类",
                "国标行业中类",
                "国标行业小类",
                "企业规模",
                "企业机构类型大类",
            ]
        ],
        on="node_id",
        how="left",
        suffixes=("", "_node"),
    )

    # build partner lists from edge_supplychain (unique dst per type)
    sup = edges[edges["edge_type"] == "supplier"].copy()
    cus = edges[edges["edge_type"] == "customer"].copy()

    sup_list = sup.groupby("src_id")["dst_name"].apply(lambda s: sorted({str(x) for x in s if isinstance(x, str) and x})).to_dict()
    cus_list = cus.groupby("src_id")["dst_name"].apply(lambda s: sorted({str(x) for x in s if isinstance(x, str) and x})).to_dict()

    def join_list(lst: list[str], limit: int = 200) -> str:
        if not lst:
            return ""
        out = " | ".join(lst)
        return out[:limit] + ("…" if len(out) > limit else "")

    merged["供应商名单_截断"] = merged["node_id"].map(lambda nid: join_list(sup_list.get(nid, [])))
    merged["客户名单_截断"] = merged["node_id"].map(lambda nid: join_list(cus_list.get(nid, [])))

    out_dir = base_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / f"supplychain_nodes_gt{t}.csv"
    merged.sort_values(by=["supplier_count", "customer_count"], ascending=False).to_csv(summary_path, index=False, encoding="utf-8-sig")

    # detail table: one row per partner
    focus_ids = set(merged["node_id"].dropna().astype(str).tolist())
    detail = edges[edges["src_id"].isin(focus_ids)][
        [
            "src_id",
            "src_name",
            "edge_type",
            "dst_id",
            "dst_name",
            "report_date_raw",
            "source",
            "relation_hint",
            "source_file",
        ]
    ].copy()
    detail_path = out_dir / f"supplychain_partners_gt{t}_detail.csv"
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")

    print(f"Wrote: {summary_path}")
    print(f"Wrote: {detail_path}")
    print(f"nodes_count={len(merged)} threshold={t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

