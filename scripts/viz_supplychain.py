import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from adjustText import adjust_text

# Ensure project root is on sys.path so `scripts.*` imports work
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_edges import (
    build_node_index,
    clean_cell,
    make_id,
    normalize_name,
    read_csv_robust,
)

DEFAULT_VIZ_DIR = PROJECT_ROOT / "outputs" / "viz" / "supplychain_pps5000"


def configure_matplotlib_chinese_font() -> str | None:
    """
    Try to configure a commonly available CJK font so PNGs render Chinese labels.
    Returns the chosen font name if found.
    """
    try:
        from matplotlib import font_manager

        candidates = [
            "Microsoft YaHei",
            "Microsoft YaHei UI",
            "SimHei",
            "SimSun",
            "NSimSun",
            "KaiTi",
            "FangSong",
            "Noto Sans CJK SC",
            "Noto Sans SC",
            "Source Han Sans SC",
        ]
        available = {f.name for f in font_manager.fontManager.ttflist}
        chosen = next((c for c in candidates if c in available), None)
        if chosen:
            plt.rcParams["font.sans-serif"] = [chosen]
            plt.rcParams["axes.unicode_minus"] = False
        return chosen
    except Exception:  # noqa: BLE001
        return None


def parse_capital_to_wan(v: object) -> float | None:
    """Parse 注册资本 to 万人民币. Best-effort for values like '635600万人民币', '2.68亿', '16072.5万'."""
    v = clean_cell(v)
    if v is None or pd.isna(v):
        return None
    s = str(v).replace(",", "").replace("人民币", "")
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(亿|万)?", s)
    if not m:
        return None
    num = float(m.group(1))
    unit = m.group(2) or ""
    if unit == "亿":
        return num * 10000.0
    if unit == "万" or unit == "":
        return num
    return None


def compute_supplychain_metrics(node_csv: Path, edges_supply: pd.DataFrame) -> pd.DataFrame:
    nodes = read_csv_robust(node_csv)
    nodes["node_id"] = nodes.apply(
        lambda r: make_id("corp", str(clean_cell(r.get("统一社会信用代码"))))
        if clean_cell(r.get("统一社会信用代码")) is not None and not pd.isna(clean_cell(r.get("统一社会信用代码")))
        else make_id("name", str(normalize_name(r.get("系统匹配企业名称")))),
        axis=1,
    )
    nodes["注册资本_万元"] = nodes["注册资本"].map(parse_capital_to_wan) if "注册资本" in nodes.columns else None

    sup = edges_supply[edges_supply["edge_type"] == "supplier"]
    cus = edges_supply[edges_supply["edge_type"] == "customer"]

    sup_map = sup.groupby("src_id")["dst_id"].apply(lambda s: set(s)).to_dict()
    cus_map = cus.groupby("src_id")["dst_id"].apply(lambda s: set(s)).to_dict()

    metrics = []
    for r in nodes.itertuples(index=False):
        node_id = getattr(r, "node_id")
        sup_set = sup_map.get(node_id, set())
        cus_set = cus_map.get(node_id, set())
        supplier_count = len(sup_set)
        customer_count = len(cus_set)
        degree = len(sup_set | cus_set)
        geom = math.sqrt(supplier_count * customer_count) if supplier_count > 0 and customer_count > 0 else 0.0
        metrics.append(
            {
                "node_id": node_id,
                "系统匹配企业名称": getattr(r, "系统匹配企业名称"),
                "统一社会信用代码": getattr(r, "统一社会信用代码") if "统一社会信用代码" in nodes.columns else "",
                "注册资本": getattr(r, "注册资本") if "注册资本" in nodes.columns else "",
                "注册资本_万元": getattr(r, "注册资本_万元"),
                "企业机构类型大类": getattr(r, "企业机构类型大类") if "企业机构类型大类" in nodes.columns else "",
                "degree": degree,
                "supplier_count": supplier_count,
                "customer_count": customer_count,
                "geom_mean_supplier_customer": geom,
            }
        )
    return pd.DataFrame(metrics)


def summarize_series(s: pd.Series) -> dict[str, float]:
    s2 = pd.to_numeric(s, errors="coerce").fillna(0)
    return {"min": float(s2.min()), "max": float(s2.max()), "median": float(s2.median()), "mean": float(s2.mean())}


def choose_integer_ticks(y_max: float) -> list[int]:
    """
    Choose a set of integer ticks for counts, up to y_max.
    """
    base = [0, 1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    ymax = int(math.ceil(float(y_max)))
    ticks = [t for t in base if t <= ymax]
    if not ticks:
        return [0, 1]
    if ticks[-1] != ymax and ymax > ticks[-1]:
        # add the max to help readability
        ticks.append(ymax)
    return sorted(set(ticks))


def set_log1p_axis_with_integer_labels(ax: plt.Axes, y_max: float) -> None:
    ticks = choose_integer_ticks(y_max)
    ax.set_yticks(np.log1p(ticks))
    ax.set_yticklabels([str(int(t)) for t in ticks])
    ax.set_ylabel(ax.get_ylabel() + " (log1p scale)")


def plot_box_by_bins_log1p(df: pd.DataFrame, x: str, y: str, bins: int, out_path: Path) -> pd.DataFrame:
    d = df[[x, y]].copy()
    d = d[pd.notna(d[x])].copy()
    d["bin"] = pd.qcut(d[x], q=bins, duplicates="drop")
    # Use numeric capital ranges for x labels (units: 万人民币)
    cats = d["bin"].cat.categories

    def fmt_wan(v: float) -> str:
        # show as integer when possible
        if abs(v - round(v)) < 1e-9:
            return f"{int(round(v)):,}"
        s = f"{v:.2f}".rstrip("0").rstrip(".")
        # add grouping for integer part
        if "." in s:
            a, b = s.split(".", 1)
            a = f"{int(a):,}"
            return f"{a}.{b}"
        return f"{int(float(s)):,}"

    def fmt_interval(iv) -> str:
        left = float(iv.left)
        right = float(iv.right)
        # qcut may produce slight negatives due to float eps; clamp
        left = max(0.0, left)
        return f"{fmt_wan(left)}–{fmt_wan(right)}万"

    label_map = {interval: fmt_interval(interval) for interval in cats}
    d["bin_label"] = d["bin"].map(label_map).astype(str)

    # log1p transform for plotting
    y_raw = pd.to_numeric(d[y], errors="coerce").fillna(0)
    d["_y_raw"] = y_raw
    d["_y_plot"] = np.log1p(y_raw)

    # Keep bin order as produced by qcut (numeric order)
    groups = []
    labels = []
    for iv in cats:
        g = d[d["bin"] == iv]
        groups.append(g["_y_plot"].to_numpy())
        labels.append(label_map[iv])

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.0), 6))
    bp = ax.boxplot(
        groups,
        tick_labels=labels,
        vert=True,
        showfliers=True,
        patch_artist=True,
        boxprops=dict(facecolor="#4C78A8", alpha=0.25, edgecolor="#4C78A8", linewidth=1.0),
        medianprops=dict(color="#222222", linewidth=1.6),
        whiskerprops=dict(color="#4C78A8", linewidth=1.0, alpha=0.9),
        capprops=dict(color="#4C78A8", linewidth=1.0, alpha=0.9),
        flierprops=dict(marker="o", markersize=2.5, markerfacecolor="#4C78A8", markeredgecolor="none", alpha=0.18),
    )
    ax.set_xlabel(f"{x} (quantile bins)")
    ax.set_ylabel(y)
    # slightly smaller x tick labels for long capital ranges
    ax.tick_params(axis="x", rotation=0, labelsize=8)
    set_log1p_axis_with_integer_labels(ax, y_max=float(pd.to_numeric(df[y], errors="coerce").fillna(0).max()))
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)

    # annotate n per bin
    counts = [int((d["bin"] == iv).sum()) for iv in cats]
    zero_counts = [int(((d["bin"] == iv) & (d["_y_raw"] == 0)).sum()) for iv in cats]
    y_top = max([np.nanmax(g) if len(g) else 0 for g in groups] + [0])
    for i, n in enumerate(counts, start=1):
        ax.text(i, y_top + 0.05, f"n={n}", ha="center", va="bottom", fontsize=8, color="#666666")
        # put zero count at the bottom of the plot area (below axis)
        ax.text(
            i,
            -0.12,
            f"0值={zero_counts[i-1]}",
            ha="center",
            va="top",
            fontsize=8,
            color="#666666",
            transform=ax.get_xaxis_transform(),
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    # Return a stable, numerically ordered bin counts table
    out = pd.DataFrame(
        {
            "bin_left_wan": [max(0.0, float(iv.left)) for iv in cats],
            "bin_right_wan": [float(iv.right) for iv in cats],
            "bin_label": [label_map[iv] for iv in cats],
            "n": counts,
        }
    )
    return out


def plot_network_supplychain(
    edges_supply: pd.DataFrame,
    metrics: pd.DataFrame,
    node_index: Any,
    out_path: Path,
    *,
    top_n: int,
    label_top: int,
    max_edges: int | None = None,
) -> None:
    def shorten_label(s: str, max_len: int = 10) -> str:
        t = str(s).strip()
        # common suffixes to remove for readability
        for suf in ("股份有限公司", "有限责任公司", "有限公司", "集团有限公司", "集团股份有限公司"):
            if t.endswith(suf) and len(t) > len(suf) + 2:
                t = t[: -len(suf)]
                break
        if len(t) > max_len:
            return t[: max_len - 1] + "…"
        return t

    top = metrics.sort_values(by="degree", ascending=False).head(top_n)["node_id"].tolist()
    top_set = set(top)
    label_set = set(metrics.sort_values(by="degree", ascending=False).head(label_top)["node_id"].tolist())

    # Aggregate-neighbor visualization:
    # - Layout only top_n main nodes (fast, readable)
    # - Show ALL neighbor nodes as local "clouds" around their most-associated top node
    e = edges_supply[["src_id", "dst_id", "edge_type"]].copy()
    e_top = e[(e["src_id"].isin(top_set)) | (e["dst_id"].isin(top_set))].copy()
    neighbor_ids = set(e_top["src_id"]).union(set(e_top["dst_id"])) - top_set

    # Build top-only graph for layout (include top->top edges)
    df_top_top = e[e["src_id"].isin(top_set) & e["dst_id"].isin(top_set)].copy()

    G = nx.DiGraph()
    for r in df_top_top.itertuples(index=False):
        G.add_edge(r.src_id, r.dst_id, edge_type=r.edge_type)

    deg_map = metrics.set_index("node_id")["degree"].to_dict()
    type_map = metrics.set_index("node_id")["企业机构类型大类"].to_dict()

    # Dashboard-friendly palette
    palette = {
        "国有企业": "#4C78A8",
        "外资及港澳台企业": "#F58518",
        "民营企业": "#54A24B",
        "其他企业": "#B279A2",
        "未知": "#9D9D9D",
    }
    edge_palette = {"supplier": "#4C78A8", "customer": "#E45756"}

    sizes = []
    colors = []
    labels = {}
    for n in G.nodes():
        d = deg_map.get(n, 0)
        # sqrt scaling prevents huge nodes dominating
        sizes.append(40 + min(520, (math.sqrt(max(d, 0)) * 55)))
        t = type_map.get(n) or "未知"
        colors.append(palette.get(t, palette["未知"]))
        raw_name = node_index.id_to_name.get(n, n.split(":", 1)[-1])
        labels[n] = shorten_label(raw_name, max_len=10)

    # Layout only top nodes (fast)
    UG = G.to_undirected()
    pos = nx.kamada_kawai_layout(UG, scale=4.0)

    fig, ax = plt.subplots(figsize=(18, 12))
    ax.set_facecolor("#FFFFFF")

    edge_colors = [edge_palette.get(G.edges[e].get("edge_type"), "#BBBBBB") for e in G.edges()]
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        alpha=0.22,
        width=1.0,
        edge_color=edge_colors,
        arrows=False,
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=sizes,
        node_color=colors,
        alpha=0.92,
        linewidths=0.8,
        edgecolors="#FFFFFF",
    )

    # Place neighbor nodes as local clouds around their most-associated top node
    # Compute association: count edges between top and neighbor (both directions)
    assoc: dict[str, dict[str, int]] = {}
    for r in e_top.itertuples(index=False):
        a, b = r.src_id, r.dst_id
        if a in top_set and b in neighbor_ids:
            assoc.setdefault(b, {}).setdefault(a, 0)
            assoc[b][a] += 1
        elif b in top_set and a in neighbor_ids:
            assoc.setdefault(a, {}).setdefault(b, 0)
            assoc[a][b] += 1

    rng = np.random.default_rng(42)
    neighbor_x = []
    neighbor_y = []
    for nb in neighbor_ids:
        tops = assoc.get(nb)
        if not tops:
            # if a neighbor has no recorded association (should be rare), scatter near origin
            cx, cy = 0.0, 0.0
        else:
            # choose the top node with highest association count
            top_choice = max(tops.items(), key=lambda kv: kv[1])[0]
            cx, cy = pos.get(top_choice, (0.0, 0.0))
        # jitter radius based on log of neighbor count for stability
        angle = rng.uniform(0, 2 * math.pi)
        radius = rng.uniform(0.08, 0.35)
        neighbor_x.append(cx + math.cos(angle) * radius)
        neighbor_y.append(cy + math.sin(angle) * radius)

    if neighbor_x:
        ax.scatter(
            neighbor_x,
            neighbor_y,
            s=6,
            c="#9D9D9D",
            alpha=0.18,
            linewidths=0,
            zorder=1,
        )

    # Draw labels as Text objects, then de-overlap them
    texts = []
    for n in G.nodes():
        if n not in label_set:
            continue
        x0, y0 = pos[n]
        texts.append(
            ax.text(
                x0,
                y0,
                labels[n],
                fontsize=8,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.72),
                zorder=5,
            )
        )
    if texts:
        adjust_text(
            texts,
            ax=ax,
            only_move={"texts": "xy"},
            expand_text=(1.05, 1.15),
            expand_points=(1.2, 1.2),
            force_text=(0.15, 0.25),
            force_points=(0.2, 0.2),
            lim=400,
        )

    ax.set_title(
        f"Supplychain network (top {top_n} main nodes, labeled {label_top}; neighbors shown as aggregated clouds)",
        fontsize=14,
        pad=12,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Supplychain visualizations (reads node.csv + edge_supplychain.csv).")
    ap.add_argument("--base-dir", required=True, help="Directory containing node.csv and edge_supplychain.csv")
    ap.add_argument("--node-csv", default="", help="Path to node.csv (default: <base-dir>/node.csv)")
    ap.add_argument("--edge-supply", default="", help="Path to edge_supplychain.csv (default: <base-dir>/edge_supplychain.csv)")
    ap.add_argument("--viz-dir", default="", help=f"Output directory for visualization files (default: {DEFAULT_VIZ_DIR})")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--top-n", type=int, default=200)
    ap.add_argument("--label-top", type=int, default=50)
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    node_csv = Path(args.node_csv) if args.node_csv else (base_dir / "node.csv")
    edge_supply = Path(args.edge_supply) if args.edge_supply else (base_dir / "edge_supplychain.csv")
    viz_dir = Path(args.viz_dir) if args.viz_dir else DEFAULT_VIZ_DIR
    viz_dir.mkdir(parents=True, exist_ok=True)

    chosen_font = configure_matplotlib_chinese_font()
    _ = chosen_font  # kept for future logging if needed

    node_index = build_node_index(node_csv)
    edges_supply = read_csv_robust(edge_supply)

    metrics = compute_supplychain_metrics(node_csv, edges_supply)
    metrics.to_csv(viz_dir / "supplychain_node_metrics.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for col in ["degree", "supplier_count", "customer_count", "geom_mean_supplier_customer"]:
        summary_rows.append({"metric": col, **summarize_series(metrics[col])})
    pd.DataFrame(summary_rows).to_csv(viz_dir / "supplychain_summary_stats.csv", index=False, encoding="utf-8-sig")

    counts = plot_box_by_bins_log1p(metrics, x="注册资本_万元", y="degree", bins=args.bins, out_path=viz_dir / "box_degree_by_capital_bins.png")
    counts.to_csv(viz_dir / "capital_bins_counts.csv", index=False, encoding="utf-8-sig")

    plot_box_by_bins_log1p(metrics, x="注册资本_万元", y="supplier_count", bins=args.bins, out_path=viz_dir / "box_supplier_count.png")
    plot_box_by_bins_log1p(metrics, x="注册资本_万元", y="customer_count", bins=args.bins, out_path=viz_dir / "box_customer_count.png")
    plot_box_by_bins_log1p(
        metrics, x="注册资本_万元", y="geom_mean_supplier_customer", bins=args.bins, out_path=viz_dir / "box_geom_mean_supplier_customer.png"
    )

    plot_network_supplychain(
        edges_supply,
        metrics,
        node_index,
        viz_dir / "network_supplychain_top.png",
        top_n=args.top_n,
        label_top=args.label_top,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

