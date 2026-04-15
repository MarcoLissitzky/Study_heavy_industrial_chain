import argparse
import math
import re
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure project root is on sys.path so `scripts.*` imports work
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_edges import clean_cell, read_csv_robust


AMOUNT_RE = re.compile(r"([+-]?[0-9][0-9,]*(?:\.[0-9]+)?)\s*(亿元|万元|元)?")


def parse_amount_wanyuan(raw: object) -> float | None:
    cell = clean_cell(raw)
    if cell is None or pd.isna(cell):
        return None
    text = str(cell).replace(",", "")
    match = AMOUNT_RE.search(text)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2) or "万元"
    if unit == "亿元":
        return value * 10000.0
    if unit == "元":
        return value / 10000.0
    return value


def add_amount_columns(df: pd.DataFrame, amount_col: str) -> pd.DataFrame:
    out = df.copy()
    out["amount_wanyuan"] = out[amount_col].apply(parse_amount_wanyuan)
    out["has_amount"] = out["amount_wanyuan"].notna()
    out["amount_is_zero"] = out["amount_wanyuan"].fillna(1.0).eq(0.0)
    return out


def quantiles(series: pd.Series) -> dict[str, float]:
    qs = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    qv = series.quantile(qs)
    return {f"q{int(q * 100):02d}": float(qv.loc[q]) for q in qs}


def bucketize(amount_series: pd.Series) -> pd.DataFrame:
    bins = [-math.inf, 1.0, 10.0, 100.0, math.inf]
    labels = ["<1", "1-10", "10-100", ">=100"]
    cat = pd.cut(amount_series, bins=bins, labels=labels, right=False)
    counts = cat.value_counts(dropna=False).reindex(labels, fill_value=0).rename("count")
    out = counts.to_frame()
    total = int(counts.sum())
    out["ratio"] = (out["count"] / total).fillna(0.0)
    out.index.name = "bucket_wanyuan"
    return out.reset_index()


def summarize_dataset(df: pd.DataFrame, label: str) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    total = int(len(df))
    has_amount = int(df["has_amount"].sum())
    zeros = int(df.loc[df["has_amount"], "amount_is_zero"].sum())
    ratio = (has_amount / total) if total else 0.0

    positive = df.loc[df["has_amount"] & ~df["amount_is_zero"], "amount_wanyuan"].astype(float)
    all_valid = df.loc[df["has_amount"], "amount_wanyuan"].astype(float)

    row = {
        "dataset": label,
        "total_rows": total,
        "rows_with_amount": has_amount,
        "rows_without_amount": total - has_amount,
        "amount_ratio": ratio,
        "zero_amount_rows": zeros,
        "positive_amount_rows": int(len(positive)),
    }

    if len(all_valid) > 0:
        desc = all_valid.describe()
        row.update(
            {
                "mean_wanyuan": float(desc["mean"]),
                "std_wanyuan": float(desc["std"]) if not pd.isna(desc["std"]) else 0.0,
                "min_wanyuan": float(desc["min"]),
                "max_wanyuan": float(desc["max"]),
                **quantiles(all_valid),
            }
        )
    else:
        row.update(
            {
                "mean_wanyuan": math.nan,
                "std_wanyuan": math.nan,
                "min_wanyuan": math.nan,
                "max_wanyuan": math.nan,
                "q25": math.nan,
                "q50": math.nan,
                "q75": math.nan,
                "q90": math.nan,
                "q95": math.nan,
                "q99": math.nan,
            }
        )

    buckets = bucketize(all_valid) if len(all_valid) > 0 else pd.DataFrame({"bucket_wanyuan": [], "count": [], "ratio": []})
    return row, all_valid.to_frame(name="amount_wanyuan"), buckets


def plot_hist(series: pd.Series, title: str, out_path: Path, use_log_x: bool) -> None:
    plt.figure(figsize=(8, 5))
    values = series[series > 0].astype(float)
    if len(values) == 0:
        plt.close()
        return
    if use_log_x:
        bins = np.logspace(np.log10(values.min()), np.log10(values.max()), 40)
        plt.hist(values, bins=bins)
        plt.xscale("log")
    else:
        plt.hist(values, bins=40)
    plt.title(title)
    plt.xlabel("amount_wanyuan")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def clone_base_dir(base_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for child in base_dir.iterdir():
        target = out_dir / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        else:
            shutil.copy2(child, target)
    ensure_root_node_csv(out_dir)


def ensure_root_node_csv(base_dir: Path) -> None:
    root_node = base_dir / "node.csv"
    if root_node.exists():
        return
    fallback_node = base_dir / "network" / "node" / "node.csv"
    if fallback_node.exists():
        shutil.copy2(fallback_node, root_node)


def run_one(base_dir: Path, filename: str, amount_col: str, label: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = read_csv_robust(base_dir / filename)
    enriched = add_amount_columns(df, amount_col=amount_col)
    summary_row, valid_amounts, buckets = summarize_dataset(enriched, label=label)
    return enriched, pd.DataFrame([summary_row]), buckets


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze customer/supplier amount coverage and distribution.")
    ap.add_argument("--base-dir", required=True, help="Directory that contains 客户.csv and 供应商.csv")
    ap.add_argument("--output-dir", default="", help="Output directory for stats (default: <base-dir>/reports_amount)")
    ap.add_argument("--amount-only-dir", default="", help="Optional output directory for filtered amount-only dataset")
    ap.add_argument("--plot", action="store_true", help="Generate histogram PNG files")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir) if args.output_dir else (base_dir / "reports_amount")
    output_dir.mkdir(parents=True, exist_ok=True)

    customer_df, customer_summary, customer_buckets = run_one(base_dir, "客户.csv", "销售金额", "customer")
    supplier_df, supplier_summary, supplier_buckets = run_one(base_dir, "供应商.csv", "采购金额", "supplier")

    summary = pd.concat([customer_summary, supplier_summary], ignore_index=True)
    summary.to_csv(output_dir / "amount_summary.csv", index=False, encoding="utf-8-sig")

    customer_buckets.to_csv(output_dir / "customer_amount_buckets.csv", index=False, encoding="utf-8-sig")
    supplier_buckets.to_csv(output_dir / "supplier_amount_buckets.csv", index=False, encoding="utf-8-sig")

    customer_amount_only = customer_df[customer_df["has_amount"]].drop(columns=["amount_wanyuan", "has_amount", "amount_is_zero"])
    supplier_amount_only = supplier_df[supplier_df["has_amount"]].drop(columns=["amount_wanyuan", "has_amount", "amount_is_zero"])
    customer_amount_only.to_csv(output_dir / "客户_amount_only_preview.csv", index=False, encoding="utf-8-sig")
    supplier_amount_only.to_csv(output_dir / "供应商_amount_only_preview.csv", index=False, encoding="utf-8-sig")

    if args.plot:
        plot_hist(
            customer_df.loc[customer_df["has_amount"], "amount_wanyuan"],
            "Customer Amount Distribution (wanyuan)",
            output_dir / "customer_amount_hist.png",
            use_log_x=False,
        )
        plot_hist(
            supplier_df.loc[supplier_df["has_amount"], "amount_wanyuan"],
            "Supplier Amount Distribution (wanyuan)",
            output_dir / "supplier_amount_hist.png",
            use_log_x=False,
        )
        plot_hist(
            customer_df.loc[customer_df["has_amount"], "amount_wanyuan"],
            "Customer Amount Distribution LogX (wanyuan)",
            output_dir / "customer_amount_hist_logx.png",
            use_log_x=True,
        )
        plot_hist(
            supplier_df.loc[supplier_df["has_amount"], "amount_wanyuan"],
            "Supplier Amount Distribution LogX (wanyuan)",
            output_dir / "supplier_amount_hist_logx.png",
            use_log_x=True,
        )

    if args.amount_only_dir:
        amount_only_dir = Path(args.amount_only_dir)
        clone_base_dir(base_dir, amount_only_dir)
        customer_amount_only.to_csv(amount_only_dir / "客户.csv", index=False, encoding="utf-8-sig")
        supplier_amount_only.to_csv(amount_only_dir / "供应商.csv", index=False, encoding="utf-8-sig")

    print(f"[ok] summary -> {output_dir / 'amount_summary.csv'}")
    if args.amount_only_dir:
        print(f"[ok] amount-only base dir -> {args.amount_only_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
