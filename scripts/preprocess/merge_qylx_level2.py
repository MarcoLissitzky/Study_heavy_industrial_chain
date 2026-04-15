import argparse
import re
from pathlib import Path

import pandas as pd


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, encoding="utf-8-sig").fillna("")


def level2_category(normalized: str) -> str:
    s = str(normalized).strip()
    if not s:
        return "一般法人机构/其他"

    # Non-legal-person orgs / branches
    if any(k in s for k in ("分公司", "分支机构", "非法人")):
        return "一般法人机构/其他"

    # State-owned / public ownership
    if "国有" in s or "全民所有制" in s:
        return "国有企业"

    # Foreign / HK/Macau/Taiwan / Sino-foreign JV / overseas
    if any(
        k in s
        for k in (
            "外商",
            "港澳台",
            "台港澳",
            "中外",
            "外国",
            "外资",
        )
    ):
        return "外资及港澳台企业"

    # Private / non-state: natural person, partnerships, co-ops
    if any(k in s for k in ("自然人", "个人独资", "合伙企业", "农民专业合作社")):
        return "民营企业"

    # Explicit unknown / residual
    if s == "其他":
        return "其他企业"

    # Collective ownership is not always state-owned; keep separate bucket
    if "集体所有制" in s:
        return "其他企业"

    # Default: corporate entities with unclear ownership
    if re.search(r"(有限责任公司|股份有限公司|股份合作制)", s):
        return "其他企业"

    return "一般法人机构/其他"


def main() -> int:
    ap = argparse.ArgumentParser(description="二级归并 企业(机构)类型 到 5 大类，并保留映射来源。")
    ap.add_argument("--mapping", required=True, help="Path to 企业机构类型_归并映射.tsv")
    ap.add_argument("--stats", required=True, help="Path to 企业机构类型_归并统计.tsv")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping = read_tsv(Path(args.mapping))
    stats = read_tsv(Path(args.stats))

    # mapping: value, normalized, count
    if not {"value", "normalized", "count"} <= set(mapping.columns):
        raise RuntimeError("mapping TSV must include columns: value, normalized, count")
    # stats: normalized, merged_count, sources
    if not {"normalized", "merged_count", "sources"} <= set(stats.columns):
        raise RuntimeError("stats TSV must include columns: normalized, merged_count, sources")

    stats["level2"] = stats["normalized"].map(level2_category)
    mapping["level2"] = mapping["normalized"].map(level2_category)

    # (1) Raw -> normalized -> level2 mapping
    out_raw = out_dir / "企业机构类型_二级归并_原始映射.tsv"
    mapping[["value", "normalized", "level2", "count"]].to_csv(out_raw, sep="\t", index=False, encoding="utf-8-sig")

    # (2) Normalized -> level2 mapping with provenance
    out_norm = out_dir / "企业机构类型_二级归并_标准映射.tsv"
    stats[["normalized", "level2", "merged_count", "sources"]].to_csv(
        out_norm, sep="\t", index=False, encoding="utf-8-sig"
    )

    # (3) Level2 aggregated stats with provenance (list normalized groups)
    prov = (
        stats.sort_values(by=["level2", "merged_count", "normalized"], ascending=[True, False, True])
        .groupby("level2")
        .apply(
            lambda g: " | ".join([f"{r.normalized}({int(r.merged_count)})" for r in g.itertuples(index=False)]),
            include_groups=False,
        )
        .reset_index(name="normalized_groups")
    )

    level2_agg = (
        stats.groupby("level2", as_index=False)["merged_count"]
        .apply(lambda s: s.astype(int).sum())
        .rename(columns={"merged_count": "level2_count"})
    )
    out_level2 = out_dir / "企业机构类型_二级归并_统计.tsv"
    level2_agg.merge(prov, on="level2", how="left").sort_values(by="level2_count", ascending=False).to_csv(
        out_level2, sep="\t", index=False, encoding="utf-8-sig"
    )

    print(f"Wrote: {out_raw}")
    print(f"Wrote: {out_norm}")
    print(f"Wrote: {out_level2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

