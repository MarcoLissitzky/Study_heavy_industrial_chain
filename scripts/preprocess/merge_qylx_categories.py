import argparse
import re
from pathlib import Path

import pandas as pd


PARENS_TRANSLATION = str.maketrans(
    {
        "（": "(",
        "）": ")",
        "【": "(",
        "】": ")",
        "［": "(",
        "］": ")",
    }
)


def norm_parens_and_space(s: str) -> str:
    s = str(s).translate(PARENS_TRANSLATION)
    s = s.replace("\u3000", " ")
    s = re.sub(r"\s+", " ", s).strip()
    # normalize around parentheses
    s = re.sub(r"\s*\(\s*", "(", s)
    s = re.sub(r"\s*\)\s*", ")", s)
    return s


def synonym_normalize(s: str) -> str:
    """
    Only do conservative, high-confidence merges:
    - unify various bracket styles and spacing (done before)
    - unify some punctuation variants: full-width comma to comma
    - remove redundant variants like '非自然人投资或控股的法人独资' vs same (already equal after parens)
    """
    s = s.replace("，", ",")
    # Some exports use '台港澳' vs '港澳台' inconsistently; keep as-is (not always synonym).
    # Keep normalization minimal to avoid wrong merges.
    return s


def normalize_category(raw: str) -> str:
    s = norm_parens_and_space(raw)
    s = synonym_normalize(s)
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge 企业(机构)类型 categories with source mapping.")
    ap.add_argument("--input", required=True, help="Path to 企业机构类型_取值统计.tsv")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse the TSV exported by inspect_qylx_values.py
    # First line is 'unique_nonempty=69', second line is header.
    lines = in_path.read_text(encoding="utf-8-sig").splitlines()
    data_lines = [ln for ln in lines if ln.strip() and not ln.startswith("unique_nonempty=")]
    if not data_lines:
        raise RuntimeError("Input file has no data lines.")

    from io import StringIO

    df = pd.read_csv(StringIO("\n".join(data_lines)), sep="\t", dtype={"value": str, "count": int})
    if "value" not in df.columns or "count" not in df.columns:
        raise RuntimeError("Input TSV must contain columns: value, count")

    df["value"] = df["value"].astype(str)
    df["normalized"] = df["value"].map(normalize_category)

    # Mapping file: each raw value -> normalized
    mapping = df[["value", "normalized", "count"]].sort_values(
        by=["normalized", "count", "value"], ascending=[True, False, True]
    )

    # Aggregated groups
    agg = (
        df.groupby("normalized", as_index=False)["count"]
        .sum()
        .rename(columns={"count": "merged_count"})
        .sort_values(by=["merged_count", "normalized"], ascending=[False, True])
    )

    # Build a human-readable provenance column listing sources
    prov_parts = (
        mapping.groupby("normalized")
        .apply(lambda g: " | ".join([f"{r.value}({int(r.count)})" for r in g.itertuples(index=False)]))
        .reset_index(name="sources")
    )
    agg = agg.merge(prov_parts, on="normalized", how="left")

    out_mapping = out_dir / "企业机构类型_归并映射.tsv"
    out_merged = out_dir / "企业机构类型_归并统计.tsv"

    mapping.to_csv(out_mapping, sep="\t", index=False, encoding="utf-8-sig")
    agg.to_csv(out_merged, sep="\t", index=False, encoding="utf-8-sig")

    print(f"Wrote: {out_mapping}")
    print(f"Wrote: {out_merged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

