import argparse
import re
from pathlib import Path

import pandas as pd


TARGET_COLUMNS = [
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
]

STATUS_COL = "登记状态"
TYPE_COL = "企业(机构)类型"
TYPE_LEVEL2_COL = "企业机构类型大类"
DEFAULT_TYPE_MAPPING_FILENAME = "企业机构类型_二级归并_标准映射.tsv"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "processed" / "supplychain_pps5000"


EXCEL_WRAPPED_RE = re.compile(r'^\s*=\s*"(.*)"\s*$')

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


def unwrap_excel_export(v: object) -> object:
    if v is None or pd.isna(v):
        return pd.NA
    s = str(v)
    m = EXCEL_WRAPPED_RE.match(s)
    if m:
        return m.group(1)
    return s


def normalize_spaces(s: str) -> str:
    # normalize full-width spaces and collapse whitespace
    s = s.replace("\u3000", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_parens_and_space(s: str) -> str:
    s = str(s).translate(PARENS_TRANSLATION)
    s = s.replace("\u3000", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s*\(\s*", "(", s)
    s = re.sub(r"\s*\)\s*", ")", s)
    return s


def level2_category(type_value: object) -> object:
    v = clean_cell(type_value)
    if v is None or pd.isna(v):
        return pd.NA
    s = normalize_parens_and_space(str(v)).replace("，", ",")

    # Non-legal-person orgs / branches
    if any(k in s for k in ("分公司", "分支机构", "非法人")):
        return "一般法人机构/其他"

    # State-owned / public ownership
    if "国有" in s or "全民所有制" in s:
        return "国有企业"

    # Foreign / HK/Macau/Taiwan / Sino-foreign JV / overseas
    if any(k in s for k in ("外商", "港澳台", "台港澳", "中外", "外国", "外资")):
        return "外资及港澳台企业"

    # Private-ish: natural person, partnerships, co-ops
    if any(k in s for k in ("自然人", "个人独资", "合伙企业", "农民专业合作社")):
        return "民营企业"

    if s in {"其他", "集体所有制"}:
        return "其他企业"

    # Default bucket for corporate entities with unclear ownership
    if re.search(r"(有限责任公司|股份有限公司|股份合作制)", s):
        return "其他企业"

    return "一般法人机构/其他"


def clean_cell(v: object) -> object:
    v = unwrap_excel_export(v)
    if v is None or pd.isna(v):
        return pd.NA
    s = normalize_spaces(str(v))
    if s in {"", "-", "—", "–"}:
        return pd.NA
    return s


def normalize_name_for_id(v: object) -> object:
    v = clean_cell(v)
    if v is None or pd.isna(v):
        return pd.NA
    s = str(v)
    # remove surrounding quotes commonly introduced in exports
    s = s.strip().strip('"').strip("'")
    s = normalize_spaces(s)
    return s if s else pd.NA


def read_csv_robust(path: Path) -> pd.DataFrame:
    # Tianyancha exports are often UTF-8-SIG or GB18030
    last_err: Exception | None = None
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, dtype=str, encoding=enc, keep_default_na=False)
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"Failed to read CSV with common encodings: {path}") from last_err


def load_type_level2_mapping(path: Path) -> dict[str, str]:
    """
    Load mapping file: 企业机构类型_二级归并_标准映射.tsv
    Expected columns: normalized, level2, ...
    Returns dict: normalized -> level2
    """
    mp = read_csv_robust(path) if path.suffix.lower() == ".csv" else pd.read_csv(path, sep="\t", dtype=str, encoding="utf-8-sig")
    mp = mp.fillna("")
    if "normalized" not in mp.columns or "level2" not in mp.columns:
        raise ValueError(f"Type mapping file must include columns: normalized, level2. Got: {list(mp.columns)}")
    mp["normalized"] = mp["normalized"].map(lambda x: normalize_parens_and_space(str(x)).replace("，", ","))
    mp["level2"] = mp["level2"].astype(str).str.strip()
    mp = mp[(mp["normalized"] != "") & (mp["level2"] != "")]
    return dict(zip(mp["normalized"], mp["level2"], strict=False))


def build_node_df(df: pd.DataFrame, *, type_mapping: dict[str, str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = [c for c in TARGET_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input: {missing}")

    # filter to 存续 only (after cleaning)
    if STATUS_COL in df.columns:
        status = df[STATUS_COL].map(clean_cell)
        df = df.loc[status == "存续"].copy()

    work = df[TARGET_COLUMNS].copy()
    for c in TARGET_COLUMNS:
        work[c] = work[c].map(clean_cell)

    if TYPE_COL in df.columns:
        if type_mapping is not None:
            raw_type = df[TYPE_COL].map(clean_cell)
            norm_type = raw_type.map(lambda x: pd.NA if pd.isna(x) else normalize_parens_and_space(str(x)).replace("，", ","))
            work[TYPE_LEVEL2_COL] = norm_type.map(lambda x: type_mapping.get(str(x), pd.NA) if not pd.isna(x) else pd.NA)
        else:
            # fallback to rule-based (kept for backward compatibility)
            work[TYPE_LEVEL2_COL] = df[TYPE_COL].map(level2_category)
    else:
        work[TYPE_LEVEL2_COL] = pd.NA

    credit = work["统一社会信用代码"].map(clean_cell)
    name = work["系统匹配企业名称"].map(normalize_name_for_id)

    node_id = credit.where(credit.notna(), name)
    work["_node_id"] = node_id

    # completeness score on target fields (exclude internal id)
    score_cols = [*TARGET_COLUMNS, TYPE_LEVEL2_COL]
    work["_score"] = work[score_cols].notna().sum(axis=1)

    # duplicates report BEFORE collapsing (only where node_id exists)
    dup_mask = work["_node_id"].notna() & work["_node_id"].duplicated(keep=False)
    duplicates = work.loc[dup_mask, ["_node_id", "_score", *score_cols]].sort_values(
        by=["_node_id", "_score"], ascending=[True, False]
    )

    # dedupe: keep best-scoring row per node_id; keep original order for ties
    keepable = work[work["_node_id"].notna()].copy()
    keepable["_row_order"] = range(len(keepable))
    keepable = keepable.sort_values(by=["_node_id", "_score", "_row_order"], ascending=[True, False, True])
    deduped = keepable.drop_duplicates(subset=["_node_id"], keep="first").drop(
        columns=["_node_id", "_score", "_row_order"]
    )

    # if there are rows with no id (no credit code AND no name), we drop them but keep in duplicates as empty
    return deduped, duplicates


def main() -> int:
    p = argparse.ArgumentParser(description="Build node.csv from Tianyancha 基础数据.csv")
    p.add_argument("--input", required=True, help="Path to 基础数据.csv")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help=f"Output directory (default: {DEFAULT_OUT_DIR})")
    p.add_argument(
        "--type-mapping",
        default="",
        help="Path to 企业机构类型_二级归并_标准映射.tsv (normalized->level2). Default: <out-dir>/企业机构类型_二级归并_标准映射.tsv",
    )
    args = p.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_csv_robust(in_path)
    mapping_path = Path(args.type_mapping) if args.type_mapping else (out_dir / DEFAULT_TYPE_MAPPING_FILENAME)
    type_mapping: dict[str, str] | None = None
    if mapping_path.exists():
        type_mapping = load_type_level2_mapping(mapping_path)

    node_df, dup_df = build_node_df(df, type_mapping=type_mapping)

    node_path = out_dir / "node.csv"
    dup_path = out_dir / "duplicates_report_node.csv"
    stats_path = out_dir / "node_build_stats.txt"

    node_df.to_csv(node_path, index=False, encoding="utf-8-sig")
    if len(dup_df) > 0:
        dup_df.to_csv(dup_path, index=False, encoding="utf-8-sig")

    # stats
    total_rows = len(df)
    out_rows = len(node_df)
    credit_nonempty = df.get("统一社会信用代码", pd.Series([], dtype=str)).map(clean_cell).notna().sum()
    with stats_path.open("w", encoding="utf-8") as f:
        f.write(f"input_rows={total_rows}\n")
        f.write(f"output_rows={out_rows}\n")
        f.write(f"credit_code_nonempty_rows={int(credit_nonempty)}\n")
        f.write(f"duplicates_rows={(len(dup_df))}\n")
        f.write("missing_rate_by_column:\n")
        miss = node_df[[*TARGET_COLUMNS, TYPE_LEVEL2_COL]].isna().mean().sort_values(ascending=False)
        for k, v in miss.items():
            f.write(f"  {k}={v:.4f}\n")

    print(f"Wrote: {node_path}")
    if len(dup_df) > 0:
        print(f"Wrote: {dup_path}")
    print(f"Wrote: {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
