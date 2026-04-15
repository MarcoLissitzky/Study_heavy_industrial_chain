import argparse
import csv
import datetime as dt
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


EXCEL_WRAPPED_RE = re.compile(r'^\s*=\s*"(.*)"\s*$')


def read_csv_robust(path: Path) -> pd.DataFrame:
    """
    Robust reader for Tianyancha exports.

    Many of these CSVs have a trailing comma on every data line, producing an extra empty field.
    pandas.read_csv will misalign columns (effectively shifting left). We therefore parse with
    csv.reader and drop ONLY the trailing empty field when present.
    """
    last_err: Exception | None = None
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            with path.open("r", encoding=enc, newline="") as f:
                reader = csv.reader(f, delimiter=",", quotechar='"', escapechar="\\")
                header = next(reader)
                rows: list[list[str]] = []
                n = len(header)
                for row in reader:
                    if not row:
                        continue
                    if len(row) == n + 1 and row[-1] == "":
                        row = row[:-1]
                    if len(row) < n:
                        row = row + [""] * (n - len(row))
                    elif len(row) > n:
                        row = row[:n]
                    rows.append(row)
            return pd.DataFrame(rows, columns=header)
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"Failed to read CSV with common encodings: {path}") from last_err


def unwrap_excel_export(v: object) -> object:
    if v is None or pd.isna(v):
        return pd.NA
    s = str(v)
    m = EXCEL_WRAPPED_RE.match(s)
    if m:
        return m.group(1)
    return s


def normalize_spaces(s: str) -> str:
    s = s.replace("\u3000", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def clean_cell(v: object) -> object:
    v = unwrap_excel_export(v)
    if v is None or pd.isna(v):
        return pd.NA
    s = normalize_spaces(str(v))
    if s in {"", "-", "—", "–"}:
        return pd.NA
    return s


def normalize_name(s: object) -> object:
    v = clean_cell(s)
    if v is None or pd.isna(v):
        return pd.NA
    t = str(v).translate(PARENS_TRANSLATION)
    t = normalize_spaces(t)
    t = re.sub(r"\s*\(\s*", "(", t)
    t = re.sub(r"\s*\)\s*", ")", t)
    t = t.replace("，", ",")
    return t if t else pd.NA


def make_id(kind: str, token: str) -> str:
    return f"{kind}:{token}"


def is_company_like(name: str) -> bool:
    keywords = [
        "公司",
        "集团",
        "有限",
        "股份",
        "厂",
        "分公司",
        "中心",
        "局",
        "所",
        "院",
        "大学",
        "委员会",
        "银行",
        "保险",
        "基金",
        "事务所",
        "研究所",
    ]
    return any(k in name for k in keywords)


def parse_percent(v: object) -> float | None:
    v = clean_cell(v)
    if v is None or pd.isna(v):
        return None
    s = str(v)
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%", s)
    if not m:
        return None
    return float(m.group(1))


def first_non_na(*vals: object) -> object:
    for v in vals:
        if v is None or (isinstance(v, float) and pd.isna(v)) or pd.isna(v):
            continue
        return v
    return pd.NA


@dataclass
class NodeIndex:
    name_to_id: dict[str, str]  # normalized_name -> corp/name id
    id_to_name: dict[str, str]
    main_node_ids: set[str]


def build_node_index(node_csv: Path) -> NodeIndex:
    df = read_csv_robust(node_csv)
    required = {"系统匹配企业名称", "统一社会信用代码"}
    if not required <= set(df.columns):
        raise ValueError(f"node.csv missing required cols: {required - set(df.columns)}")

    name_to_id: dict[str, str] = {}
    id_to_name: dict[str, str] = {}
    main_ids: set[str] = set()

    for r in df.itertuples(index=False):
        name_raw = getattr(r, "系统匹配企业名称")
        code_raw = getattr(r, "统一社会信用代码")
        name_norm = normalize_name(name_raw)
        code = clean_cell(code_raw)

        if code is not None and not pd.isna(code):
            node_id = make_id("corp", str(code))
        else:
            if name_norm is None or pd.isna(name_norm):
                continue
            node_id = make_id("name", str(name_norm))

        if name_norm is not None and not pd.isna(name_norm):
            key = str(name_norm)
            if key not in name_to_id or (name_to_id[key].startswith("name:") and node_id.startswith("corp:")):
                name_to_id[key] = node_id

        id_to_name[node_id] = str(name_raw) if name_raw else (str(name_norm) if name_norm is not None and not pd.isna(name_norm) else node_id)
        main_ids.add(node_id)

    return NodeIndex(name_to_id=name_to_id, id_to_name=id_to_name, main_node_ids=main_ids)


def map_endpoint(raw_name: object, node_index: NodeIndex, *, allow_person: bool) -> tuple[str, str, str]:
    name_norm = normalize_name(raw_name)
    name_disp = clean_cell(raw_name)

    if name_norm is not None and not pd.isna(name_norm):
        key = str(name_norm)
        if key in node_index.name_to_id:
            node_id = node_index.name_to_id[key]
            kind = "corp" if node_id.startswith("corp:") else "name"
            return node_id, (str(name_disp) if name_disp is not None and not pd.isna(name_disp) else key), kind

    if name_disp is None or pd.isna(name_disp):
        return make_id("name", "UNKNOWN"), "UNKNOWN", "name"

    disp = str(name_disp)
    if allow_person and not is_company_like(disp):
        return make_id("person", disp), disp, "person"

    token = str(name_norm) if name_norm is not None and not pd.isna(name_norm) else disp
    return make_id("name", token), disp, "name"


def parse_csv_line(line: str) -> list[str]:
    return next(csv.reader([line], delimiter=",", quotechar='"', escapechar="\\"))


def parse_shareholder_edges(path: Path, node_index: NodeIndex) -> list[dict[str, Any]]:
    text: list[str] | None = None
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            text = path.read_text(encoding=enc, errors="strict").splitlines()
            break
        except Exception:  # noqa: BLE001
            continue
    if text is None:
        text = path.read_text(encoding="gb18030", errors="ignore").splitlines()

    current_company: str | None = None
    current_dim: str | None = None
    header: list[str] | None = None
    edges: list[dict[str, Any]] = []

    for raw in text:
        line = raw.strip()
        if not line:
            header = None
            continue

        if "," not in line:
            current_company = line
            current_dim = None
            header = None
            continue

        fields = parse_csv_line(line)
        if not fields:
            continue

        if fields[0].startswith("股东信息"):
            current_dim = fields[0]
            header = None
            continue

        if fields[0] == "序号":
            header = fields
            continue

        if current_company is None or header is None:
            continue

        row = dict(zip(header, fields, strict=False))
        sh_name = clean_cell(row.get("股东名称") or row.get("发起人名称"))
        if sh_name is None or pd.isna(sh_name):
            continue

        src_id, src_name, src_kind = map_endpoint(sh_name, node_index, allow_person=True)
        dst_id, dst_name, _ = map_endpoint(current_company, node_index, allow_person=False)

        edges.append(
            {
                "src_id": src_id,
                "dst_id": dst_id,
                "src_name": src_name,
                "dst_name": dst_name,
                "edge_type": "shareholder",
                "layer": "investment",
                "report_date": first_non_na(clean_cell(row.get("首次持股日期")), clean_cell(row.get("认缴出资日期"))),
                "report_date_raw": first_non_na(clean_cell(row.get("首次持股日期")), clean_cell(row.get("认缴出资日期"))),
                "weight_raw": clean_cell(row.get("持股比例")),
                "weight": parse_percent(row.get("持股比例")),
                "amount_raw": first_non_na(clean_cell(row.get("认缴出资额(万元)")), clean_cell(row.get("实缴出资额(万元)"))),
                "source": pd.NA,
                "relation_hint": pd.NA,
                "source_file": "股东信息.csv",
                "extra": "|".join(
                    [
                        f"认缴出资日期={clean_cell(row.get('认缴出资日期'))}" if row.get("认缴出资日期") else "",
                        f"实缴出资日期={clean_cell(row.get('实缴出资日期'))}" if row.get("实缴出资日期") else "",
                        f"认缴出资额(万元)={clean_cell(row.get('认缴出资额(万元)'))}" if row.get("认缴出资额(万元)") else "",
                        f"实缴出资额(万元)={clean_cell(row.get('实缴出资额(万元)'))}" if row.get("实缴出资额(万元)") else "",
                        f"最终受益股份={clean_cell(row.get('最终受益股份'))}" if row.get("最终受益股份") else "",
                        f"关联产品/机构={clean_cell(row.get('关联产品/机构'))}" if row.get("关联产品/机构") else "",
                    ]
                ).strip("|"),
                "source_dim": current_dim or "股东信息",
                "src_kind": src_kind,
            }
        )

    return edges


def dedupe_edges(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    for c in keys:
        if c not in df.columns:
            df[c] = pd.NA
    return df.drop_duplicates(subset=keys, keep="first").copy()


def build_supplychain_edges(base_dir: Path, node_index: NodeIndex) -> pd.DataFrame:
    edges: list[dict[str, Any]] = []

    def add_edges(df: pd.DataFrame, *, edge_type: str, partner_col: str, weight_col: str, amount_col: str, source_file: str) -> None:
        for r in df.itertuples(index=False):
            src_id, src_name, _ = map_endpoint(getattr(r, "企业名称"), node_index, allow_person=False)
            dst_id, dst_name, _ = map_endpoint(getattr(r, partner_col), node_index, allow_person=False)
            edges.append(
                {
                    "src_id": src_id,
                    "dst_id": dst_id,
                    "src_name": src_name,
                    "dst_name": dst_name,
                    "edge_type": edge_type,
                    "layer": "supplychain",
                    "report_date": clean_cell(getattr(r, "报告期")) if hasattr(r, "报告期") else pd.NA,
                    "report_date_raw": clean_cell(getattr(r, "报告期")) if hasattr(r, "报告期") else pd.NA,
                    "weight_raw": clean_cell(getattr(r, weight_col)) if hasattr(r, weight_col) else pd.NA,
                    "weight": parse_percent(getattr(r, weight_col)) if hasattr(r, weight_col) else None,
                    "amount_raw": clean_cell(getattr(r, amount_col)) if hasattr(r, amount_col) else pd.NA,
                    "source": clean_cell(getattr(r, "数据来源")) if hasattr(r, "数据来源") else pd.NA,
                    "relation_hint": clean_cell(getattr(r, "关联关系")) if hasattr(r, "关联关系") else pd.NA,
                    "source_file": source_file,
                    "source_dim": pd.NA,
                }
            )

    sup = read_csv_robust(base_dir / "供应商.csv")
    cus = read_csv_robust(base_dir / "客户.csv")
    add_edges(sup, edge_type="supplier", partner_col="供应商", weight_col="采购占比", amount_col="采购金额", source_file="供应商.csv")
    add_edges(cus, edge_type="customer", partner_col="客户", weight_col="销售占比", amount_col="销售金额", source_file="客户.csv")

    out = pd.DataFrame(edges)
    out = out[(out["src_id"] != "name:UNKNOWN") & (out["dst_id"] != "name:UNKNOWN")].copy()
    return out


def build_investment_edges(base_dir: Path, node_index: NodeIndex) -> pd.DataFrame:
    edges: list[dict[str, Any]] = []

    inv = read_csv_robust(base_dir / "对外投资.csv")
    for r in inv.itertuples(index=False):
        src_id, src_name, _ = map_endpoint(getattr(r, "企业名称"), node_index, allow_person=False)
        dst_id, dst_name, _ = map_endpoint(getattr(r, "被投资企业名称"), node_index, allow_person=False)
        edges.append(
            {
                "src_id": src_id,
                "dst_id": dst_id,
                "src_name": src_name,
                "dst_name": dst_name,
                "edge_type": "invest",
                "layer": "investment",
                "report_date": clean_cell(getattr(r, "成立日期")) if hasattr(r, "成立日期") else pd.NA,
                "report_date_raw": clean_cell(getattr(r, "成立日期")) if hasattr(r, "成立日期") else pd.NA,
                "weight_raw": clean_cell(getattr(r, "持股比例")) if hasattr(r, "持股比例") else pd.NA,
                "weight": parse_percent(getattr(r, "持股比例")) if hasattr(r, "持股比例") else None,
                "amount_raw": clean_cell(getattr(r, "认缴出资额")) if hasattr(r, "认缴出资额") else pd.NA,
                "source": pd.NA,
                "relation_hint": clean_cell(getattr(r, "状态")) if hasattr(r, "状态") else pd.NA,
                "source_file": "对外投资.csv",
                "extra": "|".join(
                    [
                        f"状态={clean_cell(getattr(r, '状态'))}" if hasattr(r, "状态") else "",
                        f"最终受益股份={clean_cell(getattr(r, '最终受益股份'))}" if hasattr(r, "最终受益股份") else "",
                        f"所属地区={clean_cell(getattr(r, '所属地区'))}" if hasattr(r, "所属地区") else "",
                        f"所属行业={clean_cell(getattr(r, '所属行业'))}" if hasattr(r, "所属行业") else "",
                        f"关联产品/机构={clean_cell(getattr(r, '关联产品/机构'))}" if hasattr(r, "关联产品/机构") else "",
                    ]
                ).strip("|"),
                "source_dim": pd.NA,
                "src_kind": pd.NA,
            }
        )

    edges.extend(parse_shareholder_edges(base_dir / "股东信息.csv", node_index))
    out = pd.DataFrame(edges)
    out = out[(out["src_id"] != "name:UNKNOWN") & (out["dst_id"] != "name:UNKNOWN")].copy()
    return out


def build_fringe_nodes(
    edge_dfs: list[tuple[str, pd.DataFrame]],
    node_index: NodeIndex,
) -> pd.DataFrame:
    first_seen: dict[str, tuple[str, str]] = {}
    name_examples: dict[str, set[str]] = defaultdict(set)

    for origin, df in edge_dfs:
        for _, r in df.iterrows():
            for side in ("src", "dst"):
                node_id = r[f"{side}_id"]
                node_name = r[f"{side}_name"]
                if node_id in node_index.main_node_ids:
                    continue
                if node_id not in first_seen:
                    first_seen[node_id] = (origin, str(r.get("source_file", "")))
                if isinstance(node_name, str) and node_name:
                    name_examples[node_id].add(node_name)

    rows: list[dict[str, Any]] = []
    for node_id, (seen_in, seen_file) in first_seen.items():
        if node_id.startswith("corp:"):
            kind = "corp"
            name = node_index.id_to_name.get(node_id, node_id[len("corp:") :])
        elif node_id.startswith("person:"):
            kind = "person"
            name = node_id[len("person:") :]
        else:
            kind = "name"
            name = node_id.split(":", 1)[1] if ":" in node_id else node_id
        ex = sorted(name_examples.get(node_id, set()))
        rows.append(
            {
                "node_id": node_id,
                "node_name": name,
                "node_kind": kind,
                "first_seen_in": seen_in,
                "first_seen_file": seen_file,
                "examples": " | ".join(ex[:3]),
            }
        )

    return pd.DataFrame(rows).sort_values(by=["node_kind", "node_id"])


def main() -> int:
    ap = argparse.ArgumentParser(description="Build edges (2 layers) and node_fringe.csv.")
    ap.add_argument("--base-dir", required=True, help="Directory containing Tianyancha exports and node.csv")
    ap.add_argument("--node-csv", default="", help="Path to node.csv (default: <base-dir>/node.csv)")
    ap.add_argument("--out-dir", default="", help="Output directory (default: base-dir)")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    node_csv = Path(args.node_csv) if args.node_csv else (base_dir / "node.csv")
    out_dir = Path(args.out_dir) if args.out_dir else base_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    node_index = build_node_index(node_csv)

    edges_supply = build_supplychain_edges(base_dir, node_index)
    edges_invest = build_investment_edges(base_dir, node_index)

    supply_keys = ["src_id", "dst_id", "edge_type", "report_date_raw", "amount_raw", "weight_raw", "source_file"]
    invest_keys = ["src_id", "dst_id", "edge_type", "report_date_raw", "amount_raw", "weight_raw", "source_file", "source_dim"]
    edges_supply = dedupe_edges(edges_supply, supply_keys)
    edges_invest = dedupe_edges(edges_invest, invest_keys)

    edge_supply_path = out_dir / "edge_supplychain.csv"
    edge_invest_path = out_dir / "edge_investment.csv"
    edges_supply.to_csv(edge_supply_path, index=False, encoding="utf-8-sig")
    edges_invest.to_csv(edge_invest_path, index=False, encoding="utf-8-sig")

    fringe = build_fringe_nodes([("supplychain", edges_supply), ("investment", edges_invest)], node_index)
    fringe_path = out_dir / "node_fringe.csv"
    fringe.to_csv(fringe_path, index=False, encoding="utf-8-sig")

    stats_path = out_dir / "edge_build_stats.txt"
    with stats_path.open("w", encoding="utf-8") as f:
        f.write(f"generated_at={dt.datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"edge_supplychain_rows={len(edges_supply)}\n")
        f.write(f"edge_investment_rows={len(edges_invest)}\n")
        f.write(f"fringe_nodes={len(fringe)}\n")
        f.write("edge_type_counts_supplychain:\n")
        for k, v in edges_supply["edge_type"].value_counts().items():
            f.write(f"  {k}={int(v)}\n")
        f.write("edge_type_counts_investment:\n")
        for k, v in edges_invest["edge_type"].value_counts().items():
            f.write(f"  {k}={int(v)}\n")

    print(f"Wrote: {edge_supply_path}")
    print(f"Wrote: {edge_invest_path}")
    print(f"Wrote: {fringe_path}")
    print(f"Wrote: {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

