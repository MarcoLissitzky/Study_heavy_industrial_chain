import argparse
import csv
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl  # 新增 polars 引入

# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
LEGACY_DATA_DIR = PROJECT_ROOT / "database" / "工商企业注册信息（东北）"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "registry_ne"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports" / "sampling"
DEFAULT_SAMPLE_SIZE = 5000
DEFAULT_RANDOM_SEED = 42

HEAVY_KEYWORDS = {
    '采运',
    '洗选','开采','采选',
    '石油','化学','金属','矿物','生产和供应',
    '设备制造','汽车制造','器材制造业','仪表制造业'
}
TARGET_PROVINCES = {"黑龙江省", "吉林省", "辽宁省"}

# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────
def parse_capital(value):
    """注册资本字符串 → 万元数值，无法解析返回 None"""
    if not value or str(value).strip() in ("-", "", "None", "nan"):
        return None
    s = str(value).strip()
    m = re.search(r"[\d,]+\.?\d*", s)
    if not m:
        return None
    num = float(m.group().replace(",", ""))
#    if '亿' in s:
#        num *= 10000
#    elif '人民币' in s and '万' not in s:
#        num /= 10000
    return num if num > 0 else None

def is_alive(status_str):
    """判断经营状态是否为存续/在业"""
    if not status_str or str(status_str).strip() in ("", "nan", "NaN"):
        return False
    clean_str = str(status_str).strip()
    return clean_str in ("存续", "在业")

def is_heavy(industry_str):
    """根据行业名称（汉字）判断是否属于重工业"""
    if not industry_str or str(industry_str).strip() in ("", "nan", "NaN"):
        return False
    clean_str = str(industry_str).strip()
    return any(keyword in clean_str for keyword in HEAVY_KEYWORDS)

def is_intime(date_str):
    """判断核准日期字符串是否在可信范围内（2022-12-07 ~ 2025-12-31）"""
    if not date_str or str(date_str).strip() in ("", "nan", "NaN"):
        return False
    try:
        dt = pd.to_datetime(str(date_str).strip(), errors='coerce')
        return pd.Timestamp("2022/12/07") <= dt <= pd.Timestamp("2025/12/31")
    except Exception:
        return False  

def is_huge_capital(cap_str):
    """判断注册资本是否巨大（≥2000万元）"""
    cap = parse_capital(cap_str)
    return cap is not None and cap >= 2000  # 万元数值≥2000即≥2000万元


FIELDNAMES = [
    "企业名称",
    "经营状态",
    "法定代表人",
    "注册资本",
    "实缴资本",
    "成立日期",
    "核准日期",
    "营业期限",
    "所属省份",
    "所属城市",
    "所属区县",
    "统一社会信用代码",
    "纳税人识别号",
    "工商注册号",
    "组织机构代码",
    "参保人数",
    "企业类型",
    "所属行业",
    "曾用名",
    "注册地址",
    "网址",
    "联系电话",
    "邮箱",
    "经营范围",
    "注册资本_万元",
]


def run_sampling(data_dir: Path, output_dir: Path, sample_size: int, random_seed: int) -> int:
    temp_csv = output_dir / "_temp_filtered.csv"
    progress_f = output_dir / "_temp_progress.json"
    out_all_csv = output_dir / "黑吉辽重工业企业筛选结果.csv"
    out_smp_csv = output_dir / f"PPS抽样样本_{sample_size}家.csv"

    print("=" * 60)
    print("步骤 1：增量筛选（随取随存）")
    print("=" * 60)

    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".xlsx") and not f.startswith("~")])
    print(f"共发现 {len(all_files)} 个xlsx文件")

    done_files = set()
    if progress_f.exists():
        with progress_f.open("r", encoding="utf-8") as pf:
            done_files = set(json.load(pf).get("done", []))
        print(f"检测到断点记录，已处理 {len(done_files)} 个文件，继续处理剩余文件...")

    write_header = (not temp_csv.exists()) or len(done_files) == 0
    if len(done_files) == 0 and temp_csv.exists():
        temp_csv.unlink()

    total_written = 0
    with temp_csv.open("a", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()

        for i, fname in enumerate(all_files, 1):
            if fname in done_files:
                print(f"  [{i}/{len(all_files)}] 跳过（已处理）：{fname}")
                continue

            fpath = data_dir / fname
            try:
                df = pl.read_excel(str(fpath), engine="calamine")
            except Exception as e:
                print(f"  [{i}/{len(all_files)}] [警告] 读取失败：{fname} — {e}")
                done_files.add(fname)
                continue

            mask = (
                pl.col("所属省份").is_in(list(TARGET_PROVINCES))
                & pl.col("所属行业").map_elements(is_heavy, return_dtype=pl.Boolean)
                & pl.col("经营状态").map_elements(is_alive, return_dtype=pl.Boolean)
                & pl.col("注册资本").map_elements(is_huge_capital, return_dtype=pl.Boolean)
            )
            df_ok = df.filter(mask)
            df_ok = df_ok.with_columns(pl.col("注册资本").map_elements(parse_capital, return_dtype=pl.Float64).alias("注册资本_万元")).filter(
                pl.col("注册资本_万元").is_not_null()
            )

            dicts = df_ok.to_dicts()
            count = len(dicts)
            for row in dicts:
                writer.writerow(row)

            total_written += count
            print(f"  [{i}/{len(all_files)}] {fname}  → 筛选出 {count} 条（累计 {total_written:,}）", flush=True)

            done_files.add(fname)
            with progress_f.open("w", encoding="utf-8") as pf:
                json.dump({"done": list(done_files)}, pf, ensure_ascii=False)

    print(f"\n筛选完成，共写入 {total_written:,} 条记录到临时CSV")

    print("\n" + "=" * 60)
    print("步骤 2：读取筛选结果，统计信息")
    print("=" * 60)

    df_all = pl.read_csv(str(temp_csv))
    df_all = (
        df_all.with_columns(pl.col("注册资本_万元").cast(pl.Float64, strict=False))
        .filter(pl.col("注册资本_万元").is_not_null() & (pl.col("注册资本_万元") > 0))
        .sort("注册资本_万元", descending=True)
    )

    n_rows = df_all.height
    print(f"企业总数：{n_rows:,}")
    print("\n各省份分布：")
    print(df_all.get_column("所属省份").value_counts().sort("count", descending=True))
    print("\n各行业分布（Top 15）：")
    print(df_all.get_column("所属行业").value_counts().sort("count", descending=True).head(15))
    print("\n注册资本（万元）统计：")
    print(df_all.get_column("注册资本_万元").describe())

    df_all.with_row_index("序号", offset=1).write_csv(str(out_all_csv), include_bom=True)
    print(f"\n全量筛选结果已保存：{out_all_csv}")

    print("\n" + "=" * 60)
    print("步骤 3：PPS抽样（迭代截断与重分配）")
    print("=" * 60)

    if n_rows <= sample_size:
        print(f"[提示] 总企业数 {n_rows} ≤ 目标样本量 {sample_size}，全部选取。")
        df_sample = df_all.clone()
    else:
        rng = np.random.default_rng(random_seed)
        capital = df_all.get_column("注册资本_万元").to_numpy().copy()
        pos = np.arange(n_rows)

        selected_certain = []
        remaining_pos = pos.copy()
        remaining_cap = capital.copy()
        remaining_n = sample_size
        itr = 0

        while remaining_n > 0 and len(remaining_pos) > 0:
            itr += 1
            total = remaining_cap.sum()
            probs = remaining_n * remaining_cap / total
            certain_mask = probs >= 1.0
            if not certain_mask.any():
                break
            certain_pos = remaining_pos[certain_mask]
            selected_certain.extend(certain_pos.tolist())
            remaining_n -= len(certain_pos)
            keep = ~certain_mask
            remaining_pos = remaining_pos[keep]
            remaining_cap = remaining_cap[keep]

        print(f"迭代截断轮次：{itr}，必然入样企业数：{len(selected_certain):,}")

        selected_random = []
        if remaining_n > 0 and len(remaining_pos) > 0:
            total_rem = remaining_cap.sum()
            probs_rem = remaining_n * remaining_cap / total_rem
            probs_norm = probs_rem / probs_rem.sum()
            draw = rng.choice(len(remaining_pos), size=min(remaining_n, len(remaining_pos)), replace=False, p=probs_norm)
            selected_random = remaining_pos[draw].tolist()

        print(f"概率抽样企业数：{len(selected_random):,}")
        print(f"最终样本量：{len(selected_certain) + len(selected_random):,}")

        all_selected = sorted(set(selected_certain + selected_random))
        df_sample = df_all[all_selected].sort("注册资本_万元", descending=True)

    print("\n" + "=" * 60)
    print("步骤 4：输出抽样结果")
    print("=" * 60)

    df_sample.with_row_index("序号", offset=1).write_csv(str(out_smp_csv), include_bom=True)
    print(f"抽样样本已保存：{out_smp_csv}")
    print(f"  共 {df_sample.height:,} 家企业（按注册资本降序）")

    if progress_f.exists():
        progress_f.unlink()
    if temp_csv.exists():
        temp_csv.unlink()
    print("\n临时文件已清理。完成！")
    return 0


def list_xlsx_files(data_dir: Path) -> list[str]:
    return sorted([f for f in os.listdir(data_dir) if f.endswith(".xlsx") and not f.startswith("~")])


def main() -> int:
    ap = argparse.ArgumentParser(description="筛选东北重工业企业并进行 PPS 抽样。")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help=f"原始 xlsx 目录（默认: {DEFAULT_DATA_DIR}）")
    ap.add_argument(
        "--fallback-data-dir",
        default=str(LEGACY_DATA_DIR),
        help=f"当 data-dir 为空时可选回退目录（默认: {LEGACY_DATA_DIR}）",
    )
    ap.add_argument(
        "--allow-fallback-on-empty",
        action="store_true",
        help="仅在 data-dir 存在但没有 .xlsx 时，允许回退到 fallback-data-dir",
    )
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help=f"输出目录（默认: {DEFAULT_OUTPUT_DIR}）")
    ap.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE, help=f"样本量（默认: {DEFAULT_SAMPLE_SIZE}）")
    ap.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED, help=f"随机种子（默认: {DEFAULT_RANDOM_SEED}）")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    fallback_data_dir = Path(args.fallback_data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        raise FileNotFoundError(f"data-dir 不存在: {data_dir}")
    if not data_dir.is_dir():
        raise NotADirectoryError(f"data-dir 不是目录: {data_dir}")
    if args.sample_size <= 0:
        raise ValueError("--sample-size 必须大于 0")

    all_files = list_xlsx_files(data_dir)
    chosen_data_dir = data_dir
    if len(all_files) == 0:
        if not args.allow_fallback_on_empty:
            raise FileNotFoundError(
                f"data-dir 中未找到 .xlsx 文件: {data_dir}\n"
                "如需使用回退目录，请显式加上 --allow-fallback-on-empty 和 --fallback-data-dir。"
            )
        if not fallback_data_dir.exists():
            raise FileNotFoundError(f"fallback-data-dir 不存在: {fallback_data_dir}")
        if not fallback_data_dir.is_dir():
            raise NotADirectoryError(f"fallback-data-dir 不是目录: {fallback_data_dir}")
        fallback_files = list_xlsx_files(fallback_data_dir)
        if len(fallback_files) == 0:
            raise FileNotFoundError(f"fallback-data-dir 中也未找到 .xlsx 文件: {fallback_data_dir}")
        chosen_data_dir = fallback_data_dir
        print(f"[info] data-dir 为空，已回退到: {chosen_data_dir}")

    return run_sampling(
        data_dir=chosen_data_dir,
        output_dir=output_dir,
        sample_size=int(args.sample_size),
        random_seed=int(args.random_seed),
    )


if __name__ == "__main__":
    raise SystemExit(main())