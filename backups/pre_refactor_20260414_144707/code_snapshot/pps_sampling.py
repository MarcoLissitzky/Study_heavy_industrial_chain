import os
import re
import csv
import json
import heapq
import numpy as np
import pandas as pd
import polars as pl  # 新增 polars 引入

# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
DATA_DIR    = r'd:\Study\Study_heavy_industrial_chain\工商企业注册信息'
OUTPUT_DIR  = r'd:\Study\Study_heavy_industrial_chain'
TEMP_CSV    = os.path.join(OUTPUT_DIR, '_temp_filtered.csv')
PROGRESS_F  = os.path.join(OUTPUT_DIR, '_temp_progress.json')
OUT_ALL_CSV = os.path.join(OUTPUT_DIR, '黑吉辽重工业企业筛选结果.csv')
OUT_SMP_CSV = os.path.join(OUTPUT_DIR, 'PPS抽样样本_5000家.csv')
SAMPLE_SIZE = 5000
RANDOM_SEED = 42

HEAVY_KEYWORDS = {
    '采运',
    '洗选','开采','采选',
    '石油','化学','金属','矿物','生产和供应',
    '设备制造','汽车制造','器材制造业','仪表制造业'
}
TARGET_PROVINCES = {'黑龙江省', '吉林省', '辽宁省'}

# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────
def parse_capital(value):
    """注册资本字符串 → 万元数值，无法解析返回 None"""
    if not value or str(value).strip() in ('-', '', 'None', 'nan'):
        return None
    s = str(value).strip()
    m = re.search(r'[\d,]+\.?\d*', s)
    if not m:
        return None
    num = float(m.group().replace(',', ''))
#    if '亿' in s:
#        num *= 10000
#    elif '人民币' in s and '万' not in s:
#        num /= 10000
    return num if num > 0 else None

def is_alive(status_str):
    """判断经营状态是否为存续/在业"""
    if not status_str or str(status_str).strip() in ('', 'nan', 'NaN'):
        return False
    clean_str = str(status_str).strip()
    return clean_str in ('存续', '在业')

def is_heavy(industry_str):
    """根据行业名称（汉字）判断是否属于重工业"""
    if not industry_str or str(industry_str).strip() in ('', 'nan', 'NaN'):
        return False
    clean_str = str(industry_str).strip()
    return any(keyword in clean_str for keyword in HEAVY_KEYWORDS)

def is_intime(date_str):
    """判断核准日期字符串是否在可信范围内（2022-12-07 ~ 2025-12-31）"""
    if not date_str or str(date_str).strip() in ('', 'nan', 'NaN'):
        return False
    try:
        dt = pd.to_datetime(str(date_str).strip(), errors='coerce')
        return pd.Timestamp('2022/12/07') <= dt <= pd.Timestamp('2025/12/31')
    except Exception:
        return False  

def is_huge_capital(cap_str):
    """判断注册资本是否巨大（≥2000万元）"""
    cap = parse_capital(cap_str)
    return cap is not None and cap >= 2000  # 万元数值≥2000即≥2000万元  


# ─────────────────────────────────────────────
# 步骤 1：增量筛选，随取随存到 CSV
# ─────────────────────────────────────────────
print("=" * 60)
print("步骤 1：增量筛选（随取随存）")
print("=" * 60)

all_files = sorted([
    f for f in os.listdir(DATA_DIR)
    if f.endswith('.xlsx') and not f.startswith('~')
])
print(f"共发现 {len(all_files)} 个xlsx文件")

# 读取断点进度
done_files = set()
if os.path.exists(PROGRESS_F):
    with open(PROGRESS_F, 'r', encoding='utf-8') as pf:
        done_files = set(json.load(pf).get('done', []))
    print(f"检测到断点记录，已处理 {len(done_files)} 个文件，继续处理剩余文件...")

# 确定CSV表头（首次创建时写入）
FIELDNAMES = ['企业名称','经营状态','法定代表人','注册资本','实缴资本',
              '成立日期','核准日期','营业期限','所属省份','所属城市','所属区县',
              '统一社会信用代码','纳税人识别号','工商注册号','组织机构代码',
              '参保人数','企业类型','所属行业','曾用名','注册地址',
              '网址','联系电话','邮箱','经营范围','注册资本_万元']

write_header = not os.path.exists(TEMP_CSV) or len(done_files) == 0
if len(done_files) == 0 and os.path.exists(TEMP_CSV):
    os.remove(TEMP_CSV)  # 重新开始时清空旧文件

total_written = 0
with open(TEMP_CSV, 'a', newline='', encoding='utf-8-sig') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES, extrasaction='ignore')
    if write_header:
        writer.writeheader()

    for i, fname in enumerate(all_files, 1):
        if fname in done_files:
            print(f"  [{i}/{len(all_files)}] 跳过（已处理）：{fname}")
            continue

        fpath = os.path.join(DATA_DIR, fname)
        try:
            # 使用 polars + calamine 引擎读取 Excel，速度大幅提升
            df = pl.read_excel(fpath, engine="calamine")
        except Exception as e:
            print(f"  [{i}/{len(all_files)}] [警告] 读取失败：{fname} — {e}")
            done_files.add(fname)
            continue

        # 筛选
        mask = (
            pl.col('所属省份').is_in(list(TARGET_PROVINCES)) &
            pl.col('所属行业').map_elements(is_heavy, return_dtype=pl.Boolean) &
            pl.col('经营状态').map_elements(is_alive, return_dtype=pl.Boolean) &
            # pl.col('核准日期').map_elements(is_intime, return_dtype=pl.Boolean) &
            pl.col('注册资本').map_elements(is_huge_capital, return_dtype=pl.Boolean)
        )
        df_ok = df.filter(mask)

        # 解析注册资本并过滤空值
        df_ok = df_ok.with_columns(
            pl.col('注册资本').map_elements(parse_capital, return_dtype=pl.Float64).alias('注册资本_万元')
        ).filter(
            pl.col('注册资本_万元').is_not_null()
        )

        # 写入CSV
        dicts = df_ok.to_dicts()
        count = len(dicts)
        for row in dicts:
            writer.writerow(row)
            
        total_written += count

        print(f"  [{i}/{len(all_files)}] {fname}  → 筛选出 {count} 条（累计 {total_written:,}）", flush=True)

        # 更新断点
        done_files.add(fname)
        with open(PROGRESS_F, 'w', encoding='utf-8') as pf:
            json.dump({'done': list(done_files)}, pf, ensure_ascii=False)

print(f"\n筛选完成，共写入 {total_written:,} 条记录到临时CSV")


# ─────────────────────────────────────────────
# 步骤 2：读取临时CSV，统计并排序
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("步骤 2：读取筛选结果，统计信息")
print("=" * 60)

# 使用 polars 读取
df_all = pl.read_csv(TEMP_CSV)
df_all = df_all.with_columns(
    pl.col('注册资本_万元').cast(pl.Float64, strict=False)
).filter(
    pl.col('注册资本_万元').is_not_null() & (pl.col('注册资本_万元') > 0)
).sort('注册资本_万元', descending=True)

N = df_all.height
print(f"企业总数：{N:,}")
print(f"\n各省份分布：")
print(df_all.get_column('所属省份').value_counts().sort('count', descending=True))
print(f"\n各行业分布（Top 15）：")
print(df_all.get_column('所属行业').value_counts().sort('count', descending=True).head(15))
print(f"\n注册资本（万元）统计：")
print(df_all.get_column('注册资本_万元').describe())

# 保存全量筛选结果 (生成带 BOM 的 utf-8 以防 Excel 乱码)
df_all.with_row_index('序号', offset=1).write_csv(OUT_ALL_CSV, include_bom=True)
print(f"\n全量筛选结果已保存：{OUT_ALL_CSV}")


# ─────────────────────────────────────────────
# 步骤 3：PPS抽样（迭代截断与重分配）
#        排序已完成，直接在排序后的数组上操作
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("步骤 3：PPS抽样（迭代截断与重分配）")
print("=" * 60)

n = SAMPLE_SIZE

if N <= n:
    print(f"[提示] 总企业数 {N} ≤ 目标样本量 {n}，全部选取。")
    df_sample = df_all.clone()
else:
    rng = np.random.default_rng(RANDOM_SEED)
    # 提取数组供 numpy 计算
    capital = df_all.get_column('注册资本_万元').to_numpy().copy()   # shape (N,)
    pos     = np.arange(N)                                         # 原始位置索引

    selected_certain = []
    remaining_pos    = pos.copy()
    remaining_cap    = capital.copy()
    remaining_n      = n
    itr = 0

    # 迭代截断：将概率≥1的单元直接入样，重分配剩余名额
    while remaining_n > 0 and len(remaining_pos) > 0:
        itr += 1
        total = remaining_cap.sum()
        probs = remaining_n * remaining_cap / total
        certain_mask = probs >= 1.0
        if not certain_mask.any():
            break
        certain_pos = remaining_pos[certain_mask]
        selected_certain.extend(certain_pos.tolist())
        remaining_n  -= len(certain_pos)
        keep = ~certain_mask
        remaining_pos = remaining_pos[keep]
        remaining_cap = remaining_cap[keep]

    print(f"迭代截断轮次：{itr}，必然入样企业数：{len(selected_certain):,}")

    # 对剩余单元按归一化概率无放回抽样
    selected_random = []
    if remaining_n > 0 and len(remaining_pos) > 0:
        total_rem  = remaining_cap.sum()
        probs_rem  = remaining_n * remaining_cap / total_rem
        probs_norm = probs_rem / probs_rem.sum()   # 归一化
        draw = rng.choice(
            len(remaining_pos),
            size=min(remaining_n, len(remaining_pos)),
            replace=False,
            p=probs_norm
        )
        selected_random = remaining_pos[draw].tolist()

    print(f"概率抽样企业数：{len(selected_random):,}")
    print(f"最终样本量：{len(selected_certain) + len(selected_random):,}")

    # 合并并按注册资本排序
    all_selected = sorted(set(selected_certain + selected_random))
    df_sample = df_all[all_selected]
    df_sample = df_sample.sort('注册资本_万元', descending=True)


# ─────────────────────────────────────────────
# 步骤 4：输出抽样结果
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("步骤 4：输出抽样结果")
print("=" * 60)

df_sample.with_row_index('序号', offset=1).write_csv(OUT_SMP_CSV, include_bom=True)
print(f"抽样样本已保存：{OUT_SMP_CSV}")
print(f"  共 {df_sample.height:,} 家企业（按注册资本降序）")

# 清理断点文件
if os.path.exists(PROGRESS_F):
    os.remove(PROGRESS_F)
if os.path.exists(TEMP_CSV):
    os.remove(TEMP_CSV)
print("\n临时文件已清理。完成！")