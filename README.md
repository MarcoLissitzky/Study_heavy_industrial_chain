# 重工业产业链数据工程（Study_heavy_industrial_chain）

本项目用于整理/抽样工商注册数据，并基于天眼查导出数据构建供需关系网络，输出统计报表与可视化结果。

## 目录结构（已完成分层）

- `data/raw/`：**原始输入**（不在此处生成新结果）
  - `registry_ne/`：东北工商注册原始 `xlsx`（已物理迁移到此）
  - `registry_cn_parquet/`：全国工商注册 `parquet`（已物理迁移到此，供后续使用）
  - `tyc_pps5000/`：天眼查导出原始文件（PPS5000）
  - `tyc_pps5000_amount_only/`：天眼查导出原始文件（金额筛选版）
  - `RAW_MIGRATION_NOTES.md`：raw 数据资产说明

- `data/processed/`：**处理中间结果**（可复算）
  - `supplychain_pps5000/`：主流程中间文件（如 `network/` 等）
  - `supplychain_pps5000_amount_only/`：金额筛选版中间文件（如 `node.csv`、`edge_*.csv`、`network/` 等）

- `outputs/`：**最终产物/展示交付**
  - `reports/`
    - `sampling/`：抽样/筛选类报表输出
    - `supplychain_pps5000/`：主流程报告输出
    - `supplychain_pps5000_amount_only/`：金额筛选版报告输出
  - `viz/`
    - `supplychain_pps5000/`：主流程可视化输出
    - `supplychain_pps5000_amount_only/`：金额筛选版可视化输出

- `scripts/`：数据处理脚本（已将**默认输出**切换到新架构）
- `logs/`：日志（建议将运行日志放此处）
- `backups/`：迁移前快照与清单（结构/脚本/配置备份）

> 说明：历史上 `database/` 下曾混放 raw/processed/outputs；现已拆分迁移到以上目录。

## 数据流（推荐）

1. raw 输入（`data/raw/*`）
2. processed 中间层（`data/processed/*`）
3. outputs（`outputs/reports/*`、`outputs/viz/*`）

## 常用脚本与用法

### 1) 东北重工业筛选 + PPS 抽样

默认模式（使用新目录作为默认输入/输出）：

```bash
python pps_sampling.py
```

若 `data/raw/registry_ne` 为空，但你想显式回退到旧目录（兼容模式）：

```bash
python pps_sampling.py --allow-fallback-on-empty --fallback-data-dir "database/工商企业注册信息（东北）"
```

常用参数：
- `--data-dir`：原始 xlsx 目录
- `--output-dir`：输出目录（默认 `outputs/reports/sampling`）
- `--sample-size`：样本量（默认 5000）

### 2) 构建供需/投资边（processed）

```bash
python scripts/build_edges.py --base-dir "data/raw/tyc_pps5000"
```

说明：
- 默认输出到 `data/processed/supplychain_pps5000`
- 可用 `--processed-dir` 或 `--out-dir` 显式覆盖输出目录（兼容模式）

### 3) 供需网络可视化（outputs/viz）

```bash
python scripts/viz_supplychain.py --base-dir "data/processed/supplychain_pps5000"
```

说明：
- 默认输出到 `outputs/viz/supplychain_pps5000`
- 可用 `--viz-dir` 覆盖输出目录

### 4) Top 节点报告（outputs/reports）

```bash
python scripts/report_supplychain_gt20.py --base-dir "data/processed/supplychain_pps5000"
```

说明：
- 默认读取 `outputs/viz/supplychain_pps5000/supplychain_node_metrics.csv`
- 默认输出到 `outputs/reports/supplychain_pps5000`
- 可用 `--metrics-csv` / `--reports-dir` 覆盖（兼容模式）

### 5) 金额覆盖率与分布分析（outputs/reports）

```bash
python scripts/analyze_amounts.py --base-dir "data/raw/tyc_pps5000_amount_only"
```

说明：
- 默认输出到 `outputs/reports/supplychain_pps5000_amount_only`
- 可用 `--output-dir` 覆盖

### 6) XLSX -> Parquet（全国工商转换/基准）

```bash
python scripts/xlsx_to_parquet_benchmark.py --input-dir "data/raw/registry_ne" --list-only
```

说明：
- 默认 `--out-dir` 为 `data/raw/registry_cn_parquet`
- 如果需要写到其他目录，可显式传 `--out-dir`

### 7) PyG 导出与无孤岛链路切分（GAT 第一模块）

```bash
python scripts/graph_pyg/export_pyg_supplychain_with_fringe.py --base-dir "data/processed/supplychain_pps5000" --val-ratio 0.1 --test-ratio 0.1 --split-seed 42
```

说明：
- 训练图使用“生成森林生命线边 + 富余边抽样”构建，保证训练图不产生孤岛。
- 输出新增 `split_edges.pt` 与 `split_stats.json`（目录：`data/processed/supplychain_pps5000/pyg`）。
- 旧入口 `scripts/export_pyg_supplychain_with_fringe.py` 保留为兼容转发。

### 8) GAT Encoder 训练入口（GAT 第二模块）

```bash
python scripts/graph_pyg/train_link_prediction.py --pyg-dir "data/processed/supplychain_pps5000/pyg" --epochs 5 --gat-dropout 0.2
```

说明：
- 训练脚本会先构建 `X_new:[N,617]`：`[注册资本log1p+MinMax(1), 双向PageRank+MinMax(1), one_hot_rest(615)]`。
- `PageRank` 只基于 `train_pos_edge_index_directed`，并采用双向策略（正向/反向 PR 取均值）。
- `GATConv` 两层实例化时显式传入 `dropout` 参数。
- Decoder 固定为最简 MLP（`Linear(32,16)->ReLU->Linear(16,1)`），训练损失固定为 `BCEWithLogitsLoss`。
- 训练负样本每个 epoch 动态重采样；验证/测试负样本使用固定切分产物。

## 兼容策略（重要）

- 旧目录不会自动删除，但**默认值**已切换到新架构。
- 任何脚本都可以通过显式参数（如 `--out-dir`、`--viz-dir`、`--reports-dir` 等）指向旧路径运行。

## 已知注意事项

- `outputs/reports/sampling/` 下的 `_temp_filtered.csv`、`_temp_progress.json` 是中断运行产生的临时文件；正常跑完会自动清理。

