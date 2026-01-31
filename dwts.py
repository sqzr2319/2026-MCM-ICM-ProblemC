import argparse
import os
import re
from typing import Dict, List

import numpy as np
import pandas as pd


DWTS_COL = "Dancing with the Stars"


def process_file(file_path: str) -> pd.Series:
    """处理单个季节的趋势 CSV：
    1) 先对每一列进行纵向求和（跨所有行）
    2) 再将所有非 DWTS 列的列和除以 DWTS 列的列和
    返回：索引为名人列名、值为（该列和 / DWTS 列和）的 Series（不包含 DWTS 列）
    """
    df = pd.read_csv(file_path, dtype=str)
    if DWTS_COL not in df.columns:
        # 缺少 DWTS 列，返回空
        return pd.Series(dtype=float)

    # 转为数值
    num = df.apply(pd.to_numeric, errors="coerce")
    # 计算列和（跨所有行）
    col_sums = num.sum(axis=0, skipna=True)
    # DWTS 的总和作为分母
    try:
        denom_total = float(col_sums.get(DWTS_COL, np.nan))
    except Exception:
        denom_total = np.nan

    # 分母不可用或为 0 则跳过
    if not np.isfinite(denom_total) or denom_total == 0.0:
        return pd.Series(dtype=float)

    # 对每个非 DWTS 列：列和 / DWTS 列和
    ratio_items = {}
    for c in num.columns:
        if c == DWTS_COL:
            continue
        val = col_sums.get(c, np.nan)
        if np.isfinite(val):
            ratio_items[c] = float(val) / denom_total
    return pd.Series(ratio_items, dtype=float)


def parse_season_from_filename(fname: str) -> int:
    """Extract season number from filename, e.g., season_1_trends.csv -> 1."""
    m = re.search(r"season[_\-]?([0-9]+)", fname, re.IGNORECASE)
    return int(m.group(1)) if m else -1


def build_season_sums(folder: str) -> Dict[int, pd.Series]:
    """Aggregate sums for each season file in folder.
    Returns mapping: season -> Series (column sums)
    """
    results: Dict[int, pd.Series] = {}
    if not os.path.isdir(folder):
        return results

    for fname in os.listdir(folder):
        if not fname.lower().endswith(".csv"):
            continue
        season = parse_season_from_filename(fname)
        if season <= 0:
            continue
        series = process_file(os.path.join(folder, fname))
        if not series.empty:
            results[season] = series
    return results


def merge_into_data(data_csv: str, season_sums: Dict[int, pd.Series], write_backup: bool = True):
    """Merge aggregated sums into data_new.csv's social_media_fans by season and celebrity_name.
    For each season's Series, use column name as celebrity_name and value as social_media_fans.
    Skip the DWTS column.
    """
    df = pd.read_csv(data_csv, dtype=str)
    # Ensure numeric season
    df["season"] = pd.to_numeric(df["season"], errors="coerce")

    updated = 0
    for season, series in season_sums.items():
        for col_name, val in series.items():
            if col_name == DWTS_COL:
                continue
            # Only process finite values
            try:
                num_val = float(val)
            except Exception:
                continue
            if not np.isfinite(num_val):
                continue
            # Match rows in main data
            mask = (df["season"].astype(float) == float(season)) & (df["celebrity_name"].astype(str) == str(col_name))
            if mask.any():
                df.loc[mask, "social_media_fans"] = str(num_val)
                updated += int(mask.sum())

    # Optional backup
    if write_backup:
        backup_path = os.path.splitext(data_csv)[0] + ".bak.csv"
        df.to_csv(backup_path, index=False)
    df.to_csv(data_csv, index=False)
    print(f"已更新 social_media_fans 共 {updated} 行，写回 {data_csv}")


def main():
    parser = argparse.ArgumentParser(description="Process dwts_results and merge into data_new.csv")
    parser.add_argument("--folder", default="dwts_results", help="dwts_results 文件夹路径")
    parser.add_argument("--data", default="data_new.csv", help="目标数据文件路径")
    args = parser.parse_args()

    season_sums = build_season_sums(args.folder)
    if not season_sums:
        print(f"未在 {args.folder} 找到可用的季节CSV，跳过。")
        return
    merge_into_data(args.data, season_sums)


if __name__ == "__main__":
    main()
