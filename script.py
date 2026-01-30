import re
import pandas as pd
import numpy as np

WEEKS = 11
JUDGES = 4


def process_csv(input_path: str = "data.csv", output_path: str = "data_new.csv") -> None:
	# 读取原始CSV（全部按字符串读取，便于处理 'N/A'）
	df = pd.read_csv(input_path, dtype=str)

	# 1) 将 season 置于最左，并将 index 改为“对每个 season 从 1 开始计数”
	if "season" not in df.columns:
		raise KeyError("Missing 'season' column in input data")
	# 保持原有顺序分组计数
	df["index"] = df.groupby("season").cumcount() + 1

	# 准备并确保评委列存在（缺失则填充为 NaN）
	judge_cols = []
	for w in range(1, WEEKS + 1):
		for j in range(1, JUDGES + 1):
			col = f"week{w}_judge{j}_score"
			judge_cols.append(col)
			if col not in df.columns:
				df[col] = np.nan

	# 2) 计算每周4列的平均值（忽略 N/A / NaN；若4个全为N/A则平均为N/A），插入到最右端
	week_avg_cols = []
	for w in range(1, WEEKS + 1):
		cols = [f"week{w}_judge{j}_score" for j in range(1, JUDGES + 1)]
		# 转为数值，无法转换的（如 'N/A'）变为 NaN
		arr = df[cols].apply(pd.to_numeric, errors="coerce")
		avg = arr.mean(axis=1, skipna=True)
		avg_col = f"week{w}_avg"
		df[avg_col] = avg  # 暂存为浮点 + NaN（后续再统一转换为 'N/A'）
		week_avg_cols.append(avg_col)

	# 插入完平均数后，删除44列原始评委得分列
	df.drop(columns=[c for c in judge_cols if c in df.columns], inplace=True)

	# 3) 在最右端插入空白列 social_media_fans
	df["social_media_fans"] = ""

	# 4) 修改逻辑：last_active_week 为“最后一个平均数既不为 0 也不为 N/A 的周”
	# 若存在 results 列，仍重命名为 last_active_week，但其值由平均数重新计算覆盖
	if "results" in df.columns:
		df.rename(columns={"results": "last_active_week"}, inplace=True)
	else:
		df["last_active_week"] = np.nan

	def compute_last_active_week(row):
		last = np.nan
		for i, c in enumerate(week_avg_cols, start=1):
			val = row[c]
			# 统一为数值，无法转换（含 'N/A'）则为 NaN
			num = pd.to_numeric(val, errors="coerce")
			if not pd.isna(num) and num != 0:
				last = i
		return last

	df["last_active_week"] = df.apply(compute_last_active_week, axis=1)

	# 5) 随后将所有 0 平均数改为 N/A（不再按 last_active_week 截断之后的周）
	for c in week_avg_cols:
		num_series = pd.to_numeric(df[c], errors="coerce")
		zero_mask = num_series == 0
		df.loc[zero_mask, c] = np.nan

	# 将周平均列中的 NaN 转为 'N/A' 字符串（与需求一致）
	for c in week_avg_cols:
		df[c] = df[c].apply(lambda x: "N/A" if pd.isna(x) else x)

	# 若 placement 为 1，则将 last_active_week 记为 0；随后删除 placement 列
	if "placement" in df.columns:
		place_num = pd.to_numeric(df["placement"], errors="coerce")
		winner_mask = place_num == 1
		df.loc[winner_mask, "last_active_week"] = 0
		df.drop(columns=["placement"], inplace=True)

	# 重新排列列顺序：season 第一列，index 第二列，last_active_week 第三列，其余保持原顺序
	cols = list(df.columns)
	remaining = [c for c in cols if c not in ["season", "index", "last_active_week"]]
	df = df[["season", "index", "last_active_week"] + remaining]

	# 写出到新的文件（相当于先复制为 data_new.csv 再进行修改）
	df.to_csv(output_path, index=False)


if __name__ == "__main__":
	process_csv("data.csv", "data_new.csv")

