# 数据处理脚本使用说明

本脚本按需求处理 `data.csv` 并生成 `data_new.csv`。

## 环境准备

```powershell
python -m pip install -r requirements.txt
# 或
py -m pip install -r requirements.txt
```

## 运行脚本

```powershell
python script.py
# 或
py script.py
```

## 逻辑概要
## 逻辑概要
- 输出到 `data_new.csv`。
- 列顺序：
	- 左起第一列为 `season`；
	- 第二列为 `index`（对每个 `season` 从 1 开始计数）；
	- 第三列为 `last_active_week`；
	- 之后为各字段与 `week{w}_avg` 列，最后 `social_media_fans`。
- 每周 4 列评分取平均（忽略 `N/A`），生成 `week{w}_avg` 共 11 周，随后删除 44 列原始评分。
- 在最右侧插入空白列 `social_media_fans`。
- `last_active_week`：为“最后一个平均数既不为 0 也不为 `N/A` 的周”。
- 若某行 `placement` 为 1，则该行 `last_active_week` 记为 0（winner），随后删除 `placement` 列。
- 将所有周平均中的 0 值统一改为 `N/A`。
