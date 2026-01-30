import argparse
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import os

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import torch
    from torch import nn
except Exception as e:
    raise RuntimeError(
        "PyTorch 未安装或不可用。请先安装：py -m pip install torch --index-url https://download.pytorch.org/whl/cpu"
    )


@dataclass
class Contestant:
    season: int
    index: int  # per-season unique index
    row_idx: int  # row position in dataframe
    name: str


@dataclass
class Event:
    season: int
    week: int
    method: str  # 'percent' or 'rank'
    active_ids: List[int]  # global contestant ids
    J_norm: torch.Tensor  # shape [A]
    fans_norm: torch.Tensor  # shape [A]
    eliminated_mask: torch.Tensor  # shape [A], dtype=bool


def build_dataset(df: pd.DataFrame, weeks: int = 11) -> Tuple[List[Contestant], List[Event]]:
    # Map each (season, index) to global id
    contestants: List[Contestant] = []
    id_map: Dict[Tuple[int, int], int] = {}
    for r in range(len(df)):
        s = int(df.loc[r, "season"]) if not pd.isna(df.loc[r, "season"]) else 0
        idx = int(df.loc[r, "index"]) if not pd.isna(df.loc[r, "index"]) else r + 1
        name = str(df.loc[r, "celebrity_name"]) if "celebrity_name" in df.columns else f"{s}-{idx}"
        gid = len(contestants)
        contestants.append(Contestant(season=s, index=idx, row_idx=r, name=name))
        id_map[(s, idx)] = gid

    # Fans: 线性归一化（global min-max 到 [0,1]）
    fans_raw = pd.to_numeric(df.get("social_media_fans", pd.Series([np.nan] * len(df))), errors="coerce")
    fans_raw = fans_raw.fillna(0.0)
    fans_vals = np.clip(fans_raw.values.astype(float), a_min=0.0, a_max=None)
    fmin = float(np.min(fans_vals)) if fans_vals.size > 0 else 0.0
    fmax = float(np.max(fans_vals)) if fans_vals.size > 0 else 1.0
    if fmax > fmin:
        fans_norm_global = (fans_vals - fmin) / (fmax - fmin)
    else:
        fans_norm_global = np.zeros_like(fans_vals)

    # Build events per season-week
    events: List[Event] = []
    for season in sorted(df["season"].dropna().astype(int).unique().tolist()):
        # choose method: 3-27 percent, else rank
        method = "percent" if 3 <= season <= 27 else "rank"
        for week in range(1, weeks + 1):
            col = f"week{week}_avg"
            if col not in df.columns:
                continue
            # Active rows: numeric J is not NaN and not 'N/A'
            J_series = pd.to_numeric(df[col], errors="coerce")
            active_rows = df.index[(df["season"].astype(int) == season) & (~J_series.isna())].tolist()
            if len(active_rows) == 0:
                continue
            J_vals = J_series.loc[active_rows].values.astype(float)
            # Min-max normalize to [0,1]
            jmin = J_vals.min()
            jmax = J_vals.max()
            if jmax > jmin:
                J_norm = (J_vals - jmin) / (jmax - jmin)
            else:
                J_norm = np.zeros_like(J_vals)
            # fans for active
            active_ids = [id_map[(season, int(df.loc[r, "index"]))] for r in active_rows]
            fans_norm = fans_norm_global[[contestants[gid].row_idx for gid in active_ids]]
            # eliminated mask from last_active_week
            last_w = pd.to_numeric(df.loc[active_rows, "last_active_week"], errors="coerce")
            eliminated_mask = (last_w.values.astype(float) == float(week))
            # If no elimination this week, still include event for gradients from method coupling
            # but loss will be zero for this event
            events.append(
                Event(
                    season=season,
                    week=week,
                    method=method,
                    active_ids=active_ids,
                    J_norm=torch.tensor(J_norm, dtype=torch.float32),
                    fans_norm=torch.tensor(fans_norm, dtype=torch.float32),
                    eliminated_mask=torch.tensor(eliminated_mask, dtype=torch.bool),
                )
            )
    return contestants, events


class VotingModel(nn.Module):
    def __init__(self, num_contestants: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(num_contestants))
        # 方法专用参数：percent 与 rank 各自独立的 delta1/delta2
        self.delta1_percent = nn.Parameter(torch.tensor(0.1))
        self.delta2_percent = nn.Parameter(torch.tensor(0.0))
        self.delta1_rank = nn.Parameter(torch.tensor(0.1))
        self.delta2_rank = nn.Parameter(torch.tensor(0.0))
        self.tau = 10.0

    def forward_event(self, event: Event) -> torch.Tensor:
        # P = gamma + delta1 * J + delta2 * log(F)
        g = self.gamma[event.active_ids]
        if event.method == "percent":
            d1 = self.delta1_percent
            d2 = self.delta2_percent
        else:
            d1 = self.delta1_rank
            d2 = self.delta2_rank
        P = g + d1 * event.J_norm + d2 * event.fans_norm
        # V 线性归一化：将 P 平移到非负后按和为1归一
        P_shift = P - torch.min(P)
        denom = torch.sum(P_shift)
        if denom.item() > 1e-8:
            V = P_shift / denom
        else:
            V = torch.full_like(P_shift, 1.0 / P_shift.numel())
        if event.method == "percent":
            J_sum = torch.clamp(event.J_norm.sum(), min=1e-8)
            J_pct = event.J_norm / J_sum
            C = J_pct + V
            probs = torch.softmax(-self.tau * C, dim=0)
        else:
            # rank approximation by pairwise sigmoid comparisons
            J_diff = event.J_norm.unsqueeze(0) - event.J_norm.unsqueeze(1)
            V_diff = V.unsqueeze(0) - V.unsqueeze(1)
            sigma_J = torch.sigmoid(J_diff)
            sigma_V = torch.sigmoid(V_diff)
            # exclude self-comparisons by subtracting diagonal contribution (0.5)
            rank_J = 1.0 + (sigma_J.sum(dim=1) - torch.diag(sigma_J))
            rank_V = 1.0 + (sigma_V.sum(dim=1) - torch.diag(sigma_V))
            R_tilde = rank_J + rank_V
            probs = torch.softmax(self.tau * R_tilde, dim=0)
        return probs


def evaluate_model(model: nn.Module, events: List[Event], contestants: List[Contestant]):
    # 加权准确率 + 决赛加分；并收集绘图所需的逐季记录
    correct_total, eliminated_total = 0, 0
    season_final_week: Dict[int, int] = {}
    season_final_fullmatch: Dict[int, bool] = {}
    # 逐季的方格数据：每项为 {correct: bool, week: int, name: str, is_champion: bool}
    per_season_squares: Dict[int, List[Dict]] = {}

    for ev in events:
        probs = model.forward_event(ev).detach().cpu().numpy()
        actual = ev.eliminated_mask.detach().cpu().numpy()
        k = int(actual.sum())
        if k == 0:
            continue
        pred_topk = set(np.argsort(-probs)[:k].tolist())
        actual_idx = set(np.where(actual)[0].tolist())
        inter = pred_topk.intersection(actual_idx)
        event_correct = len(inter)
        correct_total += event_correct
        eliminated_total += k

        # 记录赛季方格（每个被淘汰的选手一个格）
        s = ev.season
        if s not in per_season_squares:
            per_season_squares[s] = []
        for idx in actual_idx:
            gid = ev.active_ids[idx]
            name = contestants[gid].name
            per_season_squares[s].append({
                "correct": (idx in inter),
                "week": ev.week,
                "name": name,
                "is_champion": False,
            })

        # 更新决赛标记
        if (s not in season_final_week) or (ev.week >= season_final_week[s]):
            season_final_week[s] = ev.week
            season_final_fullmatch[s] = (event_correct == k)

    finals_count = len(season_final_week)
    bonus_correct = sum(1 for s, full in season_final_fullmatch.items() if full)
    target_total = eliminated_total + finals_count
    accuracy = ((correct_total + bonus_correct) / target_total) if target_total > 0 else 0.0

    # 为每季添加冠军方格（放在最后）
    for s, last_w in season_final_week.items():
        if s not in per_season_squares:
            per_season_squares[s] = []
        per_season_squares[s].append({
            "correct": season_final_fullmatch.get(s, False),
            "week": last_w + 1,
            "name": "Champion",
            "is_champion": True,
        })

    return {
        "accuracy": accuracy,
        "correct_total": int(correct_total),
        "bonus_correct": int(bonus_correct),
        "eliminated_total": int(eliminated_total),
        "finals_count": int(finals_count),
        "target_total": int(target_total),
        "per_season_squares": per_season_squares,
    }


def plot_accuracy_grid(per_season_squares: Dict[int, List[Dict]], out_path: str):
    if plt is None:
        print("未安装 matplotlib，跳过绘图。")
        return
    seasons = sorted(per_season_squares.keys())
    # 每行的列数取该季的参赛人数（淘汰人数 + 1 冠军）
    row_lengths = {s: len(per_season_squares[s]) for s in seasons}
    max_len = max(row_lengths.values()) if row_lengths else 0
    if max_len == 0:
        print("没有可绘制的数据。")
        return

    fig_w = max_len * 0.35 + 2
    fig_h = len(seasons) * 0.35 + 2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, max_len)
    ax.set_ylim(0, len(seasons))
    ax.set_aspect('equal')
    ax.axis('off')

    for row_idx, s in enumerate(seasons):
        # 按 week 升序排，冠军放最后（week 已是最后周+1）
        cells = sorted(per_season_squares[s], key=lambda d: (d["week"], 0 if d["is_champion"] else -1))
        for col_idx, cell in enumerate(cells):
            color = '#4CAF50' if cell["correct"] else '#F44336'
            rect = plt.Rectangle((col_idx, len(seasons)-1-row_idx), 0.9, 0.9, facecolor=color, edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
            label = f"Wk {cell['week']}" if not cell["is_champion"] else "Champ"
            ax.text(col_idx+0.45, len(seasons)-1-row_idx+0.45, label, ha='center', va='center', fontsize=6, color='white')
        # 标注赛季号
        ax.text(-0.5, len(seasons)-1-row_idx+0.45, f"S{s}", ha='right', va='center', fontsize=8)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def train(data_path: str, out_dir: str, epochs: int = 200, lr: float = 0.05, weeks: int = 11):
    df = pd.read_csv(data_path, dtype=str)
    # Ensure numeric types for season/index
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["index"] = pd.to_numeric(df["index"], errors="coerce")
    df["last_active_week"] = pd.to_numeric(df["last_active_week"], errors="coerce")

    contestants, events = build_dataset(df, weeks=weeks)
    model = VotingModel(num_contestants=len(contestants))
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    best_acc = -1.0
    best_params_path = os.path.join(out_dir, "best_params.json")
    best_grid_path = os.path.join(out_dir, "best_accuracy_grid.png")
    os.makedirs(out_dir, exist_ok=True)
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        loss = torch.tensor(0.0)
        for ev in events:
            probs = model.forward_event(ev)
            if ev.eliminated_mask.any():
                # Negative log likelihood for eliminated contestants
                elim_probs = probs[ev.eliminated_mask]
                # Guard against zeros
                loss = loss - torch.log(torch.clamp(elim_probs, min=1e-8)).sum()
        loss.backward()
        opt.step()
        history.append(float(loss.item()))
        if ep % 10 == 0 or ep == 1:
            print(
                "epoch={} loss={:.4f} d1_pct={:.4f} d2_pct={:.4f} d1_rank={:.4f} d2_rank={:.4f}".format(
                    ep,
                    loss.item(),
                    model.delta1_percent.item(),
                    model.delta2_percent.item(),
                    model.delta1_rank.item(),
                    model.delta2_rank.item(),
                )
            )
            eval_res = evaluate_model(model, events, contestants)
            acc = eval_res["accuracy"]
            print(
                f"eval@epoch={ep}: accuracy={acc:.4f} (correct={eval_res['correct_total']}+bonus={eval_res['bonus_correct']} / target={eval_res['target_total']} [elim={eval_res['eliminated_total']}+finals={eval_res['finals_count']}])"
            )
            if acc > best_acc:
                best_acc = acc
                params_best = {
                    "delta1_percent": float(model.delta1_percent.item()),
                    "delta2_percent": float(model.delta2_percent.item()),
                    "delta1_rank": float(model.delta1_rank.item()),
                    "delta2_rank": float(model.delta2_rank.item()),
                    "gamma": {f"{c.season}-{c.index}-{c.name}": float(model.gamma[i].item()) for i, c in enumerate(contestants)},
                    "accuracy": float(acc),
                }
                with open(best_params_path, "w", encoding="utf-8") as f:
                    json.dump(params_best, f, ensure_ascii=False, indent=2)
                plot_accuracy_grid(eval_res["per_season_squares"], best_grid_path)
                print(f"best checkpoint saved: acc={best_acc:.4f}, params={best_params_path}, grid={best_grid_path}")

    # Save artifacts
    params = {
        "delta1_percent": float(model.delta1_percent.item()),
        "delta2_percent": float(model.delta2_percent.item()),
        "delta1_rank": float(model.delta1_rank.item()),
        "delta2_rank": float(model.delta2_rank.item()),
        "gamma": {f"{c.season}-{c.index}-{c.name}": float(model.gamma[i].item()) for i, c in enumerate(contestants)},
    }
    with open(os.path.join(out_dir, "params.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    # Optionally dump per-event predicted elimination probabilities
    rows = []
    for ev in events:
        probs = model.forward_event(ev).detach().numpy()
        for k, gid in enumerate(ev.active_ids):
            c = contestants[gid]
            rows.append({
                "season": c.season,
                "week": ev.week,
                "contestant_index": c.index,
                "name": c.name,
                "prob_eliminated": float(probs[k]),
            })
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "event_probs.csv"), index=False)

    pd.DataFrame({"loss": history}).to_csv(os.path.join(out_dir, "loss.csv"), index=False)
    # 最终评估与写出指标与最终网格
    eval_final = evaluate_model(model, events, contestants)
    accuracy = eval_final["accuracy"]
    metrics = {
        "accuracy": accuracy,
        "correct_total": eval_final["correct_total"],
        "bonus_correct": eval_final["bonus_correct"],
        "eliminated_total": eval_final["eliminated_total"],
        "finals_count": eval_final["finals_count"],
        "target_total": eval_final["target_total"],
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    plot_accuracy_grid(eval_final["per_season_squares"], os.path.join(out_dir, "accuracy_grid.png"))

    print(f"训练完成。参数与输出保存在 {out_dir}")
    print(
        f"验证指标：accuracy={accuracy:.4f} (correct={metrics['correct_total']}+bonus={metrics['bonus_correct']} / target={metrics['target_total']} [elim={metrics['eliminated_total']}+finals={metrics['finals_count']}])"
    )


def main():
    parser = argparse.ArgumentParser(description="Train voting model per model.md")
    parser.add_argument("--data", default="data_new.csv", help="输入数据 CSV 路径")
    parser.add_argument("--out", default="artifacts", help="输出目录")
    parser.add_argument("--epochs", type=int, default=300, help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.05, help="学习率")
    parser.add_argument("--weeks", type=int, default=11, help="最大周数")
    args = parser.parse_args()

    train(args.data, args.out, epochs=args.epochs, lr=args.lr, weeks=args.weeks)


if __name__ == "__main__":
    main()
