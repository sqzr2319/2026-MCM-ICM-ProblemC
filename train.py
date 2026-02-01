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
    is_final_week: bool  # 是否是该季最后一次发生淘汰的周


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

    # Fans: 使用当周活跃选手的总和归一化（非负）
    fans_raw = pd.to_numeric(df.get("social_media_fans", pd.Series([np.nan] * len(df))), errors="coerce")
    fans_raw = fans_raw.fillna(0.0)
    fans_vals_all = np.clip(fans_raw.values.astype(float), a_min=0.0, a_max=None)

    # 计算每季的“决赛周”（最后一次发生淘汰的周）
    finals_week_map: Dict[int, int] = {}
    for season in sorted(df["season"].dropna().astype(int).unique().tolist()):
        lw = pd.to_numeric(df.loc[df["season"].astype(int) == season, "last_active_week"], errors="coerce")
        lw = lw.dropna().astype(int)
        lw = lw[lw > 0]
        finals_week_map[season] = int(lw.max()) if not lw.empty else 0

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
            # 评委分按当周活跃选手的总和归一化
            J_sum = float(np.sum(J_vals))
            if J_sum > 0:
                J_norm = (J_vals / J_sum)
            else:
                J_norm = np.zeros_like(J_vals)
            # 粉丝热度按当周活跃选手的总和归一化（非负）
            active_ids = [id_map[(season, int(df.loc[r, "index"]))] for r in active_rows]
            fans_active = fans_vals_all[[contestants[gid].row_idx for gid in active_ids]]
            F_sum = float(np.sum(fans_active))
            if F_sum > 0:
                fans_norm = (fans_active / F_sum)
            else:
                fans_norm = np.zeros_like(fans_active)
            # eliminated mask from last_active_week
            last_w = pd.to_numeric(df.loc[active_rows, "last_active_week"], errors="coerce")
            eliminated_mask = (last_w.values.astype(float) == float(week))
            is_final_week = (week == finals_week_map.get(season, 0))
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
                    is_final_week=is_final_week,
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
        self.tau = 60.0

    def forward_event(self, event: Event) -> torch.Tensor:
        # P = gamma + delta1 * J_norm + delta2 * F_norm；V 使用 softmax
        g = self.gamma[event.active_ids]
        if event.method == "percent":
            d1 = self.delta1_percent
            d2 = self.delta2_percent
        else:
            d1 = self.delta1_rank
            d2 = self.delta2_rank
        P = g + d1 * event.J_norm + d2 * event.fans_norm
        # softmax 百分比（tau1=1）
        V = torch.softmax(P, dim=0)
        if event.method == "percent":
            J_sum = torch.clamp(event.J_norm.sum(), min=1e-8)
            J_pct = event.J_norm / J_sum
            C = J_pct + V
            probs = torch.softmax(-self.tau * C, dim=0)
        else:
            # rank approximation by pairwise sigmoid comparisons
            J_diff = event.J_norm.unsqueeze(0) - event.J_norm.unsqueeze(1)
            V_diff = V.unsqueeze(0) - V.unsqueeze(1)
            # 引入 beta 因子，并按文档用 \sum J 进行归一化
            J_sum = torch.clamp(event.J_norm.sum(), min=1e-8)
            sigma_J = torch.sigmoid(J_diff / J_sum)
            sigma_V = torch.sigmoid(V_diff)
            # exclude self-comparisons by subtracting diagonal contribution (0.5)
            rank_J = 1.0 + (sigma_J.sum(dim=1) - torch.diag(sigma_J))
            rank_V = 1.0 + (sigma_V.sum(dim=1) - torch.diag(sigma_V))
            R_tilde = rank_J + rank_V
            # 默认排名法概率
            probs_rank = torch.softmax(self.tau * R_tilde, dim=0)
            # 二次选择机制：Season 28-34，非决赛周且淘汰人数为1
            if (28 <= event.season <= 34) and (not event.is_final_week):
                k = int(event.eliminated_mask.sum().item())
                if k == 1:
                    # 数值稳定版本：基于 probs_rank 计算 Bottom2 联合概率
                    P = probs_rank
                    eps = 1e-8
                    P = torch.clamp(P, min=eps, max=1.0 - eps)
                    sum_w = torch.tensor(1.0, dtype=P.dtype, device=P.device)
                    A = P.shape[0]
                    # 评委条件淘汰概率：sigma(beta * (J_j - J_i) / sum J)
                    J_row = event.J_norm.view(-1, 1)  # [A,1]
                    J_col = event.J_norm.view(1, -1)  # [1,A]
                    judge_elim = torch.sigmoid((J_col - J_row) / J_sum)  # [A,A]
                    # 计算 P(i,j in Bottom2)
                    # p_first(i) = P[i]; p_second_given_i(j) = P[j]/(1 - P[i])
                    # 对称项相加（j 先、i 后）
                    probs_final = torch.zeros_like(R_tilde)
                    for i in range(A):
                        Pi = P[i]
                        denom_i = torch.clamp(1.0 - Pi, min=eps)
                        p_first_i = Pi
                        # i 作为第一个，j 作为第二个（剔除 i）
                        p_second_given_i = P / denom_i  # [A]
                        p_second_given_i[i] = 0.0
                        term1 = p_first_i * p_second_given_i  # [A]
                        # 对称项：j 作为第一个，i 作为第二个
                        p_first_all = P.clone()  # [A]
                        p_first_all[i] = 0.0
                        denom_j = torch.clamp(1.0 - P, min=eps)  # [A]
                        p_second_given_j_i = Pi / denom_j  # [A]
                        p_second_given_j_i[i] = 0.0
                        term2 = p_first_all * p_second_given_j_i  # [A]
                        pair_prob = term1 + term2  # [A]
                        # 累加淘汰 i 的最终概率：sum_{j != i} P_pair(i,j) * judge_elim[i,j]
                        probs_final[i] = (pair_prob * judge_elim[i]).sum()
                    probs = probs_final
                else:
                    probs = probs_rank
            else:
                probs = probs_rank
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
    # 两列布局：将赛季分为左右两列，每列最多 ~17 行，避免整图过长
    if plt is None:
        print("未安装 matplotlib，跳过绘图。")
        return
    seasons = sorted(per_season_squares.keys())
    if not seasons:
        print("没有可绘制的数据。")
        return
    # 每行的列数取该季的参赛人数（淘汰人数 + 1 冠军）
    row_lengths = {s: len(per_season_squares[s]) for s in seasons}
    max_len = max(row_lengths.values()) if row_lengths else 0
    if max_len == 0:
        print("没有可绘制的数据。")
        return

    # 划分两列
    n = len(seasons)
    rows_per_col = int(np.ceil(n / 2))
    left_seasons = seasons[:rows_per_col]
    right_seasons = seasons[rows_per_col:]

    # 画布大小：两列并排，行高按每列行数决定
    fig_w = 2 * (max_len * 0.35 + 2.0)
    fig_h = rows_per_col * 0.35 + 2.0
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))
    axes = np.array(axes).reshape(1, 2)

    def draw_column(ax, seasons_col: List[int]):
        ax.set_xlim(0, max_len)
        ax.set_ylim(0, len(seasons_col))
        ax.set_aspect('equal')
        ax.axis('off')
        for row_idx, s in enumerate(seasons_col):
            # 按 week 升序排，冠军放最后（week 已是最后周+1）
            cells = sorted(per_season_squares[s], key=lambda d: (d["week"], 0 if d["is_champion"] else -1))
            for col_idx, cell in enumerate(cells):
                color = '#4CAF50' if cell["correct"] else '#F44336'
                rect = plt.Rectangle((col_idx, len(seasons_col)-1-row_idx), 0.9, 0.9, facecolor=color, edgecolor='black', linewidth=0.5)
                ax.add_patch(rect)
                label = f"Wk {cell['week']}" if not cell["is_champion"] else "Champ"
                ax.text(col_idx+0.45, len(seasons_col)-1-row_idx+0.45, label, ha='center', va='center', fontsize=6, color='white')
            # 标注赛季号
            ax.text(-0.5, len(seasons_col)-1-row_idx+0.45, f"S{s}", ha='right', va='center', fontsize=8)

    draw_column(axes[0][0], left_seasons)
    # 右列可能为空（当季数 <= rows_per_col）
    if right_seasons:
        draw_column(axes[0][1], right_seasons)
    else:
        axes[0][1].axis('off')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def set_model_params_from_best(model: VotingModel, contestants: List[Contestant], best_params: Dict):
    # 支持旧/新键名：若不存在 percent/rank 专用参数，退化为共享
    d1p = best_params.get("delta1_percent", best_params.get("delta1", 0.1))
    d2p = best_params.get("delta2_percent", best_params.get("delta2", 0.0))
    d1r = best_params.get("delta1_rank", best_params.get("delta1", 0.1))
    d2r = best_params.get("delta2_rank", best_params.get("delta2", 0.0))
    with torch.no_grad():
        model.delta1_percent.copy_(torch.tensor(float(d1p)))
        model.delta2_percent.copy_(torch.tensor(float(d2p)))
        model.delta1_rank.copy_(torch.tensor(float(d1r)))
        model.delta2_rank.copy_(torch.tensor(float(d2r)))
        # gamma 映射："season-index-name" 或 "season-index"
        gamma_map = best_params.get("gamma", {})
        gamma_vec = torch.zeros_like(model.gamma)
        for i, c in enumerate(contestants):
            key1 = f"{c.season}-{c.index}-{c.name}"
            key2 = f"{c.season}-{c.index}"
            val = gamma_map.get(key1, gamma_map.get(key2, 0.0))
            gamma_vec[i] = float(val)
        model.gamma.copy_(gamma_vec)


def compute_event_V(model: VotingModel, event: Event) -> torch.Tensor:
    # 仅计算该事件的 V（与二次选择无关）
    g = model.gamma[event.active_ids]
    if event.method == "percent":
        d1 = model.delta1_percent
        d2 = model.delta2_percent
    else:
        d1 = model.delta1_rank
        d2 = model.delta2_rank
    P = g + d1 * event.J_norm + d2 * event.fans_norm
    V = torch.softmax(P, dim=0)
    return V


def compute_rank_probs_with_two_stage(model: VotingModel, event: Event) -> torch.Tensor:
    # 强制采用排名法+二次选择（若非决赛且淘汰1人），不依赖季节范围
    g = model.gamma[event.active_ids]
    d1 = model.delta1_rank
    d2 = model.delta2_rank
    P = g + d1 * event.J_norm + d2 * event.fans_norm
    # V 使用 softmax
    V = torch.softmax(P, dim=0)
    # 排名近似
    J_diff = event.J_norm.unsqueeze(0) - event.J_norm.unsqueeze(1)
    V_diff = V.unsqueeze(0) - V.unsqueeze(1)
    J_sum = torch.clamp(event.J_norm.sum(), min=1e-8)
    sigma_J = torch.sigmoid(J_diff / J_sum)
    sigma_V = torch.sigmoid(V_diff)
    rank_J = 1.0 + (sigma_J.sum(dim=1) - torch.diag(sigma_J))
    rank_V = 1.0 + (sigma_V.sum(dim=1) - torch.diag(sigma_V))
    R_tilde = rank_J + rank_V
    probs_rank = torch.softmax(model.tau * R_tilde, dim=0)
    # 二次选择：仅在非决赛且淘汰1人时应用
    k = int(event.eliminated_mask.sum().item())
    if (not event.is_final_week) and (k == 1):
        Pp = probs_rank
        eps = 1e-8
        Pp = torch.clamp(Pp, min=eps, max=1.0 - eps)
        A = Pp.shape[0]
        judge_elim = torch.sigmoid((event.J_norm.view(1, -1) - event.J_norm.view(-1, 1)) / J_sum)
        probs_final = torch.zeros_like(R_tilde)
        for i in range(A):
            Pi = Pp[i]
            denom_i = torch.clamp(1.0 - Pi, min=eps)
            p_first_i = Pi
            p_second_given_i = Pp / denom_i
            p_second_given_i[i] = 0.0
            term1 = p_first_i * p_second_given_i
            p_first_all = Pp.clone(); p_first_all[i] = 0.0
            denom_j = torch.clamp(1.0 - Pp, min=eps)
            p_second_given_j_i = Pi / denom_j
            p_second_given_j_i[i] = 0.0
            term2 = p_first_all * p_second_given_j_i
            pair_prob = term1 + term2
            probs_final[i] = (pair_prob * judge_elim[i]).sum()
        return probs_final
    else:
        return probs_rank


def compute_fisher_diag(model: VotingModel, events: List[Event]) -> Dict[str, torch.Tensor]:
    # 经验 Fisher：逐事件累加 (grad log p)^2 的和，近似 Hessian 对角
    model.zero_grad()
    fisher_gamma = torch.zeros_like(model.gamma)
    fisher_d1p = torch.tensor(0.0)
    fisher_d2p = torch.tensor(0.0)
    fisher_d1r = torch.tensor(0.0)
    fisher_d2r = torch.tensor(0.0)
    eps = 1e-8
    for ev in events:
        probs = model.forward_event(ev)
        if not ev.eliminated_mask.any():
            continue
        elim_probs = probs[ev.eliminated_mask]
        loss_ev = -torch.log(torch.clamp(elim_probs, min=eps)).sum()
        model.zero_grad()
        loss_ev.backward(retain_graph=True)
        # 累加梯度平方
        if model.gamma.grad is not None:
            fisher_gamma += (model.gamma.grad ** 2)
        if model.delta1_percent.grad is not None:
            fisher_d1p = fisher_d1p + model.delta1_percent.grad.pow(2)
        if model.delta2_percent.grad is not None:
            fisher_d2p = fisher_d2p + model.delta2_percent.grad.pow(2)
        if model.delta1_rank.grad is not None:
            fisher_d1r = fisher_d1r + model.delta1_rank.grad.pow(2)
        if model.delta2_rank.grad is not None:
            fisher_d2r = fisher_d2r + model.delta2_rank.grad.pow(2)
    # 计算对角协方差的倒数：加稳健项避免 0
    fisher_gamma = torch.clamp(fisher_gamma, min=eps)
    fisher_d1p = torch.clamp(fisher_d1p, min=eps)
    fisher_d2p = torch.clamp(fisher_d2p, min=eps)
    fisher_d1r = torch.clamp(fisher_d1r, min=eps)
    fisher_d2r = torch.clamp(fisher_d2r, min=eps)
    return {
        "gamma": fisher_gamma,
        "d1p": fisher_d1p,
        "d2p": fisher_d2p,
        "d1r": fisher_d1r,
        "d2r": fisher_d2r,
    }


def sd_delta_method(model: VotingModel, events: List[Event], fisher_diag: Dict[str, torch.Tensor], damp: float = 0.0):
    # 使用 Delta 方法：Var(V_a) ≈ sum_j (∂V_a/∂θ_j)^2 Var(θ_j)，SD = sqrt(Var)
    eps = 1e-12
    # 阻尼逆协方差：(H + λI)^{-1}，在对角近似下等价为 1/(Fisher_diag + λ)
    lam = float(max(damp, 0.0))
    var_gamma = 1.0 / (torch.clamp(fisher_diag["gamma"], min=eps) + lam)
    var_d1p = float((1.0 / (torch.clamp(fisher_diag["d1p"], min=eps) + lam)).item())
    var_d2p = float((1.0 / (torch.clamp(fisher_diag["d2p"], min=eps) + lam)).item())
    var_d1r = float((1.0 / (torch.clamp(fisher_diag["d1r"], min=eps) + lam)).item())
    var_d2r = float((1.0 / (torch.clamp(fisher_diag["d2r"], min=eps) + lam)).item())

    sd_map: Dict[Tuple[int, int], np.ndarray] = {}
    meanV_map: Dict[Tuple[int, int], np.ndarray] = {}
    for ev in events:
        # 均值与梯度均使用 softmax 的 V
        V_true = compute_event_V(model, ev)
        meanV = V_true.detach().cpu().numpy()
        # 构建 torch 计算图：softmax
        g = model.gamma[ev.active_ids]
        if ev.method == "percent":
            d1_param = model.delta1_percent
            d2_param = model.delta2_percent
            var_d1 = var_d1p
            var_d2 = var_d2p
        else:
            d1_param = model.delta1_rank
            d2_param = model.delta2_rank
            var_d1 = var_d1r
            var_d2 = var_d2r
        P = g + d1_param * ev.J_norm + d2_param * ev.fans_norm
        V_soft = torch.softmax(P, dim=0)
        sd_vals = np.zeros_like(meanV)
        # Fisher 对角：转为 numpy 以做索引
        var_gamma_np = (1.0 / (torch.clamp(fisher_diag["gamma"], min=eps) + lam)).detach().cpu().numpy()
        for a in range(len(ev.active_ids)):
            model.zero_grad()
            grads = torch.autograd.grad(V_soft[a], [model.gamma, d1_param, d2_param], retain_graph=True, allow_unused=True)
            # 提取 gamma 的梯度并限制到 active_ids
            grad_gamma_all = grads[0].detach().cpu().numpy() if grads[0] is not None else np.zeros_like(var_gamma_np)
            grad_gamma_active = grad_gamma_all[ev.active_ids]
            var_v = float(np.sum((grad_gamma_active ** 2) * var_gamma_np[ev.active_ids]))
            grad_d1 = float(grads[1].item()) if grads[1] is not None else 0.0
            grad_d2 = float(grads[2].item()) if grads[2] is not None else 0.0
            var_v += (grad_d1 ** 2) * var_d1
            var_v += (grad_d2 ** 2) * var_d2
            sd_v = float(np.sqrt(max(var_v, eps)))
            sd_vals[a] = sd_v
        sd_map[(ev.season, ev.week)] = sd_vals
        meanV_map[(ev.season, ev.week)] = meanV
    return sd_map, meanV_map
    # 旧 RSD 方法已弃用


def plot_sd_panels(df: pd.DataFrame, contestants: List[Contestant], events: List[Event], sd_map: Dict[Tuple[int, int], np.ndarray], out_path: str):
    if plt is None:
        print("未安装 matplotlib，跳过 SD 绘图。")
        return
    seasons = sorted(df["season"].dropna().astype(int).unique().tolist())
    # 计算每季选手的位次（按淘汰周升序，冠军最后）
    placement_by_season: Dict[int, List[Tuple[int, int]]] = {}
    for s in seasons:
        rows = df.index[df["season"].astype(int) == s].tolist()
        last_w = pd.to_numeric(df.loc[rows, "last_active_week"], errors="coerce").fillna(0).astype(int)
        names = df.loc[rows, "celebrity_name"].fillna("")
        idxs = df.loc[rows, "index"].astype(int)
        # 排序：非冠军按周升序，冠军（0）排最后
        order = sorted(range(len(rows)), key=lambda i: (last_w.iloc[i] if last_w.iloc[i] > 0 else 10**9))
        placement_by_season[s] = [(int(idxs.iloc[i]), int(last_w.iloc[i])) for i in order]

    # 布局：近似 6x6 子图容纳 34 季
    n = len(seasons)
    cols = 6
    rows_fig = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_fig, cols, figsize=(cols * 3.2, rows_fig * 2.6), constrained_layout=False)
    axes = np.array(axes).reshape(rows_fig, cols)

    # 统一色阶：按所有 SD 的 95% 分位作为上限，抑制极端值
    all_vals: List[float] = []
    for ev in events:
        arr = sd_map.get((ev.season, ev.week))
        if arr is None:
            continue
        v = np.asarray(arr).ravel()
        if v.size == 0:
            continue
        v = v[np.isfinite(v)]
        if v.size:
            all_vals.append(v)
    if all_vals:
        concat = np.concatenate(all_vals)
        try:
            vmax_global = float(np.percentile(concat, 99))
        except Exception:
            vmax_global = float(np.nanmax(concat)) if np.isfinite(np.nanmax(concat)) else 1.0
        if not (np.isfinite(vmax_global) and vmax_global > 0):
            vmax_global = 1.0
    else:
        vmax_global = 1.0

    for idx_s, s in enumerate(seasons):
        ax = axes[idx_s // cols][idx_s % cols]
        # 每季最大周数与选手数
        max_week = max([ev.week for ev in events if ev.season == s], default=0)
        N_s = len(placement_by_season[s])
        grid = np.full((N_s, max_week), np.nan, dtype=float)
        # 填充：按位次行，对每周的活跃选手填 RSD
        # 建立 (index -> 行号) 映射
        index_to_row = {placement_by_season[s][r][0]: r for r in range(N_s)}
        # 构建 (week -> sd 数组, active_ids)
        for ev in [e for e in events if e.season == s]:
            sd_vals = sd_map.get((s, ev.week))
            if sd_vals is None:
                continue
            for k, gid in enumerate(ev.active_ids):
                c = contestants[gid]
                row = index_to_row.get(c.index, None)
                if row is None:
                    continue
                # 仅在该周仍然活跃时填值
                # 近似：若该选手 last_active_week >= 当前周 或 = 0(冠军)
                # 从 df 取 last_active_week
                lw = df.loc[df["season"].astype(int) == s]
                lw = lw.loc[lw["index"].astype(int) == c.index, "last_active_week"].astype(float)
                last_active = int(pd.to_numeric(lw, errors="coerce").fillna(0).iloc[0]) if len(lw) else 0
                if last_active == 0 or ev.week <= last_active:
                    grid[row, ev.week - 1] = float(sd_vals[k])
        # 非线性归一提高低值分辨率；避免同时传 norm 与 vmin/vmax
        try:
            from matplotlib.colors import PowerNorm
            norm = PowerNorm(gamma=0.5, vmin=0.0, vmax=vmax_global)
            im = ax.imshow(grid, aspect="auto", cmap="viridis", norm=norm)
        except Exception:
            im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0.0, vmax=vmax_global)
        ax.set_title(f"S{s}", fontsize=9, pad=6)
        ax.set_xlabel("Week", fontsize=8)
        ax.set_ylabel("Placement", fontsize=8)
        ax.set_xticks(range(max_week))
        ax.set_xticklabels([str(w) for w in range(1, max_week + 1)], fontsize=7)
        ax.set_yticks(range(N_s))
        ax.set_yticklabels([str(r + 1) for r in range(N_s)], fontsize=7)
    # 清理空子图
    for i in range(n, rows_fig * cols):
        axes[i // cols][i % cols].axis("off")
    # 统一加色阶条
    # 将色阶条放到右侧，避免与小图重叠（增加右边距）
    plt.subplots_adjust(left=0.08, bottom=0.08, right=0.85, top=0.92, wspace=0.25, hspace=0.35)
    cb_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cb_ax)
    cbar.set_label("SD(V)")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_meanV_panels(df: pd.DataFrame, contestants: List[Contestant], events: List[Event], meanV_map: Dict[Tuple[int, int], np.ndarray], out_path: str):
    if plt is None:
        print("未安装 matplotlib，跳过 mean(V) 绘图。")
        return
    seasons = sorted(df["season"].dropna().astype(int).unique().tolist())
    n = len(seasons)
    cols = 6
    rows_fig = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_fig, cols, figsize=(cols * 3.2, rows_fig * 2.6), constrained_layout=False)
    axes = np.array(axes).reshape(rows_fig, cols)
    vmax_global = 0.0
    for ev in events:
        arr = meanV_map.get((ev.season, ev.week))
        if arr is not None:
            vmax_global = max(vmax_global, float(np.nanmax(arr)))
    vmax_global = vmax_global if np.isfinite(vmax_global) and vmax_global > 0 else 1.0
    for idx_s, s in enumerate(seasons):
        ax = axes[idx_s // cols][idx_s % cols]
        max_week = max([ev.week for ev in events if ev.season == s], default=0)
        rows = df.index[df["season"].astype(int) == s].tolist()
        last_w = pd.to_numeric(df.loc[rows, "last_active_week"], errors="coerce").fillna(0).astype(int)
        idxs = df.loc[rows, "index"].astype(int)
        order = sorted(range(len(rows)), key=lambda i: (last_w.iloc[i] if last_w.iloc[i] > 0 else 10**9))
        placement = [(int(idxs.iloc[i]), int(last_w.iloc[i])) for i in order]
        N_s = len(placement)
        grid = np.full((N_s, max_week), np.nan, dtype=float)
        index_to_row = {placement[r][0]: r for r in range(N_s)}
        for ev in [e for e in events if e.season == s]:
            meanV = meanV_map.get((s, ev.week))
            if meanV is None:
                continue
            for k, gid in enumerate(ev.active_ids):
                c = contestants[gid]
                row = index_to_row.get(c.index, None)
                if row is None:
                    continue
                lw = df.loc[df["season"].astype(int) == s]
                lw = lw.loc[lw["index"].astype(int) == c.index, "last_active_week"].astype(float)
                last_active = int(pd.to_numeric(lw, errors="coerce").fillna(0).iloc[0]) if len(lw) else 0
                if last_active == 0 or ev.week <= last_active:
                    grid[row, ev.week - 1] = float(meanV[k])
        # 非线性色阶，提高低值分辨率（幂次归一），并避免 norm 与 vmin/vmax 同传
        try:
            from matplotlib.colors import PowerNorm
            norm = PowerNorm(gamma=0.5, vmin=0.0, vmax=vmax_global)
            im = ax.imshow(grid, aspect="auto", cmap="magma", norm=norm)
        except Exception:
            im = ax.imshow(grid, aspect="auto", cmap="magma", vmin=0.0, vmax=vmax_global)
        ax.set_title(f"S{s}", fontsize=9, pad=6)
        ax.set_xlabel("Week", fontsize=8)
        ax.set_ylabel("Placement", fontsize=8)
        ax.set_xticks(range(max_week))
        ax.set_xticklabels([str(w) for w in range(1, max_week + 1)], fontsize=7)
        ax.set_yticks(range(N_s))
        ax.set_yticklabels([str(r + 1) for r in range(N_s)], fontsize=7)
    for i in range(n, rows_fig * cols):
        axes[i // cols][i % cols].axis("off")
    plt.subplots_adjust(left=0.08, bottom=0.08, right=0.85, top=0.92, wspace=0.25, hspace=0.35)
    cb_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cb_ax)
    cbar.set_label("mean(V)")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def build_consistency_data(model: VotingModel, events: List[Event]):
    # 计算三组两两一致率，并构建逐季的方格数据以绘图
    rates = {"rank_vs_percent": 0, "rank_vs_pureV": 0, "percent_vs_pureV": 0}
    totals = 0
    squares_rp: Dict[int, List[Dict]] = {}
    squares_rv: Dict[int, List[Dict]] = {}
    squares_pv: Dict[int, List[Dict]] = {}
    for ev in events:
        actual_mask = ev.eliminated_mask.detach().cpu().numpy()
        k = int(actual_mask.sum())
        if k == 0:
            continue
        totals += 1
        ev_percent = Event(ev.season, ev.week, "percent", ev.active_ids, ev.J_norm, ev.fans_norm, ev.eliminated_mask, ev.is_final_week)
        ev_rank = Event(ev.season, ev.week, "rank", ev.active_ids, ev.J_norm, ev.fans_norm, ev.eliminated_mask, ev.is_final_week)
        probs_p = model.forward_event(ev_percent).detach().cpu().numpy()
        probs_r = model.forward_event(ev_rank).detach().cpu().numpy()
        V = compute_event_V(model, ev).detach().cpu().numpy()
        set_p = set(np.argsort(-probs_p)[:k].tolist())
        set_r = set(np.argsort(-probs_r)[:k].tolist())
        set_v = set(np.argsort(V)[:k].tolist())
        # 构造标签字符串
        def label_for(consis: bool, week: int, A: set, B: set, tagA: str, tagB: str):
            if consis:
                return f"Wk {week} OK {sorted(list(A))}"
            else:
                return f"Wk {week} {tagA}:{sorted(list(A))} {tagB}:{sorted(list(B))}"
        # rank vs percent
        cons_rp = (set_r == set_p)
        rates["rank_vs_percent"] += int(cons_rp)
        squares_rp.setdefault(ev.season, []).append({
            "correct": cons_rp,
            "week": ev.week,
            "name": label_for(cons_rp, ev.week, set_r, set_p, "R", "P"),
            "is_champion": False,
        })
        # rank vs pureV
        cons_rv = (set_r == set_v)
        rates["rank_vs_pureV"] += int(cons_rv)
        squares_rv.setdefault(ev.season, []).append({
            "correct": cons_rv,
            "week": ev.week,
            "name": label_for(cons_rv, ev.week, set_r, set_v, "R", "V"),
            "is_champion": False,
        })
        # percent vs pureV
        cons_pv = (set_p == set_v)
        rates["percent_vs_pureV"] += int(cons_pv)
        squares_pv.setdefault(ev.season, []).append({
            "correct": cons_pv,
            "week": ev.week,
            "name": label_for(cons_pv, ev.week, set_p, set_v, "P", "V"),
            "is_champion": False,
        })
    # 转为比率
    rates_out = {
        "rank_vs_percent": (rates["rank_vs_percent"] / totals) if totals else 0.0,
        "rank_vs_pureV": (rates["rank_vs_pureV"] / totals) if totals else 0.0,
        "percent_vs_pureV": (rates["percent_vs_pureV"] / totals) if totals else 0.0,
        "events_count": float(totals),
    }
    return rates_out, squares_rp, squares_rv, squares_pv


# 已移除：mean(V)~fans 散点与每周 SD 箱线图（不再需要）


def analyze_gamma_vs_age(df: pd.DataFrame, contestants: List[Contestant], model: VotingModel, out_dir: str):
    # 使用训练好的 gamma 与 df 的 celebrity_age_during_season 做散点与回归（线性/二次择优）
    if plt is None:
        print("未安装 matplotlib，跳过 gamma~age 分析。")
        return
    os.makedirs(out_dir, exist_ok=True)
    ages = []
    gammas = []
    for i, c in enumerate(contestants):
        row = c.row_idx
        if row < 0 or row >= len(df):
            continue
        age_val = pd.to_numeric(df.loc[row, "celebrity_age_during_season"], errors="coerce")
        if pd.isna(age_val):
            continue
        ages.append(float(age_val))
        gammas.append(float(model.gamma[i].item()))
    if len(ages) < 3:
        print("gamma~age 数据不足，跳过绘图。")
        return
    x = np.array(ages)
    y = np.array(gammas)
    # 线性拟合
    p1 = np.polyfit(x, y, 1)
    y1 = p1[0]*x + p1[1]
    # 计算线性回归 R^2
    def r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - y_true.mean())**2)
        return 1.0 - (ss_res/ss_tot if ss_tot > 0 else 0.0)
    r2_lin = r2(y, y1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y, s=10, alpha=0.7, label="gamma vs age")
    # 绘制线性拟合曲线（按 x 范围排序）
    xline = np.linspace(x.min(), x.max(), 200)
    yline1 = p1[0]*xline + p1[1]
    ax.plot(xline, yline1, color='tab:orange', linewidth=1.5, label=f"Linear R^2={r2_lin:.3f}")
    ax.set_title("gamma ~ age (linear regression)")
    ax.set_xlabel("celebrity_age_during_season")
    ax.set_ylabel("gamma")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gamma_vs_age_scatter_regression.png"), dpi=200)
    plt.close(fig)
    # 输出线性回归系数到 JSON
    out_json = {
        "slope": float(p1[0]),
        "intercept": float(p1[1]),
        "r2": float(r2_lin),
        "n": int(len(x))
    }
    with open(os.path.join(out_dir, "gamma_vs_age_regression.json"), "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)


def boxplot_gamma_by_category(df: pd.DataFrame, contestants: List[Contestant], model: VotingModel, column: str, title: str, out_path: str, min_count: int = 1):
    # 按分类字段绘制 gamma 的箱线图
    if plt is None:
        print(f"未安装 matplotlib，跳过 {column} 箱线图。")
        return
    groups: Dict[str, List[float]] = {}
    for i, c in enumerate(contestants):
        row = c.row_idx
        if row < 0 or row >= len(df):
            continue
        # 提取原始值并统一处理缺失/空白/'nan'
        val_raw = df.loc[row, column] if column in df.columns else None
        key: str
        if (
            val_raw is None
            or (isinstance(val_raw, float) and np.isnan(val_raw))
            or (isinstance(val_raw, str) and (val_raw.strip() == "" or val_raw.strip().lower() == "nan"))
        ):
            # homestate 的空值统一标记为 non-america
            key = "non-america" if column == "celebrity_homestate" else "(Unknown)"
        else:
            key = str(val_raw)
        val = float(model.gamma[i].item())
        groups.setdefault(key, []).append(val)
    # 过滤最少样本
    items = [(k, v) for k, v in groups.items() if len(v) >= min_count]
    if not items:
        print(f"{column} 分类数据不足，跳过绘图。")
        return
    # 按样本数排序，截断到前 20 类以避免标签拥挤
    items.sort(key=lambda kv: len(kv[1]), reverse=True)
    labels = [k for k, _ in items[:20]]
    data = [groups[k] for k in labels]
    fig, ax = plt.subplots(figsize=(max(8, len(labels)*0.6), 5))
    ax.boxplot(data, tick_labels=labels)
    ax.set_title(title)
    ax.set_ylabel("gamma")
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_controversy_for_one(df: pd.DataFrame, contestants: List[Contestant], events: List[Event], model: VotingModel, season: int, celeb_name: str, out_path: str):
    # 为指定赛季与名人绘制4行格子图：实际、百分比法、排名法、排名法(含二次选择)
    if plt is None:
        print("未安装 matplotlib，跳过争议名人绘图。")
        return
    # 找到该名人的全局ID
    gid = None
    for i, c in enumerate(contestants):
        if c.season == season and (c.name == celeb_name or str(df.loc[c.row_idx, "celebrity_name"]) == celeb_name):
            gid = i
            break
    if gid is None:
        print(f"未找到 {celeb_name} (Season {season})，跳过。")
        return
    # 收集该季该名人的事件（活跃周），按周排序
    evs = [e for e in events if e.season == season and gid in e.active_ids]
    evs.sort(key=lambda e: e.week)
    if not evs:
        print(f"Season {season} {celeb_name} 无活跃周，跳过。")
        return
    # 生成4行的颜色序列
    rows_labels = ["Actual", "Percent", "Rank", "Rank+TwoStage"]
    rows_colors = [[] for _ in range(4)]
    # 实际结果：直到被淘汰（红），冠军则全绿
    stopped_actual = False
    for ev in evs:
        k_idx = ev.active_ids.index(gid)
        eliminated = bool(ev.eliminated_mask[k_idx].item())
        if not stopped_actual:
            rows_colors[0].append('#4CAF50' if not eliminated else '#F44336')
            if eliminated:
                stopped_actual = True
        # 如果已被淘汰，不再添加后续格子
    # 百分比法预测
    stopped_p = False
    for ev in evs:
        if stopped_p:
            break
        ev_p = Event(ev.season, ev.week, "percent", ev.active_ids, ev.J_norm, ev.fans_norm, ev.eliminated_mask, ev.is_final_week)
        probs_p = model.forward_event(ev_p).detach().cpu().numpy()
        k = int(ev.eliminated_mask.sum().item())
        if k > 0:
            pred_set = set(np.argsort(-probs_p)[:k].tolist())
            k_idx = ev.active_ids.index(gid)
            eliminated_pred = (k_idx in pred_set)
            rows_colors[1].append('#F44336' if eliminated_pred else '#4CAF50')
            if eliminated_pred:
                stopped_p = True
        else:
            rows_colors[1].append('#4CAF50')
    # 排名法预测
    stopped_r = False
    for ev in evs:
        if stopped_r:
            break
        ev_r = Event(ev.season, ev.week, "rank", ev.active_ids, ev.J_norm, ev.fans_norm, ev.eliminated_mask, ev.is_final_week)
        probs_r = model.forward_event(ev_r).detach().cpu().numpy()
        k = int(ev.eliminated_mask.sum().item())
        if k > 0:
            pred_set = set(np.argsort(-probs_r)[:k].tolist())
            k_idx = ev.active_ids.index(gid)
            eliminated_pred = (k_idx in pred_set)
            rows_colors[2].append('#F44336' if eliminated_pred else '#4CAF50')
            if eliminated_pred:
                stopped_r = True
        else:
            rows_colors[2].append('#4CAF50')
    # 排名法（含二次选择）预测：即使非 S28-34，也假设启用二次选择
    stopped_rs = False
    for ev in evs:
        if stopped_rs:
            break
        # 强制使用二次选择的排名法概率
        probs_rs = compute_rank_probs_with_two_stage(model, ev).detach().cpu().numpy()
        k = int(ev.eliminated_mask.sum().item())
        if k > 0:
            pred_set = set(np.argsort(-probs_rs)[:k].tolist())
            k_idx = ev.active_ids.index(gid)
            eliminated_pred = (k_idx in pred_set)
            rows_colors[3].append('#F44336' if eliminated_pred else '#4CAF50')
            if eliminated_pred:
                stopped_rs = True
        else:
            rows_colors[3].append('#4CAF50')

    # 绘制 4 x W 的方格图
    W = max(len(rc) for rc in rows_colors)
    fig_w = max(8, W * 0.5 + 3)
    fig_h = 3.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, W)
    ax.set_ylim(0, 4)
    ax.set_aspect('equal')
    ax.axis('off')
    for r in range(4):
        for w in range(len(rows_colors[r])):
            color = rows_colors[r][w]
            rect = plt.Rectangle((w, 3 - r), 0.9, 0.9, facecolor=color, edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
        # 行标签
        ax.text(-0.5, 3 - r + 0.45, rows_labels[r], ha='right', va='center', fontsize=8)
    # 列标签为周数
    for w in range(W):
        ax.text(w + 0.45, -0.2, str(w + 1), ha='center', va='top', fontsize=8)
    title = f"Season {season} - {celeb_name}"
    ax.set_title(title, fontsize=10, pad=6)
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

    # 不再保存最终 params.json（仅保存 best）

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
    # 不再保存最终 accuracy_grid.png（仅保存 best）

    print(f"训练完成。参数与输出保存在 {out_dir}")
    print(
        f"验证指标：accuracy={accuracy:.4f} (correct={metrics['correct_total']}+bonus={metrics['bonus_correct']} / target={metrics['target_total']} [elim={metrics['eliminated_total']}+finals={metrics['finals_count']}])"
    )


def infer(data_path: str, best_path: str, out_dir: str, weeks: int = 11, damp: float = 0.0):
    df = pd.read_csv(data_path, dtype=str)
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["index"] = pd.to_numeric(df["index"], errors="coerce")
    df["last_active_week"] = pd.to_numeric(df["last_active_week"], errors="coerce")

    contestants, events = build_dataset(df, weeks=weeks)
    model = VotingModel(num_contestants=len(contestants))
    # 读取 best 参数
    with open(best_path, "r", encoding="utf-8") as f:
        best_params = json.load(f)
    set_model_params_from_best(model, contestants, best_params)
    # Fisher 对角近似
    fisher_diag = compute_fisher_diag(model, events)
    # 使用 Delta 方法计算 SD 与 mean(V)
    sd_map, meanV_map = sd_delta_method(model, events, fisher_diag, damp=damp)
    # 输出 SD 面板图与 mean(V) 面板图，附带色阶
    os.makedirs(out_dir, exist_ok=True)
    plot_sd_panels(df, contestants, events, sd_map, os.path.join(out_dir, "sd_panels.png"))
    plot_meanV_panels(df, contestants, events, meanV_map, os.path.join(out_dir, "meanV_panels.png"))
    # 导出 mean(V) 到 JSON：逐季逐周逐选手
    meanV_rows = []
    for ev in events:
        meanV = meanV_map.get((ev.season, ev.week))
        if meanV is None:
            continue
        for k, gid in enumerate(ev.active_ids):
            c = contestants[gid]
            meanV_rows.append({
                "season": int(ev.season),
                "week": int(ev.week),
                "contestant_index": int(c.index),
                "name": c.name,
                "V": float(meanV[k])
            })
    with open(os.path.join(out_dir, "meanV.json"), "w", encoding="utf-8") as f:
        json.dump(meanV_rows, f, ensure_ascii=False, indent=2)
    # 一致性检验：三组两两一致率 + 三张栅格图
    rates, sq_rp, sq_rv, sq_pv = build_consistency_data(model, events)
    with open(os.path.join(out_dir, "consistency.json"), "w", encoding="utf-8") as f:
        json.dump(rates, f, ensure_ascii=False, indent=2)
    plot_accuracy_grid(sq_rp, os.path.join(out_dir, "consistency_rank_vs_percent.png"))
    plot_accuracy_grid(sq_rv, os.path.join(out_dir, "consistency_rank_vs_pureV.png"))
    plot_accuracy_grid(sq_pv, os.path.join(out_dir, "consistency_percent_vs_pureV.png"))
    # 回归与箱线图
    # 已删除不需要的 mean(V)~fans 回归散点与 SD 周度箱线图
    # 名人特征分析（第7节）：基于训练好的 gamma 进行年龄回归与分类箱线图
    features_out = os.path.join(out_dir, "features")
    analyze_gamma_vs_age(df, contestants, model, features_out)
    boxplot_gamma_by_category(df, contestants, model, "celebrity_industry", "gamma by industry", os.path.join(features_out, "gamma_by_industry.png"), min_count=3)
    boxplot_gamma_by_category(df, contestants, model, "celebrity_homestate", "gamma by homestate", os.path.join(features_out, "gamma_by_homestate.png"), min_count=3)
    # 注意列名可能包含斜杠，按原样访问
    col_homecountry = "celebrity_homecountry/region" if "celebrity_homecountry/region" in df.columns else "celebrity_homecountry"
    boxplot_gamma_by_category(df, contestants, model, col_homecountry, "gamma by homecountry/region", os.path.join(features_out, "gamma_by_homecountry.png"), min_count=3)
    # 争议名人检验：四个例子各绘制一张 4 行格子图
    controversies_out = os.path.join(out_dir, "controversies")
    plot_controversy_for_one(df, contestants, events, model, 2, "Jerry Rice", os.path.join(controversies_out, "S2_Jerry_Rice.png"))
    plot_controversy_for_one(df, contestants, events, model, 4, "Billy Ray Cyrus", os.path.join(controversies_out, "S4_Billy_Ray_Cyrus.png"))
    plot_controversy_for_one(df, contestants, events, model, 11, "Bristol Palin", os.path.join(controversies_out, "S11_Bristol_Palin.png"))
    plot_controversy_for_one(df, contestants, events, model, 27, "Bobby Bones", os.path.join(controversies_out, "S27_Bobby_Bones.png"))
    print(f"推理完成。输出保存在 {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train voting model per model.md")
    parser.add_argument("--data", default="data_new.csv", help="输入数据 CSV 路径")
    parser.add_argument("--out", default="artifacts", help="输出目录")
    parser.add_argument("--mode", choices=["train", "infer"], default="train", help="运行模式：train 或 infer")
    parser.add_argument("--best", default="artifacts/best_params.json", help="infer 模式下读取的 best 参数路径")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.05, help="学习率")
    parser.add_argument("--weeks", type=int, default=11, help="最大周数")
    parser.add_argument("--damp", type=float, default=1e-1, help="SD 计算的阻尼 λ (damping) 用于稳定协方差")
    args = parser.parse_args()
    if args.mode == "train":
        train(args.data, args.out, epochs=args.epochs, lr=args.lr, weeks=args.weeks)
    else:
        # 推理输出不要与 artifacts 混放，使用单独目录
        infer_out = args.out if args.out != "artifacts" else os.path.join("analysis")
        infer(args.data, args.best, out_dir=infer_out, weeks=args.weeks, damp=args.damp)


if __name__ == "__main__":
    main()
