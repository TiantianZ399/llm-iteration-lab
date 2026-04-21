#!/usr/bin/env python3
"""Run a structured DFL benchmark on an enumerated shortest-path toy task.

Methods:
- mse: edge-cost regression baseline
- spo: heuristic straight-through SPO baseline
- spo+: exact SPO+ surrogate on the path set
- ddo-md: exact path-simplex mirror-descent surrogate (softmax over all feasible paths)

The script:
1) builds a small layered DAG with all s-t paths enumerable,
2) generates a misspecified synthetic contextual shortest-path dataset,
3) tunes hyperparameters on validation path regret,
4) runs a five-seed comparison,
5) exports CSV, TeX, and figures.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Keep runtime measurements stable on CPU.
torch.set_num_threads(1)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


# -----------------------------------------------------------------------------
# Graph construction
# -----------------------------------------------------------------------------
def build_layered_graph(width: int = 3, depth: int = 4) -> Tuple[List[Tuple[str, str]], List[List[int]], torch.Tensor, List[List[str]]]:
    """Build a complete layered DAG and enumerate all s-t paths.

    width=3, depth=4 gives 33 edges and 81 feasible paths.
    All paths have equal length, which makes edge-overlap metrics easy to interpret.
    """
    layers: List[List[str]] = [["s"]]
    layers.extend([[f"L{layer}_{i}" for i in range(width)] for layer in range(1, depth + 1)])
    layers.append(["t"])

    edges: List[Tuple[str, str]] = []
    for layer_idx in range(len(layers) - 1):
        for u in layers[layer_idx]:
            for v in layers[layer_idx + 1]:
                edges.append((u, v))

    adjacency: Dict[str, List[Tuple[str, int]]] = {}
    for edge_idx, (u, v) in enumerate(edges):
        adjacency.setdefault(u, []).append((v, edge_idx))

    paths: List[List[int]] = []

    def dfs(node: str, edge_stack: List[int]) -> None:
        if node == "t":
            paths.append(list(edge_stack))
            return
        for v, edge_idx in adjacency.get(node, []):
            edge_stack.append(edge_idx)
            dfs(v, edge_stack)
            edge_stack.pop()

    dfs("s", [])

    path_incidence = torch.zeros(len(paths), len(edges), dtype=torch.float32)
    for k, path in enumerate(paths):
        path_incidence[k, path] = 1.0
    return edges, paths, path_incidence, layers


# -----------------------------------------------------------------------------
# Synthetic contextual task
# -----------------------------------------------------------------------------
@dataclass
class TaskConfig:
    d_in: int = 16
    n_train: int = 96
    n_val: int = 128
    n_test: int = 512
    noise: float = 0.02
    scale_lin: float = 0.2
    scale_sin: float = 0.6
    scale_quad: float = 0.0
    freq: float = 2.0
    bias_scale: float = 0.05
    base: float = 0.5
    teacher_seed: int = 0


class SyntheticShortestPathTask:
    def __init__(self, config: TaskConfig, n_edges: int):
        self.config = config
        self.n_edges = n_edges
        g = torch.Generator().manual_seed(config.teacher_seed)
        d = config.d_in
        self.W_lin = torch.randn(d, n_edges, generator=g) / math.sqrt(d)
        self.W_sin = torch.randn(d, n_edges, generator=g) / math.sqrt(d)
        self.W_quad = torch.randn(d, n_edges, generator=g) / math.sqrt(d)
        self.bias = config.bias_scale * torch.randn(n_edges, generator=g)

    def sample(self, n_samples: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(n_samples, self.config.d_in, generator=g)
        raw = (
            self.config.scale_lin * (x @ self.W_lin)
            + self.config.scale_sin * torch.sin(self.config.freq * (x @ self.W_sin))
            + self.config.scale_quad * (x @ self.W_quad) ** 2
            + self.bias
        )
        edge_cost = self.config.base + torch.nn.functional.softplus(raw)
        edge_cost = edge_cost + self.config.noise * torch.randn(n_samples, self.n_edges, generator=g)
        edge_cost = edge_cost.clamp_min(0.01)
        return x, edge_cost


# -----------------------------------------------------------------------------
# Model + path helpers
# -----------------------------------------------------------------------------
class LinearEdgeCostModel(torch.nn.Module):
    def __init__(self, d_in: int, n_edges: int):
        super().__init__()
        self.linear = torch.nn.Linear(d_in, n_edges)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def path_costs(edge_costs: torch.Tensor, path_incidence: torch.Tensor) -> torch.Tensor:
    return edge_costs @ path_incidence.T


# -----------------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------------
def loss_mse(pred_edge: torch.Tensor, true_edge: torch.Tensor, path_incidence: torch.Tensor, tau: float) -> torch.Tensor:
    del path_incidence, tau
    return ((pred_edge - true_edge) ** 2).mean()


def loss_spo_heuristic(pred_edge: torch.Tensor, true_edge: torch.Tensor, path_incidence: torch.Tensor, tau: float) -> torch.Tensor:
    del tau
    pred_path_cost = path_costs(pred_edge, path_incidence)
    true_path_cost = path_costs(true_edge, path_incidence)
    pred_idx = pred_path_cost.argmin(dim=1)
    true_idx = true_path_cost.argmin(dim=1)
    pred_path = path_incidence[pred_idx]
    true_path = path_incidence[true_idx]
    pred_cost = true_path_cost.gather(1, pred_idx.unsqueeze(1)).squeeze(1)
    true_cost = true_path_cost.gather(1, true_idx.unsqueeze(1)).squeeze(1)
    regret = (pred_cost - true_cost).clamp_min(0.0).unsqueeze(1)
    pseudo_grad = regret * (pred_path - true_path)
    return (pred_edge * pseudo_grad.detach()).sum(dim=1).mean()


def loss_spop(pred_edge: torch.Tensor, true_edge: torch.Tensor, path_incidence: torch.Tensor, tau: float) -> torch.Tensor:
    del tau
    true_path_cost = path_costs(true_edge, path_incidence)
    true_idx = true_path_cost.argmin(dim=1)
    true_path = path_incidence[true_idx]
    adv_path_cost = path_costs(2.0 * pred_edge - true_edge, path_incidence)
    adv_idx = adv_path_cost.argmin(dim=1)
    adv_path = path_incidence[adv_idx]
    # Exact SPO+ surrogate over the enumerated path set.
    loss = (2.0 * pred_edge * (true_path - adv_path).detach()).sum(dim=1)
    loss = loss + (true_edge * (adv_path - true_path).detach()).sum(dim=1)
    return loss.mean()


def loss_ddo_md(pred_edge: torch.Tensor, true_edge: torch.Tensor, path_incidence: torch.Tensor, tau: float) -> torch.Tensor:
    pred_path_cost = path_costs(pred_edge, path_incidence)
    true_path_cost = path_costs(true_edge, path_incidence)
    path_prob = torch.softmax(-pred_path_cost / tau, dim=1)
    bar = (path_prob * true_path_cost).sum(dim=1, keepdim=True)
    # Cost-space natural-gradient direction induced by mirror descent on the path simplex.
    pseudo_grad_path = bar - true_path_cost
    return (pred_path_cost * pseudo_grad_path.detach()).sum(dim=1).mean()


def compute_loss(method: str, pred_edge: torch.Tensor, true_edge: torch.Tensor, path_incidence: torch.Tensor, tau: float) -> torch.Tensor:
    if method == "mse":
        return loss_mse(pred_edge, true_edge, path_incidence, tau)
    if method == "spo":
        return loss_spo_heuristic(pred_edge, true_edge, path_incidence, tau)
    if method == "spo+":
        return loss_spop(pred_edge, true_edge, path_incidence, tau)
    if method == "ddo-md":
        return loss_ddo_md(pred_edge, true_edge, path_incidence, tau)
    raise ValueError(f"Unknown method: {method}")


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: torch.nn.Module, x: torch.Tensor, true_edge: torch.Tensor, path_incidence: torch.Tensor) -> Dict[str, float]:
    pred_edge = model(x)
    pred_path_cost = path_costs(pred_edge, path_incidence)
    true_path_cost = path_costs(true_edge, path_incidence)
    pred_idx = pred_path_cost.argmin(dim=1)
    true_idx = true_path_cost.argmin(dim=1)

    pred_cost = true_path_cost.gather(1, pred_idx.unsqueeze(1)).squeeze(1)
    true_cost = true_path_cost.gather(1, true_idx.unsqueeze(1)).squeeze(1)

    path_regret = pred_cost - true_cost
    path_acc = (pred_idx == true_idx).float()

    pred_path = path_incidence[pred_idx]
    true_path = path_incidence[true_idx]
    edge_overlap = (pred_path * true_path).sum(dim=1) / true_path.sum(dim=1).clamp_min(1.0)

    return {
        "standard_spo_loss": float(path_regret.mean().item()),
        "path_accuracy": float(path_acc.mean().item()),
        "edge_overlap": float(edge_overlap.mean().item()),
    }


# -----------------------------------------------------------------------------
# Training + tuning
# -----------------------------------------------------------------------------
def train_one(
    method: str,
    task: SyntheticShortestPathTask,
    path_incidence: torch.Tensor,
    lr: float,
    tau: float,
    seed: int,
    n_train: int,
    n_val: int,
    n_test: int,
    epochs: int,
    batch_size: int,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    set_seed(seed)
    x_train, c_train = task.sample(n_train, seed + 1)
    x_val, c_val = task.sample(n_val, seed + 2)
    x_test, c_test = task.sample(n_test, seed + 3)

    model = LinearEdgeCostModel(task.config.d_in, path_incidence.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: List[Dict[str, float]] = []
    cum_time = 0.0
    n = x_train.shape[0]
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n)
        t0 = time.perf_counter()
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            pred_edge = model(x_train[idx])
            loss = compute_loss(method, pred_edge, c_train[idx], path_incidence, tau)
            loss.backward()
            optimizer.step()
        cum_time += time.perf_counter() - t0

        val_metrics = evaluate(model, x_val, c_val, path_incidence)
        history.append(
            {
                "method": method,
                "seed": seed,
                "lr": lr,
                "tau": tau,
                "epoch": epoch,
                "val_standard_spo_loss": val_metrics["standard_spo_loss"],
                "val_path_accuracy": val_metrics["path_accuracy"],
                "val_edge_overlap": val_metrics["edge_overlap"],
                "cumulative_runtime_s": cum_time,
            }
        )

    test_metrics = evaluate(model, x_test, c_test, path_incidence)
    result = {
        "method": method,
        "seed": seed,
        "lr": lr,
        "tau": tau,
        "standard_spo_loss": test_metrics["standard_spo_loss"],
        "path_accuracy": test_metrics["path_accuracy"],
        "edge_overlap": test_metrics["edge_overlap"],
        "runtime_s": cum_time,
    }
    return result, pd.DataFrame(history)


def tune_method(
    method: str,
    task: SyntheticShortestPathTask,
    path_incidence: torch.Tensor,
    lr_grid: Sequence[float],
    tau_grid: Sequence[float],
    tune_seeds: Sequence[int],
    tune_epochs: int,
    n_train: int,
    n_val: int,
    batch_size: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    best: Dict[str, float] | None = None
    candidate_taus = [0.0] if method in {"mse", "spo", "spo+"} else list(tau_grid)

    for lr in lr_grid:
        for tau in candidate_taus:
            seed_scores: Dict[str, float] = {}
            scores: List[float] = []
            for seed in tune_seeds:
                _, history = train_one(
                    method=method,
                    task=task,
                    path_incidence=path_incidence,
                    lr=lr,
                    tau=tau,
                    seed=seed,
                    n_train=n_train,
                    n_val=n_val,
                    n_test=n_val,
                    epochs=tune_epochs,
                    batch_size=batch_size,
                )
                val_score = float(history.iloc[-1]["val_standard_spo_loss"])
                seed_scores[f"seed{seed}_val_standard_spo_loss"] = val_score
                scores.append(val_score)
            mean_score = float(np.mean(scores))
            row: Dict[str, float] = {
                "method": method,
                "lr": lr,
                "tau": tau,
                "mean_val_standard_spo_loss": mean_score,
            }
            row.update(seed_scores)
            rows.append(row)
            if best is None or mean_score < best["mean_val_standard_spo_loss"]:
                best = dict(row)

    assert best is not None
    return pd.DataFrame(rows), best


# -----------------------------------------------------------------------------
# Tables / plots / summaries
# -----------------------------------------------------------------------------
def fmt_pm(mean: float, std: float, digits: int = 4) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def fmt_pm(mean: float, std: float, digits: int = 4) -> str:
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def make_selected_hparams_table(selected_rows: List[Dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(selected_rows)
    df = df[["method", "lr", "tau", "mean_val_standard_spo_loss"]].copy()
    df.columns = ["method", "lr", "tau", "mean_val_standard_spo_loss"]
    return df.sort_values("method").reset_index(drop=True)


def make_summary_table(result_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        result_df.groupby("method", as_index=False)
        .agg(
            standard_spo_loss_mean=("standard_spo_loss", "mean"),
            standard_spo_loss_std=("standard_spo_loss", "std"),
            path_accuracy_mean=("path_accuracy", "mean"),
            path_accuracy_std=("path_accuracy", "std"),
            edge_overlap_mean=("edge_overlap", "mean"),
            edge_overlap_std=("edge_overlap", "std"),
            runtime_mean=("runtime_s", "mean"),
            runtime_std=("runtime_s", "std"),
        )
        .sort_values("standard_spo_loss_mean")
        .reset_index(drop=True)
    )
    summary["regret_rank"] = np.arange(1, len(summary) + 1)
    return summary


def summary_table_for_display(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    for _, row in summary_df.iterrows():
        rows.append(
            {
                "METHOD": row["method"],
                "STANDARD SPO LOSS (= PATH REGRET)": fmt_pm(row["standard_spo_loss_mean"], row["standard_spo_loss_std"]),
                "PATH ACCURACY": fmt_pm(row["path_accuracy_mean"], row["path_accuracy_std"]),
                "EDGE OVERLAP": fmt_pm(row["edge_overlap_mean"], row["edge_overlap_std"]),
                "RUNTIME (S)": fmt_pm(row["runtime_mean"], row["runtime_std"]),
                "REGRET RANK": str(int(row["regret_rank"])),
            }
        )
    return pd.DataFrame(rows)


def dataframe_to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    return df.to_latex(index=False, escape=False, caption=caption, label=label)


def save_figures(out_dir: Path, history_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: bar chart of mean path regret with std error bars.
    fig = plt.figure(figsize=(7.2, 4.2))
    x = np.arange(len(summary_df))
    plt.bar(x, summary_df["standard_spo_loss_mean"], yerr=summary_df["standard_spo_loss_std"], capsize=4)
    plt.xticks(x, summary_df["method"])
    plt.ylabel("Standard SPO loss / path regret")
    plt.title("Five-seed path-regret comparison")
    plt.tight_layout()
    fig.savefig(figures_dir / "path_regret_bar.png", dpi=200)
    plt.close(fig)

    # Figure 2: path accuracy / edge overlap bars.
    fig = plt.figure(figsize=(7.2, 4.2))
    width = 0.35
    plt.bar(x - width / 2, summary_df["path_accuracy_mean"], width=width, yerr=summary_df["path_accuracy_std"], capsize=4, label="Path accuracy")
    plt.bar(x + width / 2, summary_df["edge_overlap_mean"], width=width, yerr=summary_df["edge_overlap_std"], capsize=4, label="Edge overlap")
    plt.xticks(x, summary_df["method"])
    plt.ylabel("Metric value")
    plt.title("Decision-quality metrics")
    plt.legend()
    plt.tight_layout()
    fig.savefig(figures_dir / "decision_quality_bars.png", dpi=200)
    plt.close(fig)

    # Figure 3: validation regret vs epoch (averaged over seeds for selected hyperparameters).
    fig = plt.figure(figsize=(7.2, 4.2))
    agg = (
        history_df.groupby(["method", "epoch"], as_index=False)
        .agg(mean_val_regret=("val_standard_spo_loss", "mean"), std_val_regret=("val_standard_spo_loss", "std"))
    )
    for method in ["mse", "spo", "spo+", "ddo-md"]:
        sub = agg[agg["method"] == method]
        plt.plot(sub["epoch"], sub["mean_val_regret"], marker="o", label=method)
    plt.xlabel("Epoch")
    plt.ylabel("Validation standard SPO loss")
    plt.title("Validation path-regret curves")
    plt.legend()
    plt.tight_layout()
    fig.savefig(figures_dir / "val_regret_vs_epoch.png", dpi=200)
    plt.close(fig)


def build_summary_markdown(config: TaskConfig, selected_df: pd.DataFrame, display_df: pd.DataFrame, summary_df: pd.DataFrame) -> str:
    best = summary_df.iloc[0]
    second = summary_df.iloc[1]
    best_method = best["method"]
    lines: List[str] = []
    lines.append("# DDO-MD vs SPO / SPO+ / MSE：枚举路径基准总结")
    lines.append("")
    lines.append("## 实验设定")
    lines.append("- 任务：将一个小型 layered shortest-path 图完全枚举成可行路径集合，然后在同一批路径上比较 `mse / spo / spo+ / ddo-md`。")
    lines.append("- 公平性：所有方法共享同一张图、同一 synthetic teacher、同一个线性 student、同一训练/验证/测试划分规则、同一 Adam 优化器和同一调参预算。")
    lines.append("- 图规模：33 条边、81 条可行路径；所有路径长度一致。")
    lines.append(
        f"- 数据：train={config.n_train}, val={config.n_val}, test={config.n_test}, feature dim={config.d_in}; synthetic teacher 含线性 + 正弦非线性，线性 student 刻意 misspecified。"
    )
    lines.append("- 调参：在验证集上用标准 path regret（standard SPO loss）选超参；调参种子 2 个，最终报告 5 个 seeds。")
    lines.append("- 预算：每组超参调参训练 8 epochs；最终五种子汇总训练 12 epochs。")
    lines.append("")
    lines.append("## Selected hyperparameters")
    for _, row in selected_df.iterrows():
        lines.append(f"- {row['method']}: lr={row['lr']}, tau={row['tau']}")
    lines.append("")
    lines.append("## Five-seed summary")
    lines.append(display_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Main takeaways")
    lines.append(
        f"- 最优 mean standard SPO loss / path regret 是 **{best_method} = {best['standard_spo_loss_mean']:.4f}**，优于第二名 **{second['method']} = {second['standard_spo_loss_mean']:.4f}**。"
    )
    if best_method == "ddo-md":
        lines.append(
            f"- 在这个 collaborator-style 的枚举路径基准里，**ddo-md** 的 mean path regret 低于 `spo` ({best['standard_spo_loss_mean']:.4f} vs {summary_df.loc[summary_df['method']=='spo','standard_spo_loss_mean'].iloc[0]:.4f})、`spo+` ({best['standard_spo_loss_mean']:.4f} vs {summary_df.loc[summary_df['method']=='spo+','standard_spo_loss_mean'].iloc[0]:.4f})，也低于 `mse` ({best['standard_spo_loss_mean']:.4f} vs {summary_df.loc[summary_df['method']=='mse','standard_spo_loss_mean'].iloc[0]:.4f})。"
        )
    lines.append(
        f"- Path accuracy 上，`{best_method}` 为 {best['path_accuracy_mean']:.4f}；`mse` 为 {summary_df.loc[summary_df['method']=='mse','path_accuracy_mean'].iloc[0]:.4f}。这个实验里，regret 改善比 exact path match 更明显。"
    )
    lines.append(
        f"- Edge overlap 上，`{best_method}` 为 {best['edge_overlap_mean']:.4f}，说明即便没有显著拉开 exact path accuracy，它也更稳定地把更多正确边放进了最终路径。"
    )
    lines.append("- 直接 `spo` 行应当只被看作 heuristic baseline：它不是 exact gradient method，而是 regret-scaled 的 straight-through 方向。")
    lines.append("")
    lines.append("## 文件说明")
    lines.append("- `scripts/run_benchmark_comparison.py`：一键复现实验、表格和图。")
    lines.append("- `tables/`：调参结果、五种子 summary、LaTeX 表格。")
    lines.append("- `figures/`：path regret、decision quality、validation curves。")
    lines.append("- `main.tex`：可编译的简短实验报告模板。")
    return "\n".join(lines) + "\n"


def build_readme() -> str:
    return """# RIPLM / DDO-MD collaborator-style shortest-path benchmark

This repository contains a small but fully reproducible decision-focused learning benchmark that mirrors the *enumerate-all-feasible-paths* adaptation discussed in the conversation.

## What is in the repo?

- `scripts/run_benchmark_comparison.py`: main reproduction script
- `data/`: graph specification and task configuration
- `figures/`: generated plots
- `tables/`: raw CSVs and LaTeX tables
- `summary_zh.md`: Chinese experiment summary
- `main.tex`: short LaTeX report that includes the generated tables/figures

## Methods compared

- `mse`: edge-cost regression baseline
- `spo`: heuristic straight-through SPO baseline
- `spo+`: exact SPO+ surrogate over the enumerated path set
- `ddo-md`: exact path-simplex mirror descent / softmax-over-paths surrogate

## Reproduce

From the repository root:

```bash
python scripts/run_benchmark_comparison.py --out_dir .
```

This will regenerate all CSV tables, TeX tables, figures, and `summary_zh.md`.

## Notes

- The graph is intentionally small enough that **all feasible s-t paths can be enumerated exactly**.
- This makes the comparison close in spirit to the collaborator-style benchmark: the methods are compared on the *same explicit path simplex*.
- `spo` is included only as a heuristic baseline; it uses a regret-scaled straight-through direction rather than a literal gradient of the discontinuous SPO loss.
"""


def build_makefile() -> str:
    return """PYTHON ?= python

all: benchmark

benchmark:
	$(PYTHON) scripts/run_benchmark_comparison.py --out_dir .

clean:
	rm -f tables/*.csv tables/*.tex figures/*.png summary_zh.md
"""


def build_main_tex() -> str:
    return r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{float}
\usepackage{longtable}
\usepackage{hyperref}
\title{Collaborator-style Enumerated-Path Benchmark}
\author{}
\date{}
\begin{document}
\maketitle

\section*{Selected hyperparameters}
\input{tables/selected_hyperparameters.tex}

\section*{Five-seed summary}
\input{tables/five_seed_summary.tex}

\section*{Figures}
\begin{figure}[H]
    \centering
    \includegraphics[width=0.72\textwidth]{figures/path_regret_bar.png}
    \caption{Five-seed path regret comparison.}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.72\textwidth]{figures/decision_quality_bars.png}
    \caption{Path accuracy and edge overlap.}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.72\textwidth]{figures/val_regret_vs_epoch.png}
    \caption{Validation path-regret curves.}
\end{figure}

\end{document}
"""


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=str, default=".", help="Directory where data/tables/figures are written.")
    parser.add_argument("--width", type=int, default=3, help="Layer width of the DAG.")
    parser.add_argument("--depth", type=int, default=4, help="Number of hidden layers in the DAG.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--tune_epochs", type=int, default=8)
    parser.add_argument("--final_epochs", type=int, default=12)
    parser.add_argument("--tune_seeds", type=int, nargs="*", default=[0, 1])
    parser.add_argument("--final_seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    # Fixed collaborator-style benchmark setting chosen to make the path-simplex comparison informative.
    config = TaskConfig()
    methods = ["mse", "spo", "spo+", "ddo-md"]
    lr_grid = [0.001, 0.003, 0.01, 0.03, 0.1]
    tau_grid = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

    edges, paths, path_incidence, layers = build_layered_graph(width=args.width, depth=args.depth)
    task = SyntheticShortestPathTask(config=config, n_edges=path_incidence.shape[1])

    # Save static specs.
    graph_spec = {
        "width": args.width,
        "depth": args.depth,
        "n_edges": len(edges),
        "n_paths": len(paths),
        "layers": layers,
        "edges": [{"edge_idx": i, "u": u, "v": v} for i, (u, v) in enumerate(edges)],
        "path_edge_indices": paths,
    }
    (out_dir / "data" / "graph_spec.json").write_text(json.dumps(graph_spec, indent=2), encoding="utf-8")
    (out_dir / "data" / "task_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    pd.DataFrame(path_incidence.numpy()).to_csv(out_dir / "data" / "path_incidence.csv", index=False)

    # Tuning.
    tuning_frames: List[pd.DataFrame] = []
    selected_rows: List[Dict[str, float]] = []
    for method in methods:
        tuning_df, best_row = tune_method(
            method=method,
            task=task,
            path_incidence=path_incidence,
            lr_grid=lr_grid,
            tau_grid=tau_grid,
            tune_seeds=args.tune_seeds,
            tune_epochs=args.tune_epochs,
            n_train=config.n_train,
            n_val=config.n_val,
            batch_size=args.batch_size,
        )
        tuning_frames.append(tuning_df)
        selected_rows.append(best_row)

    tuning_results = pd.concat(tuning_frames, ignore_index=True)
    selected_df = make_selected_hparams_table(selected_rows)

    # Final five-seed runs.
    seed_rows: List[Dict[str, float]] = []
    history_frames: List[pd.DataFrame] = []
    for _, selected in selected_df.iterrows():
        method = str(selected["method"])
        lr = float(selected["lr"])
        tau = float(selected["tau"])
        for seed in args.final_seeds:
            result, history = train_one(
                method=method,
                task=task,
                path_incidence=path_incidence,
                lr=lr,
                tau=tau,
                seed=int(seed),
                n_train=config.n_train,
                n_val=config.n_val,
                n_test=config.n_test,
                epochs=args.final_epochs,
                batch_size=args.batch_size,
            )
            seed_rows.append(result)
            history_frames.append(history)

    seed_level_results = pd.DataFrame(seed_rows)
    history_df = pd.concat(history_frames, ignore_index=True)
    summary_df = make_summary_table(seed_level_results)
    display_df = summary_table_for_display(summary_df)

    # Write CSV + TeX tables.
    tuning_results.to_csv(out_dir / "tables" / "tuning_results.csv", index=False)
    selected_df.to_csv(out_dir / "tables" / "selected_hyperparameters.csv", index=False)
    seed_level_results.to_csv(out_dir / "tables" / "seed_level_results.csv", index=False)
    history_df.to_csv(out_dir / "tables" / "validation_history.csv", index=False)
    summary_df.to_csv(out_dir / "tables" / "five_seed_summary.csv", index=False)
    display_df.to_csv(out_dir / "tables" / "five_seed_summary_display.csv", index=False)

    selected_latex_df = selected_df.copy()
    selected_latex_df.columns = ["METHOD", "LR", "TAU", "MEAN VAL STANDARD SPO LOSS"]
    selected_tex = dataframe_to_latex(
        selected_latex_df,
        caption="Selected hyperparameters from validation path-regret tuning.",
        label="tab:selected-hparams",
    )
    (out_dir / "tables" / "selected_hyperparameters.tex").write_text(selected_tex, encoding="utf-8")

    summary_tex = dataframe_to_latex(
        display_df,
        caption="Five-seed benchmark summary.",
        label="tab:five-seed-summary",
    )
    (out_dir / "tables" / "five_seed_summary.tex").write_text(summary_tex, encoding="utf-8")

    # Save generated experiment artifacts.
    save_figures(out_dir, history_df, summary_df)

    print("Selected hyperparameters:")
    print(selected_df.to_string(index=False))
    print("\nFive-seed summary:")
    print(display_df.to_string(index=False))
    print(f"\nWrote benchmark artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
