#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


METHOD_ORDER = [
    "Plasmod HNSW",
    "Milvus HNSW",
    "Qdrant HNSW",
    "ChromaDB HNSW",
    "LanceDB IVF_HNSW_PQ",
]

METHOD_LABELS = {
    "Plasmod HNSW": "Plasmod",
    "Milvus HNSW": "Milvus",
    "Qdrant HNSW": "Qdrant",
    "ChromaDB HNSW": "ChromaDB",
    "LanceDB IVF_HNSW_PQ": "LanceDB",
}

METHOD_COLORS = {
    "Plasmod HNSW": "#2563eb",
    "Milvus HNSW": "#dc2626",
    "Qdrant HNSW": "#059669",
    "ChromaDB HNSW": "#9333ea",
    "LanceDB IVF_HNSW_PQ": "#ca8a04",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def plot_curve(points_csv: Path, frontier_csv: Path, output: Path, title: str, log_y: bool) -> None:
    import matplotlib.pyplot as plt

    points = read_csv(points_csv)
    frontier = read_csv(frontier_csv)

    plt.figure(figsize=(10, 6), dpi=180)
    for method in METHOD_ORDER:
        method_points = [row for row in points if row["method"] == method]
        method_frontier = sorted(
            [row for row in frontier if row["method"] == method],
            key=lambda row: float(row["recall"]),
        )
        if not method_points and not method_frontier:
            continue

        color = METHOD_COLORS[method]
        label = METHOD_LABELS[method]
        if method_points:
            plt.scatter(
                [float(row["recall"]) for row in method_points],
                [float(row["batch_qps"]) for row in method_points],
                s=12,
                alpha=0.22,
                color=color,
                linewidths=0,
            )
        if method_frontier:
            plt.plot(
                [float(row["recall"]) for row in method_frontier],
                [float(row["batch_qps"]) for row in method_frontier],
                marker="o",
                markersize=3,
                linewidth=1.5,
                color=color,
                label=label,
            )

    if log_y:
        plt.yscale("log")
        ylabel = "Batch QPS (log scale)"
    else:
        ylabel = "Batch QPS"

    plt.xlabel("Recall@10")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", alpha=0.25)
    plt.legend(title="Database", fontsize=8)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot HNSW recall/QPS curve from generated curve CSV files.")
    parser.add_argument(
        "--results-dir",
        default="results/hnsw_recall_sweep_deep10M_n1000000_q10000_k10_controlled_complete_20260604",
        help="Directory containing curve_points.csv and curve_frontier.csv.",
    )
    parser.add_argument("--points-csv", default="", help="Override path to curve_points.csv.")
    parser.add_argument("--frontier-csv", default="", help="Override path to curve_frontier.csv.")
    parser.add_argument("--output", default="", help="Output PNG path.")
    parser.add_argument(
        "--title",
        default="HNSW Recall/QPS Curve, deep10M 1M index / 10k queries / topk=10",
    )
    parser.add_argument("--linear-y", action="store_true", help="Use a linear Y axis instead of log scale.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    points_csv = Path(args.points_csv) if args.points_csv else results_dir / "curve_points.csv"
    frontier_csv = Path(args.frontier_csv) if args.frontier_csv else results_dir / "curve_frontier.csv"
    output = Path(args.output) if args.output else results_dir / "recall_qps_curve_db_legend.png"

    plot_curve(points_csv, frontier_csv, output, args.title, log_y=not args.linear_y)
    print(output)


if __name__ == "__main__":
    main()
