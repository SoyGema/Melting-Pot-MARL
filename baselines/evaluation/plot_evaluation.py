"""Plot per capita cumulative return for focal vs background populations.

Reproduces Fig. 2-style evaluation plots: one subplot per substrate,
focal (red) and background (gray) lines with mean ± std bands across runs.

Expected directory structure
-----------------------------
<results_dir>/
    <substrate_name>/
        results_evals_0.csv    # one CSV per run/seed
        results_evals_1.csv
        ...

Each CSV is produced by baselines/evaluation/evaluate.py and has one row
per episode with at least the columns:
    focal_per_capita_return      (float)
    background_per_capita_return (float)

Usage
-----
python baselines/evaluation/plot_evaluation.py \\
    --results_dir ./results/eval \\
    --focal_label "farmer" \\
    --background_label "private_property" \\
    --output focal_vs_background.png
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


FOCAL_COLOR = "#E05555"       # red
BACKGROUND_COLOR = "#666666"  # dark gray
ALPHA_BAND = 0.25


def load_runs(substrate_dir: str) -> list[pd.DataFrame]:
    """Load all CSV files from a substrate directory, one per run."""
    csv_files = sorted(
        os.path.join(substrate_dir, f)
        for f in os.listdir(substrate_dir)
        if f.endswith(".csv")
    )
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {substrate_dir}")
    return [pd.read_csv(f) for f in csv_files]


def cumulative_per_capita(dfs: list[pd.DataFrame], col: str) -> np.ndarray:
    """Return cumulative per capita return array of shape (n_runs, n_episodes)."""
    cumulative_runs = []
    for df in dfs:
        cumulative_runs.append(df[col].values.cumsum())
    # Align to the shortest run in case episode counts differ
    min_len = min(len(r) for r in cumulative_runs)
    return np.stack([r[:min_len] for r in cumulative_runs])  # (n_runs, n_episodes)


def plot_substrate(ax, dfs: list[pd.DataFrame], substrate_name: str,
                   focal_label: str, background_label: str):
    has_background = all(
        "background_per_capita_return" in df.columns and df["background_per_capita_return"].notna().any()
        for df in dfs
    )

    focal_curves = cumulative_per_capita(dfs, "focal_per_capita_return")
    focal_mean = focal_curves.mean(axis=0)
    focal_std = focal_curves.std(axis=0)
    episodes = np.arange(1, len(focal_mean) + 1)

    ax.plot(episodes, focal_mean, color=FOCAL_COLOR, linewidth=1.5, label=focal_label)
    ax.fill_between(episodes,
                    focal_mean - focal_std,
                    focal_mean + focal_std,
                    color=FOCAL_COLOR, alpha=ALPHA_BAND)

    if has_background:
        bg_curves = cumulative_per_capita(dfs, "background_per_capita_return")
        bg_mean = bg_curves.mean(axis=0)
        bg_std = bg_curves.std(axis=0)

        ax.plot(episodes, bg_mean, color=BACKGROUND_COLOR, linewidth=1.5, label=background_label)
        ax.fill_between(episodes,
                        bg_mean - bg_std,
                        bg_mean + bg_std,
                        color=BACKGROUND_COLOR, alpha=ALPHA_BAND)

    ax.set_title(f"Substrate {substrate_name}", fontsize=9)
    ax.set_xlabel("Episode", fontsize=8)
    ax.set_ylabel("Per capita return", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main(args):
    results_dir = args.results_dir

    if args.substrates:
        substrate_names = args.substrates
    else:
        substrate_names = sorted(
            d for d in os.listdir(results_dir)
            if os.path.isdir(os.path.join(results_dir, d))
        )

    if not substrate_names:
        raise ValueError(f"No substrate subdirectories found in {results_dir}")

    n = len(substrate_names)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, substrate_name in zip(axes, substrate_names):
        substrate_dir = os.path.join(results_dir, substrate_name)
        dfs = load_runs(substrate_dir)
        plot_substrate(ax, dfs, substrate_name,
                       focal_label=args.focal_label,
                       background_label=args.background_label)

    # Shared legend
    focal_patch = mpatches.Patch(color=FOCAL_COLOR, label=args.focal_label)
    bg_patch = mpatches.Patch(color=BACKGROUND_COLOR, label=args.background_label)
    fig.legend(handles=[focal_patch, bg_patch],
               loc="upper center", ncol=2, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 1.02))

    title = f"{args.focal_label} as focal agent and {args.background_label} as background agent"
    fig.suptitle(title, fontsize=10, y=1.06)

    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, bbox_inches="tight", dpi=150)
        print(f"Figure saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot focal vs background per capita cumulative return across substrates."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Root directory containing one subdirectory per substrate, each with CSV files.",
    )
    parser.add_argument(
        "--substrates",
        type=str,
        nargs="+",
        default=None,
        help="Substrate names to plot (subdirectory names). Defaults to all subdirs found.",
    )
    parser.add_argument(
        "--focal_label",
        type=str,
        default="focal",
        help="Label for the focal population (e.g. 'farmer').",
    )
    parser.add_argument(
        "--background_label",
        type=str,
        default="background",
        help="Label for the background population (e.g. 'private_property').",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the figure (e.g. plot.png). If not set, displays interactively.",
    )

    args = parser.parse_args()
    main(args)
