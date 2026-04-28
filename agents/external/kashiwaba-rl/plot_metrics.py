"""
绘制训练 loss 曲线和 episode_reward_mean（return）曲线。
用法：
    python plot_metrics.py                        # 使用默认路径
    python plot_metrics.py --csv path/to/metrics.csv --out plots/
"""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def moving_average(series, window: int):
    return series.rolling(window=window, min_periods=1).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="artifacts/orbit_wars_ppo/metrics.csv",
        help="Path to metrics CSV (default: artifacts/orbit_wars_ppo/metrics.csv)",
    )
    parser.add_argument(
        "--out",
        default="artifacts/orbit_wars_ppo",
        help="Directory to save plots (default: same as CSV dir)",
    )
    parser.add_argument("--window", type=int, default=20, help="Smoothing window size (default: 20)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    window = args.window

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("PPO Training — orbit-wars-lab (kashiwaba-rl)", fontsize=14, fontweight="bold")

    # --- Loss ---
    ax = axes[0]
    ax.plot(df["update"], df["loss"], alpha=0.3, color="#4C72B0", linewidth=0.8, label="loss (raw)")
    ax.plot(df["update"], moving_average(df["loss"], window), color="#4C72B0", linewidth=1.8,
            label=f"loss (MA{window})")
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Loss Curve")

    # --- Episode Reward Mean (Return) ---
    ax = axes[1]
    ax.plot(df["update"], df["episode_reward_mean"], alpha=0.3, color="#DD8452", linewidth=0.8,
            label="reward mean (raw)")
    ax.plot(df["update"], moving_average(df["episode_reward_mean"], window), color="#DD8452",
            linewidth=1.8, label=f"reward mean (MA{window})")
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_ylabel("Episode Reward Mean")
    ax.set_xlabel("Update")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Return Curve (Episode Reward Mean)")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))

    plt.tight_layout()
    out_path = out_dir / "training_curves.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
