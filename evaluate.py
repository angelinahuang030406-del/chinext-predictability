"""Figures for the README, built from the result tables."""

import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

LABELS = {
    "composite-csi300": "Composite - CSI300",
    "chinext-comp": "ChiNext - Comp",
    "chinext50-comp": "ChiNext50 - Comp",
    "chinext300-comp": "ChiNext300 - Comp",
    "basic-comp": "Basic - Comp",
}
# spreads whose in-sample regression had a factor with p < 0.05
INSAMPLE_SIG = {"composite-csi300", "chinext50-comp", "basic-comp"}


def fig_oos():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "oos_summary.csv"))
    df["label"] = df["spread"].map(LABELS)
    colors = ["#c0392b" if s in INSAMPLE_SIG else "#7f8c8d" for s in df["spread"]]

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.barh(df["label"], df["oos_r2"] * 100, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Out-of-sample R² (%) vs historical-mean benchmark")
    ax.set_title("In-sample significant spreads (red) fail out of sample")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "fig_oos.png"), dpi=150)


def fig_size_tilt():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "loadings.csv")).dropna(subset=["s_size"])
    order = df.sort_values("s_size")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.barh(order["index"], order["s_size"], color="#2c3e50")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Size loading (largecap-minus-composite factor)")
    ax.set_title("Sector betas are all ~1.0; size tilt is the only real difference")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "fig_size_tilt.png"), dpi=150)


if __name__ == "__main__":
    fig_oos()
    fig_size_tilt()
    print("figures written to results/")
