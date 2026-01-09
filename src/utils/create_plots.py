from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_ecdf(df: pd.DataFrame, name: str) -> None:
    df_melted = df.melt(var_name="Distribution", value_name="Value")

    plt.figure(figsize=(12, 8))
    sns.ecdfplot(data=df_melted, x="Value", hue="Distribution", legend=True)
    plt.savefig(f"{name}/ecdf.jpg")


def plot_history(df: pd.DataFrame, name: str) -> None:
    plt.figure(figsize=(12, 8))

    for col in df.columns:
        plt.plot(df.index, df[col], label=col)
    plt.xlabel("Observation Order")
    plt.ylabel("Value")
    plt.title("Historical Value Evolution")
    plt.legend()
    plt.savefig(f"{name}/history.jpg")


def plot_heatmaps(
    dataframes: List[pd.DataFrame],
    titles: List[str],
    folder_name: str,
    cmap: str = "Blues",
) -> None:
    n = len(dataframes)

    n_cols = n
    n_rows = 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    for i, data in enumerate(dataframes):
        sns.heatmap(data, ax=axes[i], cmap=cmap, xticklabels=True, yticklabels=True)
        axes[i].set_title(titles[i])

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(f"{folder_name}/masks_heatmap.png")
