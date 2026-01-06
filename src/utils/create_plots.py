import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_ecdf(df: pd.DataFrame, name: str) -> None:
    df_melted = df.melt(var_name="Distribution", value_name="Value")

    plt.figure(figsize=(12, 8))
    sns.ecdfplot(data=df_melted, x='Value', hue='Distribution', legend='full')
    plt.savefig(f"{name}_ecdf.jpg")


def plot_history(df: pd.DataFrame, name: str):
    plt.figure(figsize=(12, 8))

    for col in df.columns:
        plt.plot(df.index, df[col], label=col)
    plt.xlabel("Observation Order")
    plt.ylabel("Value")
    plt.title("Historical Value Evolution")
    plt.legend()
    plt.savefig(f"{name}_history.jpg")
