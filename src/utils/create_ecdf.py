# Code from https://www.bomberbot.com/python/creating-informative-
# ecdf-plots-with-seaborn-a-comprehensive-python-tutorial/
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_ecdf(df: pd.DataFrame, name: str) -> None:

    # Melt the DataFrame for easier plotting
    df_melted = df.melt(var_name="Distribution", value_name="Value")

    # Create ECDF plot
    plt.figure(figsize=(12, 7))
    sns.ecdfplot(data=df_melted, x="Value", hue="Distribution")
    plt.title("Comparison of Three Normal Distributions", fontsize=16)
    plt.xlabel("Value", fontsize=12)
    plt.ylabel("Cumulative Proportion", fontsize=12)
    plt.legend(title="Distribution", title_fontsize="12", fontsize="10")
    plt.savefig(f"{name}_ecdf.png")
