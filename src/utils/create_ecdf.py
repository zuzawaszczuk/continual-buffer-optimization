import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_ecdf(df: pd.DataFrame, name: str) -> None:
    df_melted = df.melt(var_name="Distribution", value_name="Value")

    plt.figure(figsize=(12, 8))
    sns.ecdfplot(data=df_melted, x='Value', hue='Distribution', legend='full')
    plt.savefig(f"{name}_ecdf.jpg")
