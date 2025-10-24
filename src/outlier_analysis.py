import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import sys 
sys.path.append("../")
from src.vintage_theme import set_vintage_theme


def detect_outliers_iqr(df, column):
    """Detect outliers using the IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower) | (df[column] > upper)]
    return outliers, lower, upper


def outlier_analysis(df_clean):
    """Perform full outlier detection, removal & visualization."""
    vintage_colors = set_vintage_theme()
    df_before_outliers = df_clean.copy()

    current_year = 2025
    df_clean['Age'] = current_year - df_clean['Year_Birth']

    df_clean['Total Spending'] = (
        df_clean['MntWines'] + df_clean['MntFruits'] +
        df_clean['MntMeatProducts'] + df_clean['MntFishProducts'] +
        df_clean['MntSweetProducts'] + df_clean['MntGoldProds']
    )

    numerical_vars = ['Income', 'Total Spending', 'Age', 'Recency', 
                      'MntWines', 'MntMeatProducts', 'MntGoldProds']

    outliers_summary = {}
    total_outliers_indices = set()

    for var in numerical_vars:
        outliers, low, up = detect_outliers_iqr(df_clean, var)
        outliers_summary[var] = {
            'count': len(outliers),
            'percentage': (len(outliers) / len(df_clean)) * 100,
            'lower_bound': low,
            'upper_bound': up,
            'outlier_indices': set(outliers.index)
        }
        total_outliers_indices.update(outliers.index)

    # Remove all outlier rows
    df_no_outliers = df_clean.drop(total_outliers_indices)

    verification = {
        "records_before": len(df_clean),
        "records_after": len(df_no_outliers),
        "records_removed": len(df_clean) - len(df_no_outliers),
        "data_retained_pct": (len(df_no_outliers) / len(df_clean)) * 100,
    }

    return df_no_outliers, outliers_summary, verification, numerical_vars, vintage_colors


def plot_outliers(df, numerical_vars, vintage_colors, title, color_key):
    """Return a matplotlib figure for boxplots."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, var in enumerate(numerical_vars):
        if i < len(axes):
            axes[i].boxplot(
                df[var], patch_artist=True,
                boxprops=dict(facecolor=vintage_colors[color_key], alpha=0.7),
                flierprops=dict(marker='o', markerfacecolor=vintage_colors['alert'], 
                               markersize=6, markeredgecolor=vintage_colors['dark'])
            )
            axes[i].set_title(var, fontsize=10, fontweight='bold')
            axes[i].set_ylabel(var)
            mean_val = df[var].mean()
            axes[i].axhline(mean_val, color=vintage_colors['accent2'], linestyle='--', linewidth=1)
    for i in range(len(numerical_vars), len(axes)):
        fig.delaxes(axes[i])
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    return fig
