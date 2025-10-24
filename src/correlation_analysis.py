import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def analyze_correlations(df_clean, vintage_colors):
    """Generate correlation heatmap and structured statistics."""

    numerical_features = [
        'Year_Birth', 'Income', 'Kidhome', 'Teenhome', 'Recency',
        'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
        'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases',
        'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
        'NumWebVisitsMonth', 'Total_Spending', 'Age', 'Children_Count'
    ]

    corr_matrix = df_clean[numerical_features].corr()

    # --- Plot Heatmap ---
    plt.figure(figsize=(16, 14))
    heatmap = sns.heatmap(
        corr_matrix,
        annot=True, fmt=".2f",
        cmap=sns.light_palette(vintage_colors['primary'], as_cmap=True),
        center=0, square=True,
        cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'},
        linewidths=0.5, linecolor=vintage_colors['light']
    )

    plt.title("Correlation Heatmap of Numerical Features", fontsize=16, fontweight="bold", pad=20)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    # --- DataFrames for Stats ---
    strong_pos, strong_neg = [], []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            feat1, feat2 = corr_matrix.columns[i], corr_matrix.columns[j]
            corr = corr_matrix.iloc[i, j]
            if corr > 0.5:
                strong_pos.append((feat1, feat2, round(corr, 3)))
            elif corr < -0.3:
                strong_neg.append((feat1, feat2, round(corr, 3)))

    df_pos = pd.DataFrame(strong_pos, columns=["Feature 1", "Feature 2", "Correlation"]).sort_values("Correlation", ascending=False)
    df_neg = pd.DataFrame(strong_neg, columns=["Feature 1", "Feature 2", "Correlation"]).sort_values("Correlation")

    # --- Income Relationships ---
    income_corr = corr_matrix["Income"].sort_values(ascending=False)
    df_income = pd.DataFrame({
        "Feature": income_corr.index,
        "Correlation with Income": income_corr.values
    })
    df_income = df_income[df_income["Feature"] != "Income"].reset_index(drop=True)
    df_income["Correlation with Income"] = df_income["Correlation with Income"].round(3)

    # --- High Correlation Clusters ---
    high_corr_pairs = [
        (f1, f2, round(corr, 3))
        for i, f1 in enumerate(corr_matrix.columns)
        for j, f2 in enumerate(corr_matrix.columns)
        if i < j and abs(corr_matrix.iloc[i, j]) > 0.6
    ]
    df_clusters = pd.DataFrame(high_corr_pairs, columns=["Feature 1", "Feature 2", "Correlation"]).sort_values(
        "Correlation", ascending=False
    )

    # --- Key Insights (text summary) ---
    insights = []
    web_corr = corr_matrix.loc["NumWebVisitsMonth", "NumStorePurchases"]
    insights.append(f"Website visits vs store purchases correlation: {web_corr:.3f}")
    insights.append("Strong positive correlations indicate related spending patterns.")
    insights.append("Negative correlations often suggest inverse behaviors like web vs store visits.")

    return plt, df_pos, df_neg, df_income, df_clusters, insights
