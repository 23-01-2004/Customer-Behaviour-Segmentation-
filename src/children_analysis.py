# src/children_analysis.py

import matplotlib.pyplot as plt
import seaborn as sns

def analyze_children_distribution(df_clean, vintage_colors):
    """Analyze and visualize children distribution and spending patterns."""
    df_clean['Children_Count'] = df_clean['Kidhome'] + df_clean['Teenhome']
    df_clean['Total_Spending'] = (df_clean['MntWines'] + df_clean['MntFruits'] + 
                                  df_clean['MntMeatProducts'] + df_clean['MntFishProducts'] + 
                                  df_clean['MntSweetProducts'] + df_clean['MntGoldProds'])
    
    children_distribution = df_clean['Children_Count'].value_counts().sort_index()
    total_customers = len(df_clean)

    # --- PLOT ---
    plt.figure(figsize=(12, 8))
    bars = plt.bar(children_distribution.index, children_distribution.values,
                   color=[vintage_colors['primary'], vintage_colors['secondary'], 
                          vintage_colors['accent1'], vintage_colors['accent2']],
                   alpha=0.8, edgecolor=vintage_colors['dark'], linewidth=1.5)

    for bar, value in zip(bars, children_distribution.values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                 f'{value:,}', ha='center', va='bottom',
                 fontsize=14, fontweight='bold', color=vintage_colors['dark'])

    plt.title('Distribution of Customer Children\nVintage Style', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Number of Children', fontsize=12)
    plt.ylabel('Number of Customers', fontsize=12)
    plt.xticks(children_distribution.index)
    plt.grid(True, alpha=0.3, axis='y')
    sns.despine(left=True)
    plt.tight_layout()

    # --- PRINT STATS ---
    stats_output = []
    stats_output.append("\n CHILDREN DISTRIBUTION DETAILS:")
    stats_output.append("=" * 40)
    for children_count, count in children_distribution.items():
        percentage = (count / total_customers) * 100
        stats_output.append(f"{children_count} Children: {count:>4} customers ({percentage:5.1f}%)")

    stats_output.append(f"\n SUMMARY:")
    stats_output.append(f"Total Customers: {total_customers:,}")
    stats_output.append(f"Customers with children: {(total_customers - children_distribution[0]):,}")
    stats_output.append(f"Percentage with children: {((total_customers - children_distribution[0]) / total_customers * 100):.1f}%")
    stats_output.append(f"Average children per customer: {df_clean['Children_Count'].mean():.2f}")

    stats_output.append(f"\n SPENDING PATTERNS BY CHILDREN COUNT:")
    stats_output.append("=" * 50)
    spending_by_children = df_clean.groupby('Children_Count').agg({
        'Total_Spending': ['mean', 'median', 'count'],
        'Income': 'mean'
    }).round(0)
    spending_by_children.columns = ['Avg_Spending', 'Median_Spending', 'Count', 'Avg_Income']

    for children_count, row in spending_by_children.iterrows():
        stats_output.append(f"\n{children_count} Children:")
        stats_output.append(f"  Customers    : {row['Count']:>4}")
        stats_output.append(f"  Avg Income   : ${row['Avg_Income']:>10,.0f}")
        stats_output.append(f"  Avg Spending : ${row['Avg_Spending']:>10,.0f}")
        stats_output.append(f"  Med Spending : ${row['Median_Spending']:>10,.0f}")

    return plt, "\n".join(stats_output)
