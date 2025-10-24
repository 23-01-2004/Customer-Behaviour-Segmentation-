# customer_income_spending_analysis_app.py

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from src.vintage_theme import set_vintage_theme

# Assume df_clean and set_vintage_theme() are already defined
# df_clean: Your customer dataset
# set_vintage_theme: returns a dictionary of colors for the vintage theme

def income_spending_cluster_analysis(df_clean):
    vintage_colors = set_vintage_theme()
    
    clusters = df_clean['Cluster'].unique()
    n_clusters = len(clusters)
    
    st.header(f"📊 Cluster Characteristics: Income & Spending Analysis ({n_clusters} clusters)")
    st.write(f"Total customers analyzed: {len(df_clean):,}")
    
    # -----------------------------
    # Set up cluster colors
    # -----------------------------
    if n_clusters == 2:
        cluster_colors = [vintage_colors['primary'], vintage_colors['accent1']]
    elif n_clusters == 3:
        cluster_colors = [vintage_colors['primary'], vintage_colors['secondary'], vintage_colors['accent1']]
    else:
        cluster_colors = [vintage_colors['primary'], vintage_colors['secondary'], 
                          vintage_colors['accent1'], vintage_colors['accent2'], 
                          vintage_colors['accent3']][:n_clusters]
    
    cluster_names = [f'Cluster {i}' for i in range(n_clusters)]
    
    # -----------------------------
    # PLOTS
    # -----------------------------
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 14))
    
    # Income Distribution by Cluster
    for i, cluster_id in enumerate(clusters):
        cluster_data = df_clean[df_clean['Cluster'] == cluster_id]
        sns.kdeplot(data=cluster_data, x='Income', fill=True, alpha=0.6,
                    color=cluster_colors[i], label=f'{cluster_names[i]} (n={len(cluster_data):,})', ax=ax1)
        mean_income = cluster_data['Income'].mean()
        ax1.axvline(mean_income, color=cluster_colors[i], linestyle='--', linewidth=2, alpha=0.8)
        
    ax1.set_xlabel('Annual Income ($)')
    ax1.set_ylabel('Density')
    ax1.set_title('Income Distribution by Cluster')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Spending Distribution by Cluster
    for i, cluster_id in enumerate(clusters):
        cluster_data = df_clean[df_clean['Cluster'] == cluster_id]
        sns.kdeplot(data=cluster_data, x='Total_Spending', fill=True, alpha=0.6,
                    color=cluster_colors[i], label=f'{cluster_names[i]} (n={len(cluster_data):,})', ax=ax2)
        mean_spending = cluster_data['Total_Spending'].mean()
        ax2.axvline(mean_spending, color=cluster_colors[i], linestyle='--', linewidth=2, alpha=0.8)
        
    ax2.set_xlabel('Total Spending ($)')
    ax2.set_ylabel('Density')
    ax2.set_title('Spending Distribution by Cluster')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Income vs Spending Scatter Plot
    scatter = sns.scatterplot(data=df_clean, x='Income', y='Total_Spending',
                              hue='Cluster', palette=cluster_colors, s=60, alpha=0.7,
                              edgecolor=vintage_colors['dark'], linewidth=0.5, ax=ax3)
    ax3.set_xlabel('Annual Income ($)')
    ax3.set_ylabel('Total Spending ($)')
    ax3.set_title('Income vs Spending by Cluster')
    ax3.legend(title='Cluster')
    ax3.grid(True, alpha=0.3)
    
    # Box Plot Comparison
    boxplot_data = []
    for cluster_id in clusters:
        cluster_data = df_clean[df_clean['Cluster'] == cluster_id]
        for metric, values in [('Income', cluster_data['Income']), ('Spending', cluster_data['Total_Spending'])]:
            for value in values:
                boxplot_data.append({'Cluster': f'Cluster {cluster_id}', 'Metric': metric, 'Value': value})
    boxplot_df = pd.DataFrame(boxplot_data)
    
    sns.boxplot(data=boxplot_df, x='Cluster', y='Value', hue='Metric',
                palette=[vintage_colors['primary'], vintage_colors['accent1']], ax=ax4)
    ax4.set_xlabel('Cluster')
    ax4.set_ylabel('Amount ($)')
    ax4.set_title('Income & Spending Distribution by Cluster')
    ax4.legend(title='Metric')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # -----------------------------
    # Detailed statistics
    # -----------------------------
    cluster_stats = df_clean.groupby('Cluster').agg({
        'Income': ['count', 'mean', 'median', 'std', 'min', 'max'],
        'Total_Spending': ['mean', 'median', 'std', 'min', 'max'],
        'Spending_Income_Ratio': 'mean',
        'Response': 'mean'
    }).round(2)
    cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns.values]
    cluster_stats = cluster_stats.reset_index()
    
    st.subheader("📋 Cluster Statistics")
    st.dataframe(cluster_stats)
    
    # -----------------------------
    # Cluster profiling & insights
    # -----------------------------
    overall_income = df_clean['Income'].mean()
    overall_spending = df_clean['Total_Spending'].mean()
    overall_ratio = df_clean['Spending_Income_Ratio'].mean()
    
    st.subheader("💡 Cluster Profiles & Strategic Recommendations")
    for _, row in cluster_stats.iterrows():
        cluster_id = int(row['Cluster'])
        income_level = "High" if row['Income_mean'] > overall_income else "Low"
        spending_level = "High" if row['Total_Spending_mean'] > overall_spending else "Low"
        ratio_level = "High" if row['Spending_Income_Ratio_mean'] > overall_ratio else "Low"
        
        if income_level == "High" and spending_level == "High":
            profile = "Premium Customers"
        elif income_level == "High" and spending_level == "Low":
            profile = "Wealthy Savers"
        elif income_level == "Low" and spending_level == "High":
            profile = "Value-focused Spenders"
        else:
            profile = "Budget-conscious Customers"
        
        st.markdown(f"**Cluster {cluster_id}**: {profile}")
        st.markdown(f"• Income: {income_level} (${row['Income_mean']:,.0f})")
        st.markdown(f"• Spending: {spending_level} (${row['Total_Spending_mean']:,.0f})")
        st.markdown(f"• Spending Efficiency: {ratio_level} ({row['Spending_Income_Ratio_mean']:.1f}% of income)")
        st.markdown(f"• Response Rate: {row['Response_mean']*100:.1f}%")
    
    # Total revenue impact
    total_revenue = df_clean['Total_Spending'].sum()
    st.subheader("💰 Potential Revenue Impact by Cluster")
    revenue_data = []
    for _, row in cluster_stats.iterrows():
        cluster_id = int(row['Cluster'])
        cluster_revenue = row['Total_Spending_mean'] * row['Income_count']
        revenue_share = (cluster_revenue / total_revenue) * 100
        revenue_data.append({'Cluster': f'Cluster {cluster_id}', 'Revenue ($)': cluster_revenue, 'Share (%)': revenue_share})
    
    revenue_df = pd.DataFrame(revenue_data)
    st.dataframe(revenue_df)
    
    st.success("✅ Income & Spending Cluster Analysis Complete!")
    return df_clean


# -----------------------------
# STREAMLIT APP
# -----------------------------
# st.title("Customer Income & Spending Cluster Analysis")

# df_clean = income_spending_cluster_analysis(df_clean)
