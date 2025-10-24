# dynamic_cluster_analysis_app.py

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from src.vintage_theme import set_vintage_theme

# Assume df_clean and set_vintage_theme() are already defined
# df_clean: Your customer dataset
# set_vintage_theme: returns a dictionary of colors for the vintage theme

def analyze_cluster_characteristics_dynamic_streamlit(df_clean):
    """
    Streamlit-friendly dynamic cluster characteristics analysis
    """
    vintage_colors = set_vintage_theme()
    n_clusters = len(df_clean['Cluster'].unique())
    
    st.header(f"📊 Dynamic Cluster Characteristics Analysis ({n_clusters} Clusters)")
    st.write(f"Total customers analyzed: {len(df_clean):,}")
    
    # Cluster colors and names
    color_palette = [vintage_colors['primary'], vintage_colors['secondary'], 
                     vintage_colors['accent1'], vintage_colors['accent2'],
                     vintage_colors['accent3'], vintage_colors['accent4'],
                     vintage_colors['neutral'], vintage_colors['light']]
    cluster_colors = color_palette[:n_clusters]
    cluster_names = [f'Cluster {i}' for i in range(n_clusters)]
    
    # -----------------------------
    # Create multi-panel figure
    # -----------------------------
    fig = plt.figure(figsize=(20, 16))
    if n_clusters <= 3:
        gs = fig.add_gridspec(2, 3)
    else:
        gs = fig.add_gridspec(3, 3)

    # 1️⃣ 2D Density Plot
    ax1 = fig.add_subplot(gs[0, 0])
    for i, cluster_id in enumerate(df_clean['Cluster'].unique()):
        cluster_data = df_clean[df_clean['Cluster'] == cluster_id]
        sns.kdeplot(data=cluster_data, x='Age', y='Total_Spending', 
                    color=cluster_colors[i], label=cluster_names[i],
                    alpha=0.6, fill=True, ax=ax1, levels=5)
    ax1.set_xlabel('Age')
    ax1.set_ylabel('Total Spending ($)')
    ax1.set_title('2D Density: Age vs Spending')
    ax1.legend()

    # 2️⃣ Violin + Swarm Plot
    ax2 = fig.add_subplot(gs[0, 1])
    sns.violinplot(data=df_clean, x='Cluster', y='Total_Spending', 
                   palette=cluster_colors, inner='quartile', ax=ax2)
    if n_clusters <= 4:
        sns.swarmplot(data=df_clean, x='Cluster', y='Total_Spending', 
                      color=vintage_colors['dark'], size=2, alpha=0.6, ax=ax2)
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Total Spending ($)')
    ax2.set_title('Spending Distribution: Violin Plot')

    # 3️⃣ Hexbin + Regression
    ax3 = fig.add_subplot(gs[0, 2])
    cmaps = ['Blues', 'Reds', 'Greens', 'Purples', 'Oranges', 'Greys', 'YlOrBr', 'BuPu']
    for i, cluster_id in enumerate(df_clean['Cluster'].unique()):
        cluster_data = df_clean[df_clean['Cluster'] == cluster_id]
        ax3.hexbin(cluster_data['Age'], cluster_data['Total_Spending'], 
                   gridsize=20, cmap=cmaps[i % len(cmaps)], alpha=0.7, mincnt=1)
        z = np.polyfit(cluster_data['Age'], cluster_data['Total_Spending'], 1)
        p = np.poly1d(z)
        age_range = np.linspace(cluster_data['Age'].min(), cluster_data['Age'].max(), 100)
        ax3.plot(age_range, p(age_range), color=cluster_colors[i], linewidth=3)
    ax3.set_xlabel('Age')
    ax3.set_ylabel('Total Spending ($)')
    ax3.set_title('Hexbin + Regression: Age vs Spending')

    # 4️⃣ Radar Chart or Boxplot
    if 3 <= n_clusters <= 6:
        ax4 = fig.add_subplot(gs[1, 0], polar=True)
        categories = ['Avg Age', 'Avg Spending', 'Spending/Age Ratio', 'Income', 'Response Rate']
        N = len(categories)
        stats_to_plot = []
        for cluster_id in df_clean['Cluster'].unique():
            cluster_data = df_clean[df_clean['Cluster'] == cluster_id]
            profile = [
                cluster_data['Age'].mean() / 100,
                cluster_data['Total_Spending'].mean() / 2000,
                (cluster_data['Total_Spending'].mean() / cluster_data['Age'].mean()) * 10,
                cluster_data['Income'].mean() / 100000,
                cluster_data['Response'].mean() * 100
            ]
            stats_to_plot.append(profile)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        for i, profile in enumerate(stats_to_plot):
            profile += profile[:1]
            ax4.plot(angles, profile, color=cluster_colors[i], linewidth=3)
            ax4.fill(angles, profile, color=cluster_colors[i], alpha=0.25)
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(categories)
        ax4.set_title('Cluster Profile Radar', pad=20)
    else:
        ax4 = fig.add_subplot(gs[1, 0])
        sns.boxplot(data=df_clean, x='Cluster', y='Age', palette=cluster_colors, ax=ax4)
        ax4.set_xlabel('Cluster')
        ax4.set_ylabel('Age')
        ax4.set_title('Age Distribution by Cluster')

    # 5️⃣ 3D Scatter Plot
    ax5 = fig.add_subplot(gs[1, 1], projection='3d')
    for i, cluster_id in enumerate(df_clean['Cluster'].unique()):
        cluster_data = df_clean[df_clean['Cluster'] == cluster_id]
        if len(cluster_data) > 100:
            cluster_data = cluster_data.sample(100, random_state=42)
        ax5.scatter(cluster_data['Age'], cluster_data['Income'], cluster_data['Total_Spending'],
                   color=cluster_colors[i], alpha=0.6, s=30)
    ax5.set_xlabel('Age')
    ax5.set_ylabel('Income ($)')
    ax5.set_zlabel('Spending ($)')
    ax5.set_title('3D: Age, Income, Spending')

    # 6️⃣ Joint Distribution / Response Rate
    ax6 = fig.add_subplot(gs[1, 2])
    if n_clusters <= 4:
        for i, cluster_id in enumerate(df_clean['Cluster'].unique()):
            cluster_data = df_clean[df_clean['Cluster'] == cluster_id]
            ax6.scatter(cluster_data['Age'], cluster_data['Total_Spending'],
                       color=cluster_colors[i], alpha=0.6, s=20)
        ax6.set_xlabel('Age')
        ax6.set_ylabel('Total Spending ($)')
        ax6.set_title('Joint Distribution: Age vs Spending')
    else:
        response_rates = df_clean.groupby('Cluster')['Response'].mean() * 100
        bars = ax6.bar(range(len(response_rates)), response_rates.values, color=cluster_colors)
        ax6.set_xlabel('Cluster')
        ax6.set_ylabel('Response Rate (%)')
        ax6.set_title('Response Rates by Cluster')
        ax6.set_xticks(range(len(response_rates)))
        ax6.set_xticklabels([f'Cluster {i}' for i in response_rates.index])
        for bar, rate in zip(bars, response_rates.values):
            ax6.text(bar.get_x() + bar.get_width()/2., rate + 0.5, f'{rate:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    st.pyplot(fig)

    # -----------------------------
    # Cluster profiles summary table
    # -----------------------------
    summary_list = []
    for i, cluster_id in enumerate(df_clean['Cluster'].unique()):
        cluster_data = df_clean[df_clean['Cluster'] == cluster_id]
        summary_list.append({
            'Cluster': cluster_names[i],
            'Customers': len(cluster_data),
            'Avg Age': round(cluster_data['Age'].mean(), 1),
            'Avg Spending': round(cluster_data['Total_Spending'].mean(), 0),
            'Avg Income': round(cluster_data['Income'].mean(), 0),
            'Age-Spending Corr': round(cluster_data['Age'].corr(cluster_data['Total_Spending']), 3),
            'Response Rate (%)': round(cluster_data['Response'].mean() * 100, 1)
        })
    summary_df = pd.DataFrame(summary_list)
    st.subheader("📋 Cluster Profiles Summary")
    st.dataframe(summary_df)

    # Comparative insights
    st.subheader("💡 Comparative Insights")
    spending_by_cluster = df_clean.groupby('Cluster')['Total_Spending'].mean()
    age_by_cluster = df_clean.groupby('Cluster')['Age'].mean()
    st.write(f"• Highest Spending Cluster: Cluster {spending_by_cluster.idxmax()} (${spending_by_cluster.max():,.0f})")
    st.write(f"• Lowest Spending Cluster: Cluster {spending_by_cluster.idxmin()} (${spending_by_cluster.min():,.0f})")
    st.write(f"• Youngest Cluster: Cluster {age_by_cluster.idxmin()} ({age_by_cluster.min():.1f} years)")
    st.write(f"• Oldest Cluster: Cluster {age_by_cluster.idxmax()} ({age_by_cluster.max():.1f} years)")

    st.success("✅ Dynamic Cluster Analysis Complete!")


# -----------------------------
# STREAMLIT APP
# -----------------------------
# st.title("Dynamic Customer Cluster Analysis")
# analyze_cluster_characteristics_dynamic_streamlit(df_clean)
