# customer_cluster_analysis_app.py

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from src.vintage_theme import set_vintage_theme

# -------------------------
# Assume df_clean and set_vintage_theme() are defined/imported
# df_clean: Your customer dataset
# set_vintage_theme: function returning a dict of colors
# -------------------------

def analyze_custom_clusters(df_clean, n_clusters=3):
    vintage_colors = set_vintage_theme()
    
    st.header(f" Customer Cluster Analysis (K={n_clusters})")
    
    numerical_features = ['Income', 'Total_Spending', 'Age', 'Children_Count', 
                          'MntWines', 'MntMeatProducts', 'NumWebPurchases', 
                          'NumStorePurchases', 'Recency']
    
    X = df_clean[numerical_features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    df_clean['Cluster'] = cluster_labels
    
    cluster_distribution = df_clean['Cluster'].value_counts().sort_index()
    total_customers = len(df_clean)
    percentages = (cluster_distribution / total_customers) * 100
    
    # PLOTS
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
    
    color_palette = [vintage_colors['primary'], vintage_colors['secondary'], 
                     vintage_colors['accent1'], vintage_colors['accent2'],
                     vintage_colors['accent3'], vintage_colors['accent4'],
                     vintage_colors['neutral'], vintage_colors['light']]
    colors = color_palette[:n_clusters]
    
    # Bar chart
    bars = ax1.bar(range(len(cluster_distribution)), cluster_distribution.values,
                   color=colors, alpha=0.8, edgecolor=vintage_colors['dark'], linewidth=1.5)
    ax1.set_xticks(range(len(cluster_distribution)))
    ax1.set_xticklabels([f'Cluster {i}' for i in cluster_distribution.index])
    ax1.set_title('Customer Distribution by Cluster')
    
    # Pie chart
    ax2.pie(cluster_distribution.values, labels=[f'Cluster {i}' for i in cluster_distribution.index],
            colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Cluster Distribution - Percentage View')
    
    # Donut chart
    wedges, texts = ax3.pie(cluster_distribution.values, colors=colors, startangle=90)
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    ax3.add_artist(centre_circle)
    ax3.set_title('Cluster Distribution - Donut Chart')
    
    # Response rate chart
    cluster_response_rates = df_clean.groupby('Cluster')['Response'].mean() * 100
    ax4.bar(range(len(cluster_response_rates)), cluster_response_rates.values, color=colors, alpha=0.8)
    ax4.set_xticks(range(len(cluster_response_rates)))
    ax4.set_xticklabels([f'Cluster {i}' for i in cluster_response_rates.index])
    ax4.set_ylabel('Response Rate (%)')
    ax4.set_title('Campaign Response Rate by Cluster')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display cluster stats in Streamlit
    st.subheader(" Cluster Size Distribution")
    cluster_sizes_df = pd.DataFrame({
        "Cluster": cluster_distribution.index,
        "Customers": cluster_distribution.values,
        "Percentage": percentages.values.round(1)
    })
    st.dataframe(cluster_sizes_df)
    
    st.subheader(" Key Insights")
    st.write(f"Largest Cluster: Cluster {cluster_distribution.idxmax()} ({cluster_distribution.max()} customers)")
    st.write(f"Smallest Cluster: Cluster {cluster_distribution.idxmin()} ({cluster_distribution.min()} customers)")
    st.write(f"Best Response Rate: Cluster {cluster_response_rates.idxmax()} ({cluster_response_rates.max():.1f}%)")
    
    return df_clean


# # -------------------------
# # STREAMLIT APP
# # -------------------------
# st.title("Customer Segmentation Dashboard")

# # Select number of clusters
# n_clusters = st.slider("Select number of clusters:", 2, 8, 3)

# # Run analysis
# df_clean = analyze_custom_clusters(df_clean, n_clusters)
