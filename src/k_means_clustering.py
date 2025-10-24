import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import streamlit as st

def kmeans_cluster_analysis(X_pca_df, df_clean, cumulative_variance, vintage_colors):
    st.subheader(" K-Means Clustering Analysis")

    # Select number of PCA components explaining 90% variance
    n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
    X_pca_for_clustering = X_pca_df.iloc[:, :n_components_90]

    st.markdown(f"Using first **{n_components_90} PCA components** (explains {cumulative_variance[n_components_90-1]:.3f} variance)")
    st.markdown(f"Data shape for clustering: {X_pca_for_clustering.shape}")

    # Define range of clusters to test
    k_range = range(2, 15)
    inertia = []
    silhouette_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_pca_for_clustering)
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_pca_for_clustering, kmeans.labels_))

    # Determine elbow point
    inertia_diff2 = np.diff(np.diff(inertia))
    elbow_point = np.argmin(inertia_diff2) + 3
    optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
    diminishing_returns = np.argmax(np.array([(inertia[i-1]-inertia[i])/inertia[i-1]*100 for i in range(1,len(inertia))]) < 10) + 3

    # Decide optimal K
    if elbow_point == optimal_k_silhouette:
        optimal_k = elbow_point
        reason = "Both methods agree"
    else:
        optimal_k = optimal_k_silhouette
        reason = "Silhouette score preferred"

    # Plot visualizations
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Elbow Curve
    ax1.plot(k_range, inertia, 'o-', color=vintage_colors['primary'], linewidth=3)
    ax1.axvline(elbow_point, color=vintage_colors['alert'], linestyle='--', label=f'Elbow K={elbow_point}')
    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow Method")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Silhouette Scores
    ax2.plot(k_range, silhouette_scores, 's-', color=vintage_colors['secondary'], linewidth=3)
    ax2.axvline(optimal_k_silhouette, color=vintage_colors['alert'], linestyle='--', label=f'Best K={optimal_k_silhouette}')
    ax2.set_xlabel("Number of Clusters (K)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Analysis")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Combined plot
    ax3.plot(k_range, inertia, 'o-', color=vintage_colors['primary'], label='Inertia', alpha=0.7)
    ax3_twin = ax3.twinx()
    ax3_twin.plot(k_range, silhouette_scores, 's-', color=vintage_colors['secondary'], label='Silhouette', alpha=0.7)
    ax3.set_xlabel("Number of Clusters (K)")
    ax3.set_ylabel("Inertia", color=vintage_colors['primary'])
    ax3_twin.set_ylabel("Silhouette Score", color=vintage_colors['secondary'])
    ax3.set_title("Combined Analysis")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')

    plt.tight_layout()
    st.pyplot(fig)

    # Perform final clustering
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df_clean['Cluster'] = final_kmeans.fit_predict(X_pca_for_clustering)

    # Cluster sizes table
    cluster_sizes = df_clean['Cluster'].value_counts().sort_index()
    cluster_sizes_df = pd.DataFrame({
        "Cluster": cluster_sizes.index,
        "Customers": cluster_sizes.values,
        "Percentage": (cluster_sizes.values / len(df_clean) * 100).round(1)
    })
    st.markdown("### 📊 Cluster Size Distribution")
    st.dataframe(cluster_sizes_df)

    # Cluster profiles table
    key_features = ['Income', 'Total_Spending', 'Age', 'Children_Count', 'Response']
    cluster_profiles = df_clean.groupby('Cluster')[key_features].mean().round(2)
    cluster_profiles['Response_Rate (%)'] = cluster_profiles['Response'] * 100
    cluster_profiles.drop(columns='Response', inplace=True)
    st.markdown("###  Cluster Profiles (Average Values)")
    st.dataframe(cluster_profiles)

    # Cluster characteristics
    st.markdown("###  Cluster Characteristics")
    for cluster_id in range(optimal_k):
        cluster_data = df_clean[df_clean['Cluster'] == cluster_id]
        characteristics = []
        if cluster_data['Income'].mean() > df_clean['Income'].mean(): characteristics.append("High Income")
        if cluster_data['Total_Spending'].mean() > df_clean['Total_Spending'].mean(): characteristics.append("High Spending")
        if cluster_data['Response'].mean() > df_clean['Response'].mean(): characteristics.append("High Response Rate")
        if not characteristics: characteristics.append("Average Profile")
        st.markdown(f"- **Cluster {cluster_id}**: {', '.join(characteristics)}")

    st.success(f"✅ Clustering Complete! Optimal K = {optimal_k} ({reason})")
