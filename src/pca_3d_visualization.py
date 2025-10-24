import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import streamlit as st

def pca_3d_visualization(X_pca_df, df_clean, explained_variance, vintage_colors):
    st.subheader(" 3D PCA Scatter Plot Visualization")

    # Map colors and labels
    colors = df_clean['Response'].map({0: vintage_colors['primary'], 1: vintage_colors['accent1']})
    labels = df_clean['Response'].map({0: 'Non-Responder', 1: 'Responder'})

    # --- Plot ---
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        X_pca_df['PC1'], X_pca_df['PC2'], X_pca_df['PC3'],
        c=colors, s=40, alpha=0.7, edgecolor=vintage_colors['dark'], linewidth=0.3
    )

    # Axis labels
    ax.set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}% Var)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}% Var)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_zlabel(f'PC3 ({explained_variance[2]*100:.1f}% Var)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_title('3D PCA Visualization - Customer Segmentation\nVintage Style', fontsize=16, fontweight='bold', pad=20)

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=vintage_colors['primary'], markersize=8, label='Non-Responder'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=vintage_colors['accent1'], markersize=8, label='Responder')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1), framealpha=0.9)

    ax.grid(True, alpha=0.3)
    ax.view_init(elev=20, azim=45)
    plt.tight_layout()
    st.pyplot(fig)

    # --- Compute statistics ---
    responder_pca = X_pca_df[df_clean['Response'] == 1]
    non_responder_pca = X_pca_df[df_clean['Response'] == 0]

    summary_stats = []

    for pc in ['PC1', 'PC2', 'PC3']:
        resp_mean = responder_pca[pc].mean()
        non_resp_mean = non_responder_pca[pc].mean()
        diff = resp_mean - non_resp_mean
        summary_stats.append({
            "Principal Component": pc,
            "Responders Mean": round(resp_mean, 3),
            "Non-Responders Mean": round(non_resp_mean, 3),
            "Difference": round(diff, 3),
            "Higher In": "Responders" if diff > 0 else "Non-Responders"
        })

    summary_df = pd.DataFrame(summary_stats)
    
    st.markdown("###  PCA Space Statistics (Responders vs Non-Responders)")
    st.dataframe(summary_df)

    # Response rate summary
    total_customers = len(df_clean)
    responders = len(responder_pca)
    non_responders = len(non_responder_pca)
    response_rate = responders / total_customers * 100

    st.markdown("###  Response Rate Summary")
    st.markdown(f"- Total Customers: {total_customers:,}")
    st.markdown(f"- Responders: {responders:,} ({response_rate:.1f}%)")
    st.markdown(f"- Non-Responders: {non_responders:,} ({100-response_rate:.1f}%)")
    st.markdown(f"- PCA components needed for 95% variance: {np.argmax(np.cumsum(explained_variance) >= 0.95) + 1}")
    st.markdown(f"- Variance explained by first 3 PCs: {np.sum(explained_variance[:3]):.3f}")

    st.success("3D PCA Visualization & Statistics Complete ")
