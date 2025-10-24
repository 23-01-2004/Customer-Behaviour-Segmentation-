from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def perform_pca(x_scaled, x, numerical_features, vintage_colors):
    # --- PCA ---
    st.subheader(" Principal Component Analysis (PCA)")
    pca = PCA()
    X_pca = pca.fit_transform(x_scaled)
    X_pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])], index=x.index)

    st.success(f"PCA Transformation Complete! Transformed shape: {X_pca_df.shape}")

    # --- Explained Variance ---
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    df_variance = pd.DataFrame({
        "Component": [f"PC{i+1}" for i in range(len(explained_variance))],
        "Explained Variance": explained_variance.round(4),
        "Cumulative Variance": cumulative_variance.round(4)
    })
    st.markdown("###  Explained Variance Analysis")
    st.dataframe(df_variance.head(10))  # show top 10 components

    # --- Visualization ---
    plt.figure(figsize=(12, 6))

    # Scree Plot
    plt.subplot(1, 2, 1)
    components = range(1, len(explained_variance) + 1)
    plt.bar(components[:10], explained_variance[:10], color=vintage_colors['primary'], alpha=0.7, label='Individual')
    plt.plot(components[:10], cumulative_variance[:10], color=vintage_colors['accent1'], marker='o', linewidth=2, label='Cumulative')
    plt.axhline(y=0.95, color=vintage_colors['alert'], linestyle='--', label='95% Variance')
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Scree Plot (Vintage Style)', fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)

    # Cumulative Plot
    plt.subplot(1, 2, 2)
    plt.plot(components, cumulative_variance, color=vintage_colors['secondary'], marker='o', linewidth=2)
    plt.axhline(y=0.95, color=vintage_colors['alert'], linestyle='--', label='95% Variance')
    plt.axvline(x=np.argmax(cumulative_variance >= 0.95) + 1, color=vintage_colors['accent2'], linestyle='--',
                label=f'PC needed: {np.argmax(cumulative_variance >= 0.95) + 1}')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Variance (Vintage Style)', fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(plt)

    # --- Feature Loadings ---
    st.markdown("### 🔗 Feature Loadings (Top 5 per Principal Component)")
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loadings_df = pd.DataFrame(loadings, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])], index=numerical_features)

    # Prepare a clean display of top 5 features for first 3 PCs
    for i in range(3):
        pc = f'PC{i+1}'
        top_features = (
            loadings_df[pc]
            .sort_values(key=abs, ascending=False)
            .head(5)
            .reset_index()
            .rename(columns={'index': 'Feature', pc: 'Loading'})
        )
        top_features["Direction"] = np.where(top_features["Loading"] > 0, "Positive", "Negative")
        st.markdown(f"**{pc} — Top 5 Most Influential Features**")
        st.dataframe(top_features)

    st.success("PCA Analysis Complete ✅")
    return X_pca_df, df_variance, loadings_df
