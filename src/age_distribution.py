import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from src.vintage_theme import set_vintage_theme

def analyze_age_distribution(df_clean):
    current_year = datetime.now().year
    df_clean['Age'] = current_year - df_clean['Year_Birth']
    vintage_colors = set_vintage_theme()

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.histplot(data=df_clean, x='Age', bins=30, kde=True,
                 color=vintage_colors['primary'], alpha=0.7,
                 edgecolor=vintage_colors['dark'], linewidth=0.5,
                 stat='density', ax=ax)

    ax.lines[0].set_color(vintage_colors['accent1'])
    ax.lines[0].set_linewidth(2.5)

    mean_age = df_clean['Age'].mean()
    median_age = df_clean['Age'].median()
    min_age = df_clean['Age'].min()
    max_age = df_clean['Age'].max()
    std_age = df_clean['Age'].std()
    Q1 = df_clean['Age'].quantile(0.25)
    Q3 = df_clean['Age'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_clean[(df_clean['Age'] < lower_bound) | (df_clean['Age'] > upper_bound)]

    ax.axvline(mean_age, color=vintage_colors['alert'], linestyle='--', linewidth=2,
               label=f'Mean: {mean_age:.1f}')
    ax.axvline(median_age, color=vintage_colors['accent2'], linestyle='--', linewidth=2,
               label=f'Median: {median_age:.1f}')
    ax.legend()

    ax.set_title('Distribution of Customer Ages', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Age (Years)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)

    sns.despine(left=True)
    plt.tight_layout()

    # ✅ Display plot in Streamlit
    st.pyplot(fig)

    # ✅ Display summary statistics below the plot
    st.markdown("###  Age Statistics Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Age", f"{mean_age:.1f} years")
        st.metric("Median Age", f"{median_age:.1f} years")
    with col2:
        st.metric("Min Age", f"{min_age:.0f} years")
        st.metric("Max Age", f"{max_age:.0f} years")
    with col3:
        st.metric("Std Deviation", f"{std_age:.1f} years")
        st.metric("IQR", f"{IQR:.1f} years")

    st.markdown("###  Percentiles")
    percentiles = [25, 50, 75, 90, 95]
    perc_data = {f"{p}th": df_clean['Age'].quantile(p/100) for p in percentiles}
    st.dataframe(
        { "Percentile": perc_data.keys(), "Age (years)": [f"{v:.1f}" for v in perc_data.values()] },
        use_container_width=True
    )

    st.markdown(f"###  Outlier Analysis")
    st.info(f"Detected **{len(outliers)}** potential outliers (IQR method). "
            f"Outlier age range: {outliers['Age'].min():.1f} - {outliers['Age'].max():.1f} years.")
