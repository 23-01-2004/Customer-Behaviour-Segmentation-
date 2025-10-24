import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_education_distribution(df_clean, set_vintage_theme):
    vintage_colors = set_vintage_theme()

    education_counts = df_clean['Education'].value_counts()
    education_percentages = (education_counts / len(df_clean)) * 100

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = [
        vintage_colors['primary'], vintage_colors['secondary'], 
        vintage_colors['accent1'], vintage_colors['accent2'], 
        vintage_colors['accent3']
    ]

    wedges, texts, autotexts = ax.pie(
        education_counts.values,
        labels=education_counts.index,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 11, 'color': vintage_colors['dark']},
        wedgeprops={'edgecolor': vintage_colors['dark'], 'linewidth': 1.5, 'alpha': 0.9}
    )

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)

    for text in texts:
        text.set_fontsize(12)
        text.set_fontweight('bold')
        text.set_color(vintage_colors['dark'])

    ax.set_title(
        'Distribution of Customer Education Levels\nVintage Style',
        fontsize=16, fontweight='bold', pad=20, color=vintage_colors['dark']
    )

    legend_labels = [
        f'{edu} ({count:,} customers, {percent:.1f}%)'
        for edu, count, percent in zip(
            education_counts.index,
            education_counts.values,
            education_percentages
        )
    ]

    ax.legend(
        wedges, legend_labels,
        title="Education Levels",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        frameon=True, framealpha=0.9,
        edgecolor=vintage_colors['neutral'],
        facecolor=vintage_colors['light']
    )

    centre_circle = plt.Circle(
        (0, 0), 0.70, fc=vintage_colors['light'],
        edgecolor=vintage_colors['neutral'], linewidth=1
    )
    ax.add_artist(centre_circle)

    total_customers = len(df_clean)
    ax.text(0, 0, f'Total\nCustomers\n{total_customers:,}',
            ha='center', va='center', fontsize=14, fontweight='bold',
            color=vintage_colors['dark'])

    ax.axis('equal')
    plt.tight_layout()

    # Display in Streamlit
    st.pyplot(fig)

    # --- Stats Section ---
    st.markdown("### 📚 Education Distribution Details")
    st.write(f"**Total Customers:** {total_customers:,}")
    st.write(f"**Most Common Education:** {education_counts.index[0]} ({education_counts.iloc[0]:,} customers)")
    st.write(f"**Least Common Education:** {education_counts.index[-1]} ({education_counts.iloc[-1]:,} customers)")
    st.write(f"**Educational Diversity:** {len(education_counts)} unique levels")

    # Percentage table
    st.dataframe(
        education_percentages.reset_index().rename(columns={'index': 'Education', 'Education': 'Percentage (%)'}),
        use_container_width=True
    )

    # --- Education vs Income ---
    st.markdown("###  Education vs Average Income")
    education_income = df_clean.groupby('Education')['Income'].mean().sort_values(ascending=False)
    st.bar_chart(education_income)

    st.success("🎓 Education distribution analysis complete.")
