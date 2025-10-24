import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_income_spending(df_clean, set_vintage_theme):
    vintage_colors = set_vintage_theme()

    df_clean['Total Spending'] = (
        df_clean['MntWines'] + df_clean['MntFruits'] +
        df_clean['MntMeatProducts'] + df_clean['MntFishProducts'] +
        df_clean['MntSweetProducts'] + df_clean['MntGoldProds']
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 6))

    # ===== INCOME DISTRIBUTION =====
    sns.histplot(
        data=df_clean, x='Income', bins=40, kde=True,
        color=vintage_colors['primary'], alpha=0.7,
        linewidth=0.5, edgecolor=vintage_colors['dark'],
        stat='density', ax=ax1
    )

    ax1.lines[0].set_color(vintage_colors['accent1'])
    ax1.lines[0].set_linewidth(2.5)

    mean_income = df_clean['Income'].mean()
    median_income = df_clean['Income'].median()
    std_income = df_clean['Income'].std()
    min_income = df_clean['Income'].min()
    max_income = df_clean['Income'].max()

    income_stats = (
        f"**Income Statistics**  \n"
        f"• Mean Income: ${mean_income:,.0f}  \n"
        f"• Median Income: ${median_income:,.0f}  \n"
        f"• Std. Deviation: ${std_income:,.0f}  \n"
        f"• Range: ${min_income:,.0f} - ${max_income:,.0f}"
    )

    ax1.text(0.98, 0.98, income_stats, transform=ax1.transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor=vintage_colors['light'],
                       edgecolor=vintage_colors['neutral'], alpha=0.8),
             verticalalignment='top', horizontalalignment='right', fontsize=9)
    ax1.set_title("Income Distribution", fontsize=14, fontweight='bold')

    # ===== TOTAL SPENDING DISTRIBUTION =====
    sns.histplot(
        data=df_clean, x='Total Spending', bins=40, kde=True,
        color=vintage_colors['secondary'], alpha=0.7,
        linewidth=0.5, edgecolor=vintage_colors['dark'],
        stat='density', ax=ax2
    )

    ax2.lines[0].set_color(vintage_colors['accent3'])
    ax2.lines[0].set_linewidth(2.5)

    mean_spending = df_clean['Total Spending'].mean()
    median_spending = df_clean['Total Spending'].median()
    std_spending = df_clean['Total Spending'].std()
    min_spending = df_clean['Total Spending'].min()
    max_spending = df_clean['Total Spending'].max()

    spending_stats = (
        f"**Spending Statistics**  \n"
        f"• Mean Spending: ${mean_spending:,.0f}  \n"
        f"• Median Spending: ${median_spending:,.0f}  \n"
        f"• Std. Deviation: ${std_spending:,.0f}  \n"
        f"• Range: ${min_spending:,.0f} - ${max_spending:,.0f}"
    )

    ax2.text(0.98, 0.98, spending_stats, transform=ax2.transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor=vintage_colors['light'],
                       edgecolor=vintage_colors['neutral'], alpha=0.8),
             verticalalignment='top', horizontalalignment='right', fontsize=9)
    ax2.set_title("Total Spending Distribution", fontsize=14, fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)

    # --- Summary in Streamlit layout ---
    st.markdown("###  Summary Statistics")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("####  Income Overview")
        st.metric("Mean Income", f"${mean_income:,.0f}")
        st.metric("Median Income", f"${median_income:,.0f}")
        st.metric("Standard Deviation", f"${std_income:,.0f}")
        st.metric("Range", f"${min_income:,.0f} - ${max_income:,.0f}")

    with col2:
        st.markdown("####  Spending Overview")
        st.metric("Mean Spending", f"${mean_spending:,.0f}")
        st.metric("Median Spending", f"${median_spending:,.0f}")
        st.metric("Standard Deviation", f"${std_spending:,.0f}")
        st.metric("Range", f"${min_spending:,.0f} - ${max_spending:,.0f}")

    st.success(" Income & Total Spending analysis complete.")
