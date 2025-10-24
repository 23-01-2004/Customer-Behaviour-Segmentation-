import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_education_income_spending(df_clean, set_vintage_theme):
    vintage_colors = set_vintage_theme()

    # Compute total spending
    df_clean['Total_Spending'] = (
        df_clean['MntWines'] + df_clean['MntFruits'] + 
        df_clean['MntMeatProducts'] + df_clean['MntFishProducts'] + 
        df_clean['MntSweetProducts'] + df_clean['MntGoldProds']
    )

    # --- Plot Section ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    #  Income Distribution by Education
    sns.boxplot(
        data=df_clean, x='Education', y='Income',
        palette=[vintage_colors['primary'], vintage_colors['secondary'],
                 vintage_colors['accent1'], vintage_colors['accent2'],
                 vintage_colors['accent3']],
        ax=ax1
    )
    ax1.set_title('Income Distribution by Education Level\nVintage Style',
                  fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Education Level', fontsize=12)
    ax1.set_ylabel('Annual Income ($)', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)

    # Income mean annotations
    education_income_means = df_clean.groupby('Education')['Income'].mean()
    income_stats_text = "Income by Education:\n"
    for edu, mean_income in education_income_means.items():
        income_stats_text += f"{edu}: ${mean_income:,.0f}\n"

    ax1.text(0.98, 0.02, income_stats_text, transform=ax1.transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor=vintage_colors['light'],
                       edgecolor=vintage_colors['neutral'], alpha=0.8),
             verticalalignment='bottom', horizontalalignment='right', fontsize=9)

    #  Spending Distribution by Education
    sns.boxplot(
        data=df_clean, x='Education', y='Total_Spending',
        palette=[vintage_colors['primary'], vintage_colors['secondary'],
                 vintage_colors['accent1'], vintage_colors['accent2'],
                 vintage_colors['accent3']],
        ax=ax2
    )
    ax2.set_title('Total Spending Distribution by Education Level\nVintage Style',
                  fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('Education Level', fontsize=12)
    ax2.set_ylabel('Total Spending ($)', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)

    # Spending mean annotations
    education_spending_means = df_clean.groupby('Education')['Total_Spending'].mean()
    spending_stats_text = "Avg Spending by Education:\n"
    for edu, mean_spending in education_spending_means.items():
        spending_stats_text += f"{edu}: ${mean_spending:,.0f}\n"

    ax2.text(0.98, 0.02, spending_stats_text, transform=ax2.transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor=vintage_colors['light'],
                       edgecolor=vintage_colors['neutral'], alpha=0.8),
             verticalalignment='bottom', horizontalalignment='right', fontsize=9)

    plt.tight_layout()
    st.pyplot(fig)

    # --- Statistics Section ---
    st.markdown("###  Detailed Income and Spending Analysis by Education")

    education_income_stats = df_clean.groupby('Education')['Income'].agg(['mean', 'median', 'std', 'count']).sort_values('mean', ascending=False)
    education_spending_stats = df_clean.groupby('Education')['Total_Spending'].agg(['mean', 'median', 'std', 'count']).sort_values('mean', ascending=False)

    df_clean['Spending_Income_Ratio'] = (df_clean['Total_Spending'] / df_clean['Income']) * 100
    ratio_by_education = df_clean.groupby('Education')['Spending_Income_Ratio'].mean().sort_values(ascending=False)

    st.write("#### Income Statistics by Education")
    st.dataframe(education_income_stats.style.format({'mean': '${:,.0f}', 'median': '${:,.0f}', 'std': '${:,.0f}'}))

    st.write("#### Spending Statistics by Education")
    st.dataframe(education_spending_stats.style.format({'mean': '${:,.0f}', 'median': '${:,.0f}', 'std': '${:,.0f}'}))

    st.write("#### Spending-to-Income Ratio by Education")
    st.bar_chart(ratio_by_education)

    # --- Insights ---
    st.markdown("###  Key Insights")
    highest_income_edu = education_income_stats.index[0]
    lowest_income_edu = education_income_stats.index[-1]
    income_diff = education_income_stats.loc[highest_income_edu, 'mean'] - education_income_stats.loc[lowest_income_edu, 'mean']

    highest_spending_edu = education_spending_stats.index[0]
    lowest_spending_edu = education_spending_stats.index[-1]
    spending_diff = education_spending_stats.loc[highest_spending_edu, 'mean'] - education_spending_stats.loc[lowest_spending_edu, 'mean']

    st.info(f"""
    • **Highest Income Group:** {highest_income_edu} ({education_income_stats.loc[highest_income_edu, 'mean']:,.0f})  
    • **Lowest Income Group:** {lowest_income_edu} ({education_income_stats.loc[lowest_income_edu, 'mean']:,.0f})  
    • **Income Difference:** ${income_diff:,.0f}  
    • **Highest Spending Group:** {highest_spending_edu} ({education_spending_stats.loc[highest_spending_edu, 'mean']:,.0f})  
    • **Top Spenders by Ratio:** {ratio_by_education.index[0]} ({ratio_by_education.iloc[0]:.1f}% of income)
    """)

    st.success(" Education–Income–Spending analysis complete.")
