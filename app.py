import streamlit as st
import pandas as pd
from src.data_description import load_data, dataset_overview, categorical_summary, target_summary, null_value_analysis
from src.outlier_analysis import outlier_analysis, plot_outliers
from src.vintage_theme import set_vintage_theme
import sys, os
sys.path.append(os.path.abspath("src"))
from src.age_distribution import analyze_age_distribution
from src.income_spending import analyze_income_spending
from src.education_distribution import analyze_education_distribution
from src.education_income_spending import analyze_education_income_spending
from src.children_analysis import analyze_children_distribution
from src.correlation_analysis import analyze_correlations
from pca_analysis import perform_pca
from src.pca_3d_visualization import pca_3d_visualization 
from src.k_means_clustering import kmeans_cluster_analysis
import numpy as np
from src.customer_clusters import analyze_custom_clusters
from src.income_spending_clusters import income_spending_cluster_analysis
from src.spending_age_analysis import analyze_cluster_characteristics_dynamic_streamlit

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Customer Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================
st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --background-color: #f8f9fa;
        --card-background: #ffffff;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main title styling */
    .main-title {
        text-align: center;
        padding: 2rem 0 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #6c757d;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: white;
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.15);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 2rem 0 1rem 0;
        font-size: 1.3rem;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Upload section */
    .upload-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 3rem;
        border-radius: 16px;
        text-align: center;
        margin: 2rem 0;
        border: 2px dashed #667eea;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1a1a1a;
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #2d2d2d;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        border: 1px solid #3d3d3d;
        color: #e0e0e0;
    }
    
    .stTabs [aria-selected="true"] {
        background: #000000;
        color: white;
        border: 1px solid #667eea;
        box-shadow: 0 0 10px rgba(102, 126, 234, 0.3);
    }
    
    /* Data frame styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Success/Info boxes */
    .stSuccess, .stInfo, .stWarning {
        border-radius: 8px;
        border-left: 4px solid;
    }
    
    /* Card container */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }
    
    /* Divider */
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER SECTION
# ============================================================================
st.markdown('<h1 class="main-title"> Customer Analytics Platform</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced Customer Behavior Segmentation & Analysis Dashboard</p>', unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.image("https://via.placeholder.com/200x80/667eea/ffffff?text=Analytics+Pro", use_container_width=True)
    st.markdown("---")
    st.markdown("### 🎯 About This Platform")
    st.info("""
    This comprehensive analytics platform helps you understand customer behavior through:
    - **Data Quality Analysis**
    - **Statistical Insights**
    - **Advanced Clustering**
    - **Predictive Modeling**
    """)
    st.markdown("---")
    st.markdown("###  Quick Guide")
    st.markdown("""
    1. **Upload** your customer data (CSV/TXT)
    2. **Explore** automated insights
    3. **Analyze** patterns and trends
    4. **Segment** customers intelligently
    """)
    st.markdown("---")
    st.markdown("##### Made with ❤️ using Streamlit")

# ============================================================================
# FILE UPLOAD SECTION
# ============================================================================
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown("### 📁 Upload Customer Data")
st.markdown("Support for CSV and TXT files with tab-separated values")
uploaded_file = st.file_uploader("", type=["csv", "txt"], label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# MAIN DASHBOARD
# ============================================================================
if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file, delimiter='\t')
    
    # Display success message
    st.success(f" Successfully loaded {len(df):,} records with {len(df.columns)} features!")
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Create tabs with icons
    tabs = st.tabs([
        " Overview",
        " Data Quality",
        " Outliers",
        " Age Analysis",
        " Income & Spending",
        " Education",
        " Education-Income",
        " Children Analysis",
        " Correlations",
        " PCA Analysis",
        " 3D PCA",
        " K-Means",
        " Custom Clusters",
        " Income Clusters",
        " Spending-Age"
    ])

    # ========================================================================
    # TAB 0: OVERVIEW
    # ========================================================================
    with tabs[0]:
        st.markdown('<div class="section-header">📊 Dataset Overview</div>', unsafe_allow_html=True)
        
        overview = dataset_overview(df)
        
        # Metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Records</div>
                <div class="metric-value">{overview['total_records']:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                <div class="metric-label">Total Features</div>
                <div class="metric-value">{overview['total_features']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            numeric_cols = len(df.select_dtypes(include=['int64', 'float64']).columns)
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                <div class="metric-label">Numeric Features</div>
                <div class="metric-value">{numeric_cols}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            categorical_cols = len(df.select_dtypes(include=['object']).columns)
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
                <div class="metric-label">Categorical Features</div>
                <div class="metric-value">{categorical_cols}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Data preview
        st.markdown('<div class="section-header">🔎 Data Preview</div>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True, height=400)
        
        # Categorical summary
        st.markdown('<div class="section-header">📑 Categorical Features Summary</div>', unsafe_allow_html=True)
        categorical_cols = ['Education', 'Marital_Status']
        summary = categorical_summary(df, categorical_cols)
        
        cols = st.columns(len(categorical_cols))
        for idx, (col, info) in enumerate(summary.items()):
            with cols[idx]:
                st.markdown(f"**{col}**")
                st.dataframe(info["value_counts"], use_container_width=True)
                st.caption(f"Unique values: {info['unique_values']}")
        
        # Target variable
        st.markdown('<div class="section-header">🎯 Target Variable Distribution</div>', unsafe_allow_html=True)
        counts, response_rate = target_summary(df, "Response")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.bar_chart(counts)
        with col2:
            if response_rate:
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">
                    <div class="metric-label">Response Rate</div>
                    <div class="metric-value">{response_rate:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

    # ========================================================================
    # TAB 1: DATA QUALITY
    # ========================================================================
    with tabs[1]:
        st.markdown('<div class="section-header">🧹 Data Quality Analysis</div>', unsafe_allow_html=True)
        null_info, df_clean, verify = null_value_analysis(df)
        
        if not null_info.empty:
            st.warning(" Null values detected in the dataset")
            st.dataframe(null_info, use_container_width=True)
        else:
            st.success(" No null values found! Dataset is clean.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Records After Cleaning", f"{verify['after_shape'][0]:,}")
        with col2:
            st.metric("Data Retained", f"{verify['data_retained_pct']:.2f}%")

    # ========================================================================
    # TAB 2: OUTLIER ANALYSIS
    # ========================================================================
    with tabs[2]:
        st.markdown('<div class="section-header">🔍 Outlier Detection & Removal</div>', unsafe_allow_html=True)
        df_no_outliers, outlier_summary, verify_outliers, numerical_vars, colors = outlier_analysis(df_clean)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Before Outlier Removal**")
            fig_before = plot_outliers(df_clean, numerical_vars, colors, "Outlier Detection (Before)", "primary")
            st.pyplot(fig_before)
        
        with col2:
            st.markdown("** After Outlier Removal**")
            fig_after = plot_outliers(df_no_outliers, numerical_vars, colors, "Data Distribution (After)", "secondary")
            st.pyplot(fig_after)
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Records Removed", f"{verify_outliers['records_removed']:,}")
        with col2:
            st.metric("Data Retained", f"{verify_outliers['data_retained_pct']:.1f}%")
        with col3:
            st.metric("Final Records", f"{verify_outliers['records_after']:,}")
        
        st.success(" Outlier removal process completed successfully!")

    # ========================================================================
    # TAB 3-14: ANALYSIS TABS
    # ========================================================================
    with tabs[3]:
        st.markdown('<div class="section-header">👥 Age Distribution Analysis</div>', unsafe_allow_html=True)
        analyze_age_distribution(df_clean)
        st.success(" Age distribution analysis complete")

    with tabs[4]:
        st.markdown('<div class="section-header"> Income & Total Spending Analysis</div>', unsafe_allow_html=True)
        analyze_income_spending(df_clean, set_vintage_theme)
        st.success(" Income & spending analysis complete")

    with tabs[5]:
        st.markdown('<div class="section-header"> Education Distribution Analysis</div>', unsafe_allow_html=True)
        analyze_education_distribution(df_clean, set_vintage_theme)
        st.success(" Education analysis complete")

    with tabs[6]:
        st.markdown('<div class="section-header"> Education-Income-Spending Analysis</div>', unsafe_allow_html=True)
        analyze_education_income_spending(df_clean, set_vintage_theme)

    with tabs[7]:
        st.markdown('<div class="section-header"> Children Distribution Analysis</div>', unsafe_allow_html=True)
        vintage_colors = set_vintage_theme()
        plt, children_stats = analyze_children_distribution(df_clean, vintage_colors)
        st.pyplot(plt)
        st.text(children_stats)
        st.success(" Children analysis complete")

    with tabs[8]:
        st.markdown('<div class="section-header">🔗 Correlation Analysis</div>', unsafe_allow_html=True)
        vintage_colors = set_vintage_theme()
        plt, df_pos, df_neg, df_income, df_clusters, insights = analyze_correlations(df_clean, vintage_colors)
        
        st.pyplot(plt)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("** Strong Positive Correlations (r > 0.5)**")
            st.dataframe(df_pos, use_container_width=True)
        
        with col2:
            st.markdown("** Strong Negative Correlations (r < -0.3)**")
            st.dataframe(df_neg, use_container_width=True)
        
        st.markdown("** Income Relationships**")
        st.dataframe(df_income.head(10), use_container_width=True)
        
        st.markdown("**🎯 Highly Correlated Feature Clusters (|r| > 0.6)**")
        if not df_clusters.empty:
            st.dataframe(df_clusters, use_container_width=True)
        else:
            st.info("No highly correlated feature pairs found")
        
        st.markdown("** Key Insights**")
        for point in insights:
            st.markdown(f"- {point}")
        
        st.success(" Correlation analysis complete")

    with tabs[9]:
        st.markdown('<div class="section-header">📈 Principal Component Analysis</div>', unsafe_allow_html=True)
        vintage_colors = set_vintage_theme()
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        
        numerical_features = df_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
        x = df_clean[numerical_features]
        x_scaled = StandardScaler().fit_transform(x)
        
        perform_pca(x_scaled, x, numerical_features, vintage_colors)

    with tabs[10]:
        st.markdown('<div class="section-header">🌐 3D PCA Visualization</div>', unsafe_allow_html=True)
        vintage_colors = set_vintage_theme()
        
        numerical_features = df_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
        x = df_clean[numerical_features]
        x_scaled = StandardScaler().fit_transform(x)
        
        pca = PCA()
        X_pca = pca.fit_transform(x_scaled)
        X_pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])], index=x.index)
        explained_variance = pca.explained_variance_ratio_
        
        pca_3d_visualization(X_pca_df, df_clean, explained_variance, vintage_colors)

    with tabs[11]:
        st.markdown('<div class="section-header">🎯 K-Means Clustering Analysis</div>', unsafe_allow_html=True)
        vintage_colors = set_vintage_theme()
        
        numerical_features = df_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
        x = df_clean[numerical_features]
        x_scaled = StandardScaler().fit_transform(x)
        
        pca = PCA()
        X_pca = pca.fit_transform(x_scaled)
        X_pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])], index=x.index)
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        kmeans_cluster_analysis(X_pca_df, df_clean, cumulative_variance, vintage_colors)

    with tabs[12]:
        st.markdown('<div class="section-header">🔧 Custom Cluster Analysis</div>', unsafe_allow_html=True)
        st.markdown("**Adjust the number of clusters to create custom segments**")
        n_clusters = st.slider("Select number of clusters:", 2, 4, 3)
        df_clean = analyze_custom_clusters(df_clean, n_clusters)

    with tabs[13]:
        st.markdown('<div class="section-header">💳 Income & Spending Cluster Analysis</div>', unsafe_allow_html=True)
        df_clean = income_spending_cluster_analysis(df_clean)

    with tabs[14]:
        st.markdown('<div class="section-header">🎪 Spending & Age Cluster Characteristics</div>', unsafe_allow_html=True)
        analyze_cluster_characteristics_dynamic_streamlit(df_clean)

else:
    # Empty state with beautiful design
    st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem;">
            <div style="font-size: 5rem; margin-bottom: 1rem;">📊</div>
            <h2 style="color: #667eea; margin-bottom: 1rem;">Ready to Analyze Your Customer Data?</h2>
            <p style="font-size: 1.2rem; color: #6c757d; max-width: 600px; margin: 0 auto;">
                Upload your CSV or TXT file to unlock powerful insights about your customers.
                Our advanced analytics engine will automatically process and visualize your data.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Feature highlights
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        **🎯 Smart Analysis**
        
        Automated data quality checks and outlier detection
        """)
    
    with col2:
        st.markdown("""
        **📊 Rich Visualizations**
        
        Interactive charts and 3D plots for deep insights
        """)
    
    with col3:
        st.markdown("""
        **🤖 ML Clustering**
        
        Advanced segmentation using PCA and K-Means
        """)
    
    with col4:
        st.markdown("""
        **⚡ Fast Processing**
        
        Instant results with optimized algorithms
        """)
