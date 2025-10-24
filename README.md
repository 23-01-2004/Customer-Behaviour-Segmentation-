# Customer Behaviour Segmentation

## Project Overview

This project performs **customer segmentation and cluster analysis** to better understand purchasing behavior. By using unsupervised learning techniques like **K-Means**, we categorize customers into distinct groups based on their demographic and transaction data. The insights derived help in targeted marketing, campaign optimization, and revenue maximization.

---

## Objectives

- Segment customers into meaningful clusters based on spending, income, and demographic features.
- Analyze cluster characteristics and visualize trends using interactive and dynamic plots.
- Identify high-value and high-response customer groups for targeted campaigns.
- Provide actionable insights and strategic recommendations for business decisions.

---

## Features

- **K-Means Clustering:** Dynamically adjustable number of clusters.
- **Cluster Analysis:**
  - Income & spending distribution
  - Age & spending correlation
  - Response rates per cluster
- **Visualizations:**
  - Density, violin, scatter, box, radar, and 3D plots
  - Comparative cluster dashboards
- **Statistical Analysis:**
  - Cluster-level descriptive statistics
  - Key metrics like spending-income ratio, response rates
- **Strategic Insights:**
  - Profiles for each cluster (e.g., premium, budget-conscious, value-focused)
  - Recommendations for marketing and loyalty programs

---

## Dataset

- **Input Data:** `df_clean` – a preprocessed customer dataset
- **Key Features:**
  - Demographic: `Age`, `Income`, `Children_Count`
  - Spending: `Total_Spending`, `MntWines`, `MntMeatProducts`
  - Interaction: `NumWebPurchases`, `NumStorePurchases`, `Recency`
  - Target: `Response` (campaign response)

> **Note:** Dataset should be cleaned and numeric features scaled for clustering.

---

## Installation & Requirements

Install Python dependencies using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
