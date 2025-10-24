import pandas as pd
import numpy as np

def load_data(file_path: str, delimiter: str = '\t'):
    df = pd.read_csv(file_path, delimiter=delimiter)
    return df

def dataset_overview(df: pd.DataFrame):
    overview = {
        "shape": df.shape,
        "total_records": df.shape[0],
        "total_features": df.shape[1],
    }
    return overview

def categorical_summary(df: pd.DataFrame, categorical_cols):
    summary = {}
    for col in categorical_cols:
        summary[col] = {
            "value_counts": df[col].value_counts(),
            "unique_values": df[col].nunique()
        }
    return summary

def target_summary(df: pd.DataFrame, target_col: str):
    counts = df[target_col].value_counts()
    response_rate = (df[target_col].mean() * 100) if df[target_col].dtype != 'O' else None
    return counts, response_rate

def null_value_analysis(df: pd.DataFrame):
    """Analyze and remove null values."""
    null_summary = df.isnull().sum()
    null_percentage = (df.isnull().sum() / len(df)) * 100

    null_info = pd.DataFrame({
        'Null Count': null_summary,
        'Null Percentage': null_percentage
    })

    null_info_filtered = null_info[null_info['Null Count'] > 0]

    df_clean = df.dropna()

    verification = {
        "after_shape": df_clean.shape,
        "records_removed": df.shape[0] - df_clean.shape[0],
        "no_null_remaining": df_clean.isnull().sum().sum() == 0,
        "data_retained_pct": (df_clean.shape[0] / df.shape[0]) * 100
    }

    return null_info_filtered, df_clean, verification
