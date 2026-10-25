"""
data_validation.py
Schema and data quality validation for European Bank Churn dataset.

Preserves raw data; asserts schema integrity, binary target, absence of duplicate customer IDs,
missing values, and feature value ranges. Excludes CustomerId, Surname, and Year from modeling features.
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

REQUIRED_COLUMNS = [
    "Year",
    "CustomerId",
    "Surname",
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Exited"
]

EXCLUDE_FEATURES = ["CustomerId", "Surname", "Year"]
TARGET_COLUMN = "Exited"

FEATURE_COLUMNS = [col for col in REQUIRED_COLUMNS if col not in EXCLUDE_FEATURES and col != TARGET_COLUMN]
NUMERIC_FEATURES = ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"]
CATEGORICAL_FEATURES = ["Geography", "Gender", "NumOfProducts", "HasCrCard", "IsActiveMember"]

EXPECTED_GEOGRAPHIES = {"France", "Germany", "Spain"}
EXPECTED_GENDERS = {"Female", "Male"}


def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates the dataset against blueprint specifications.
    Fails loudly (raises ValueError) if critical schema, ID uniqueness,
    or target requirements are violated.
    
    Returns an audit summary dictionary.
    """
    audit = {}
    
    # 1. Required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Schema validation failed: missing required columns: {missing_cols}")
    audit["columns_present"] = True
    audit["column_count"] = len(df.columns)
    audit["columns"] = list(df.columns)
    
    # 2. Row count
    row_count = len(df)
    audit["row_count"] = row_count
    if row_count == 0:
        raise ValueError("Data validation failed: dataset is empty (0 rows).")
    
    # 3. CustomerId uniqueness
    dup_customers = int(df["CustomerId"].duplicated().sum())
    audit["duplicate_customer_ids"] = dup_customers
    if dup_customers > 0:
        raise ValueError(f"Data validation failed: found {dup_customers} duplicate CustomerId values.")
        
    # 4. Target variable Exited
    target_vals = set(df[TARGET_COLUMN].dropna().unique())
    if not target_vals.issubset({0, 1}):
        raise ValueError(f"Data validation failed: target '{TARGET_COLUMN}' contains non-binary values: {target_vals}")
    audit["target_unique_values"] = sorted(list(target_vals))
    exited_counts = df[TARGET_COLUMN].value_counts().to_dict()
    audit["exited_counts"] = {int(k): int(v) for k, v in exited_counts.items()}
    audit["overall_churn_rate"] = float(df[TARGET_COLUMN].mean())
    audit["total_exited"] = int(df[TARGET_COLUMN].sum())
    audit["total_retained"] = int((df[TARGET_COLUMN] == 0).sum())
    
    # 5. Missing values
    missing_per_col = df[REQUIRED_COLUMNS].isnull().sum().to_dict()
    audit["missing_per_column"] = {k: int(v) for k, v in missing_per_col.items()}
    audit["total_missing_cells"] = int(sum(missing_per_col.values()))
    
    # 6. Year check (constant 2025 as noted in blueprint)
    unique_years = [int(y) for y in df["Year"].unique()]
    audit["unique_years"] = unique_years
    
    # 7. Value ranges & categories
    geo_set = set(df["Geography"].unique())
    audit["geographies"] = sorted(list(geo_set))
    if not geo_set.issubset(EXPECTED_GEOGRAPHIES):
        raise ValueError(f"Unexpected Geography values: {geo_set - EXPECTED_GEOGRAPHIES}")
        
    gender_set = set(df["Gender"].unique())
    audit["genders"] = sorted(list(gender_set))
    if not gender_set.issubset(EXPECTED_GENDERS):
        raise ValueError(f"Unexpected Gender values: {gender_set - EXPECTED_GENDERS}")
        
    # Numeric ranges
    audit["credit_score_range"] = (int(df["CreditScore"].min()), int(df["CreditScore"].max()))
    audit["age_range"] = (int(df["Age"].min()), int(df["Age"].max()))
    audit["tenure_range"] = (int(df["Tenure"].min()), int(df["Tenure"].max()))
    audit["balance_range"] = (float(df["Balance"].min()), float(df["Balance"].max()))
    audit["products_range"] = (int(df["NumOfProducts"].min()), int(df["NumOfProducts"].max()))
    
    if df["Age"].min() < 18 or df["Age"].max() > 110:
        raise ValueError(f"Age values outside reasonable range [18, 110]: min={df['Age'].min()}, max={df['Age'].max()}")
        
    if df["NumOfProducts"].min() < 1 or df["NumOfProducts"].max() > 10:
        raise ValueError(f"NumOfProducts out of expected bounds: min={df['NumOfProducts'].min()}, max={df['NumOfProducts'].max()}")

    return audit


def load_and_validate_data(csv_path: str = "data/raw/European_Bank.csv") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads raw CSV without in-place mutation, runs full validation audit,
    and returns a clean copy of the DataFrame alongside the audit dictionary.
    """
    df = pd.read_csv(csv_path)
    # Never mutate the raw file or the input
    audit = validate_dataset(df)
    return df.copy(), audit


def get_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extracts candidate feature matrix X and target series y.
    Strictly excludes CustomerId, Surname, Year, and Exited from X.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Cannot extract target: '{TARGET_COLUMN}' missing from dataframe.")
    y = df[TARGET_COLUMN].copy()
    
    # Strictly select features
    X = df[FEATURE_COLUMNS].copy()
    
    # Defensive assertion: Exited must never be in X
    assert TARGET_COLUMN not in X.columns, "Leakage detected: Exited is in feature matrix X!"
    for col in EXCLUDE_FEATURES:
        assert col not in X.columns, f"Feature policy violation: {col} present in feature matrix X!"
        
    return X, y


if __name__ == "__main__":
    df, audit = load_and_validate_data()
    print("Data validation successful!")
    print(f"Rows: {audit['row_count']}, Columns: {audit['column_count']}")
    print(f"Exited: {audit['total_exited']} ({audit['overall_churn_rate']:.2%})")
    print(f"Missing cells: {audit['total_missing_cells']}")
    print(f"Duplicate CustomerIds: {audit['duplicate_customer_ids']}")
    X, y = get_features_and_target(df)
    print("Features matrix X shape:", X.shape)
    print("Feature columns:", list(X.columns))
