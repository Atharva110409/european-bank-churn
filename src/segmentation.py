"""
segmentation.py
Reusable customer segmentation analysis functions for European Bank Churn data.

Provides standardized summaries for Geography, Age bands (<30, 30-45, 46-59, 60+),
NumOfProducts, and IsActiveMember.
Each summary returns:
- segment: category label
- n: customer count
- exited: churned customer count
- retained: non-churned customer count
- churn_rate: exited / n
- share_of_all_exits: exited / total_exits
"""

from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np

AGE_BANDS = ["<30", "30-45", "46-59", "60+"]
GEOGRAPHIES = ["France", "Germany", "Spain"]
PRODUCT_COUNTS = [1, 2, 3, 4]
ENGAGEMENT_LABELS = {0: "Inactive", 1: "Active"}


def assign_age_band(age_series: pd.Series) -> pd.Series:
    """
    Categorizes numeric age into exact blueprint age bands:
    - '<30'   : Age < 30
    - '30-45' : 30 <= Age <= 45
    - '46-59' : 46 <= Age <= 59
    - '60+'   : Age >= 60
    """
    conditions = [
        age_series < 30,
        (age_series >= 30) & (age_series <= 45),
        (age_series >= 46) & (age_series <= 59),
        age_series >= 60
    ]
    choices = AGE_BANDS
    return pd.Series(np.select(conditions, choices, default="Unknown"), index=age_series.index)


def _compute_segment_table(
    df: pd.DataFrame,
    group_col: str,
    categories: Optional[List[Any]] = None,
    category_labels: Optional[Dict[Any, str]] = None,
    total_exits: Optional[int] = None
) -> pd.DataFrame:
    """
    Internal helper to compute segment statistics safely, even on empty slices.
    """
    columns = ["segment", "n", "exited", "retained", "churn_rate", "share_of_all_exits"]
    
    if total_exits is None:
        total_exits = int(df["Exited"].sum()) if ("Exited" in df.columns and len(df) > 0) else 0

    if len(df) == 0:
        if categories is not None:
            records = []
            for cat in categories:
                label = category_labels.get(cat, str(cat)) if category_labels else str(cat)
                records.append({
                    "segment": label,
                    "n": 0,
                    "exited": 0,
                    "retained": 0,
                    "churn_rate": 0.0,
                    "share_of_all_exits": 0.0
                })
            return pd.DataFrame(records, columns=columns)
        return pd.DataFrame(columns=columns)

    grouped = df.groupby(group_col, observed=False).agg(
        n=("Exited", "count"),
        exited=("Exited", "sum"),
        retained=("Exited", lambda x: int((x == 0).sum())),
        churn_rate=("Exited", "mean")
    ).reset_index()

    if categories is not None:
        # Ensure all expected categories are present
        cat_df = pd.DataFrame({group_col: categories})
        merged = cat_df.merge(grouped, on=group_col, how="left")
        merged["n"] = merged["n"].fillna(0).astype(int)
        merged["exited"] = merged["exited"].fillna(0).astype(int)
        merged["retained"] = merged["retained"].fillna(0).astype(int)
        merged["churn_rate"] = np.where(merged["n"] > 0, merged["exited"] / merged["n"], 0.0)
        res = merged
    else:
        res = grouped

    if category_labels is not None:
        res["segment"] = res[group_col].map(lambda x: category_labels.get(x, str(x)))
    else:
        res["segment"] = res[group_col].astype(str)

    res["share_of_all_exits"] = np.where(total_exits > 0, res["exited"] / total_exits, 0.0)
    
    return res[columns]


def summarize_geography(df: pd.DataFrame, total_exits: Optional[int] = None) -> pd.DataFrame:
    """
    Summarizes churn by Geography (France, Germany, Spain).
    """
    return _compute_segment_table(
        df=df,
        group_col="Geography",
        categories=GEOGRAPHIES,
        total_exits=total_exits
    )


def summarize_age_bands(df: pd.DataFrame, total_exits: Optional[int] = None) -> pd.DataFrame:
    """
    Summarizes churn by Age Band (<30, 30-45, 46-59, 60+).
    """
    if len(df) == 0:
        return _compute_segment_table(
            df=df,
            group_col="AgeBand",
            categories=AGE_BANDS,
            total_exits=total_exits
        )
        
    df_copy = df.copy()
    if "AgeBand" not in df_copy.columns:
        df_copy["AgeBand"] = assign_age_band(df_copy["Age"])
        
    return _compute_segment_table(
        df=df_copy,
        group_col="AgeBand",
        categories=AGE_BANDS,
        total_exits=total_exits
    )


def summarize_num_of_products(df: pd.DataFrame, total_exits: Optional[int] = None) -> pd.DataFrame:
    """
    Summarizes churn by NumOfProducts (1, 2, 3, 4).
    """
    return _compute_segment_table(
        df=df,
        group_col="NumOfProducts",
        categories=PRODUCT_COUNTS,
        total_exits=total_exits
    )


def summarize_is_active_member(df: pd.DataFrame, total_exits: Optional[int] = None) -> pd.DataFrame:
    """
    Summarizes churn by Engagement / IsActiveMember (Inactive=0, Active=1).
    """
    return _compute_segment_table(
        df=df,
        group_col="IsActiveMember",
        categories=[0, 1],
        category_labels=ENGAGEMENT_LABELS,
        total_exits=total_exits
    )


def get_all_segment_summaries(df: pd.DataFrame, total_exits: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    """
    Returns a dictionary of all four standard segment summary tables.
    """
    return {
        "Geography": summarize_geography(df, total_exits=total_exits),
        "AgeBand": summarize_age_bands(df, total_exits=total_exits),
        "NumOfProducts": summarize_num_of_products(df, total_exits=total_exits),
        "IsActiveMember": summarize_is_active_member(df, total_exits=total_exits)
    }


if __name__ == "__main__":
    from src.data_validation import load_and_validate_data
    df, _ = load_and_validate_data()
    total_exits = int(df["Exited"].sum())
    summaries = get_all_segment_summaries(df, total_exits=total_exits)
    for name, table in summaries.items():
        print(f"\n--- {name} ---")
        print(table.to_string(index=False))
