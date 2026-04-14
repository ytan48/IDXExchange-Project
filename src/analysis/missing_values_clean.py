import pandas as pd
import numpy as np


def missing_summary(df, threshold=90):
    """
    Calculate missing count and missing percentage for each column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    threshold : float, default=90
        Threshold used to flag high-missing columns.

    Returns
    -------
    summary : pd.DataFrame
        DataFrame with column, missing_count, missing_pct.
    high_missing : pd.DataFrame
        Subset of summary where missing_pct > threshold.
    """
    missing_count = df.isna().sum()
    missing_pct = df.isna().mean() * 100

    summary = pd.DataFrame({
        "column": df.columns,
        "missing_count": missing_count.values,
        "missing_pct": missing_pct.values
    }).sort_values("missing_pct", ascending=False).reset_index(drop=True)

    high_missing = summary[summary["missing_pct"] > threshold].reset_index(drop=True)

    return summary, high_missing


def decide_drop_or_retain(summary_df, core_fields, threshold=90):
    """
    Decide whether to drop or retain columns based on missing percentage
    and whether the column is a core field.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output from missing_summary().
    core_fields : list
        List of core columns to retain even when missingness is high.
    threshold : float, default=90
        Missing percentage threshold for dropping.

    Returns
    -------
    decision_summary : pd.DataFrame
        Summary table with recommended action and reason.
    drop_cols : list
        Columns recommended to drop.
    retain_cols : list
        Columns recommended to retain.
    """
    decision_summary = summary_df.copy()

    decision_summary["is_core_field"] = decision_summary["column"].isin(core_fields)

    decision_summary["recommended_action"] = np.where(
        (decision_summary["missing_pct"] > threshold) & (~decision_summary["is_core_field"]),
        "drop",
        "retain"
    )

    decision_summary["reason"] = np.where(
        (decision_summary["missing_pct"] > threshold) & (~decision_summary["is_core_field"]),
        f"missing > {threshold}% and not a core field",
        np.where(
            (decision_summary["missing_pct"] > threshold) & (decision_summary["is_core_field"]),
            f"missing > {threshold}% but retained because it is a core field",
            "retain"
        )
    )

    drop_cols = decision_summary.loc[
        decision_summary["recommended_action"] == "drop", "column"
    ].tolist()

    retain_cols = decision_summary.loc[
        decision_summary["recommended_action"] == "retain", "column"
    ].tolist()

    return decision_summary, drop_cols, retain_cols


def clean_by_missing_rule(df, core_fields, threshold=90):
    """
    Full pipeline for missing-value-based column cleaning.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    core_fields : list
        Core columns to retain.
    threshold : float, default=90
        Missing percentage threshold.

    Returns
    -------
    result : dict
        Dictionary containing cleaned dataframe and all intermediate summaries.
    """
    summary_df, high_missing = missing_summary(df, threshold=threshold)

    decision_summary, drop_cols, retain_cols = decide_drop_or_retain(
        summary_df=summary_df,
        core_fields=core_fields,
        threshold=threshold
    )

    df_clean = df.drop(columns=drop_cols, errors="ignore").copy()

    result = {
        "df_clean": df_clean,
        "summary_df": summary_df,
        "high_missing": high_missing,
        "decision_summary": decision_summary,
        "drop_cols": drop_cols,
        "retain_cols": retain_cols
    }

    return result