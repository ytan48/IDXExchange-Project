# %%
# Imports
import pandas as pd
import numpy as np
from IPython.display import display
from src.load_data import load_data

# %% 
# ===========================================================================================
# 1. Data Loading
# ===========================================================================================
data_listing, data_sold = load_data()


# %% 
# ===========================================================================================
# 2. Missing Value Analysis Functions
# ===========================================================================================
def missing_summary(df, df_name="data"):
    missing_count = df.isna().sum()
    missing_pct = df.isna().mean() * 100

    summary = pd.DataFrame({
        "column": df.columns,
        "missing_count": missing_count.values,
        "missing_pct": missing_pct.values
    }).sort_values("missing_pct", ascending=False)

    high_missing = summary[summary["missing_pct"] > 90]

    print(f"\nMissing value summary for {df_name}:")
    display(summary)

    print(f"\nColumns in {df_name} with >90% missing values:")
    display(high_missing)

    return summary, high_missing



# %% 
# ===========================================================================================
# 2.1. Run missing value analysis on both datasets (Sold and Listing)
# ===========================================================================================
listing_missing_summary, listing_high_missing = missing_summary(data_listing, "data_listing")
sold_missing_summary, sold_high_missing = missing_summary(data_sold, "data_sold")



# %% 
# ===========================================================================================
# 3. Columns that should retain
# ===========================================================================================
core_fields = [
    "ListingKey", "ListingId", "MlsStatus",
    "ListPrice", "OriginalListPrice", "ClosePrice", "CloseDate", "DaysOnMarket",
    "PropertyType", "PropertySubType",
    "BedroomsTotal", "BathroomsTotalInteger",
    "LivingArea", "LotSizeSquareFeet", "YearBuilt",
    "UnparsedAddress", "City", "StateOrProvince", "PostalCode", "CountyOrParish",
    "Latitude", "Longitude","LotSizeAcres", "DaysOnMarket"
]



# %% 
# ===========================================================================================
# 3.1. Decide which columns to drop vs retain based on missing percentage and core field status
# ===========================================================================================
def decide_drop_or_retain(summary_df, core_fields, threshold=90):
    summary_df = summary_df.copy()

    summary_df["is_core_field"] = summary_df["column"].isin(core_fields)

    summary_df["recommended_action"] = np.where(
        (summary_df["missing_pct"] > threshold) & (~summary_df["is_core_field"]),
        "drop",
        "retain"
    )

    summary_df["reason"] = np.where(
        (summary_df["missing_pct"] > threshold) & (~summary_df["is_core_field"]),
        f"missing > {threshold}% and not a core field",
        np.where(
            (summary_df["missing_pct"] > threshold) & (summary_df["is_core_field"]),
            f"missing > {threshold}% but retained because it is a core field",
            "retain"
        )
    )

    drop_cols = summary_df.loc[
        summary_df["recommended_action"] == "drop", "column"
    ].tolist()

    retain_cols = summary_df.loc[
        summary_df["recommended_action"] == "retain", "column"
    ].tolist()

    return summary_df, drop_cols, retain_cols



# %% 
# ===========================================================================================
# 3.2. Apply the decision function to both datasets and summarize the results
# ===========================================================================================
listing_decision_summary, listing_drop_cols, listing_retain_cols = decide_drop_or_retain(
    listing_missing_summary, core_fields
)

sold_decision_summary, sold_drop_cols, sold_retain_cols = decide_drop_or_retain(
    sold_missing_summary, core_fields
)

print("Listing dataset decision summary:")
display(listing_decision_summary)

print("Columns recommended to drop in data_listing:")
display(pd.DataFrame({"drop_column": listing_drop_cols}))

print("Sold dataset decision summary:")
display(sold_decision_summary)

print("Columns recommended to drop in data_sold:")
display(pd.DataFrame({"drop_column": sold_drop_cols}))



# %% 
# ===========================================================================================
# 4. Finalize the cleaned datasets by dropping the identified columns
# ===========================================================================================
data_listing_clean = data_listing.drop(columns=listing_drop_cols)
data_sold_clean = data_sold.drop(columns=sold_drop_cols)

print("Original shape of data_listing:", data_listing.shape)
print("Cleaned shape of data_listing:", data_listing_clean.shape)

print("Original shape of data_sold:", data_sold.shape)
print("Cleaned shape of data_sold:", data_sold_clean.shape)