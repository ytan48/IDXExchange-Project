# %% [markdown]
# # 03_Data_Cleaning
# 
# This notebook starts from `listing_clean_missing.parquet` and `sold_clean_missing.parquet`, which were created in a prior missing-value cleaning notebook. In that earlier step, high-missing non-core columns were removed. Therefore, this notebook focuses on duplicate-column removal, datatype conversion, invalid value checks, date consistency checks, and geographic data quality checks.

# %% [markdown]
# ## Imports

# %%
import pandas as pd
import numpy as np

# %% [markdown]
# ## Data Exploration
# 
# ### Load the two datasets & Make working copies

# %%
sold = pd.read_parquet("data/processed/sold_clean_missing.parquet")
listing = pd.read_parquet("data/processed/listing_clean_missing.parquet")

# %%
sold_df = sold.copy()
listing_df = listing.copy()

# %%
sold_df.head()

# %%
listing_df.head()

# %% [markdown]
# ### Check shape and columns

# %%
print("sold_df shape:", sold_df.shape)
print("listing_df shape:", listing_df.shape)

# %%
sold_cols = sold_df.columns.tolist()
listing_cols = listing_df.columns.tolist()

print("Sold columns:")
print(sold_cols)

print("\nListing columns:")
print(listing_cols)

# %% [markdown]
# ## Data Cleaning
# 
# ### Convert date fields to datetime
# 
# As an initial standardization step, the main transaction-related date fields are converted from raw string/object values into pandas datetime format. This transformation makes date values comparable across both datasets and prevents later timeline checks from failing because of inconsistent types.
# 
# Using `errors="coerce"` also turns malformed date strings into missing values, which is preferable to silently keeping invalid text in fields that will later be used for date-based quality checks.

# %%
date_cols = [
    "CloseDate",
    "PurchaseContractDate",
    "ListingContractDate",
    "ContractStatusChangeDate"
]

for col in date_cols:
    if col in sold_df.columns:
        sold_df[col] = pd.to_datetime(sold_df[col], errors="coerce")
    if col in listing_df.columns:
        listing_df[col] = pd.to_datetime(listing_df[col], errors="coerce")

# %%
print("Sold Date Types:")
print(sold_df[[col for col in date_cols if col in sold_df.columns]].dtypes)
print("\nListing Date Types:")
print(listing_df[[col for col in date_cols if col in listing_df.columns]].dtypes)

# %% [markdown]
# ### Remove redundant columns
# 
# #### Check duplicates within each dataset (duplicates and ending with `__dup` or `__dup2` columns)

# %%
# Check for duplicate columns
listing_dupes = listing_df.columns[listing_df.columns.duplicated()].tolist()
sold_dupes = sold_df.columns[sold_df.columns.duplicated()].tolist()

print("Listing duplicates:", listing_dupes)
print("Sold duplicates:", sold_dupes)

# %%
# Identify columns with __dup pattern
listing_dup_pattern = [col for col in listing_cols if '__dup' in col]
sold_dup_pattern = [col for col in sold_cols if '__dup' in col]

print("\nListing columns with __dup pattern:", listing_dup_pattern)
print("Sold columns with __dup pattern:", sold_dup_pattern)

# %% [markdown]
# #### Drop the duplicate columns based on the identified patterns
# 
# After identifying columns that were created as duplicate artifacts (for example, names ending in `__dup` or `__dup2`), those redundant versions are removed from the working DataFrames. This transformation keeps one canonical copy of each field so later analysis is not distorted by repeated information or ambiguous column selection.

# %%
# Drop the duplicate columns based on the identified patterns
sold_df = sold_df.drop(columns=sold_dup_pattern, errors="ignore")
listing_df = listing_df.drop(columns=listing_dup_pattern, errors="ignore")

print(f"Dropped {len(sold_dup_pattern)} columns from sold_df")
print(f"Dropped {len(listing_dup_pattern)} columns from listing_df")

# %%
# check the columns again after dropping duplicates
print("Sold columns after dropping duplicates:")
print(sold_df.columns.tolist())
print("\nListing columns after dropping duplicates:")
print(listing_df.columns.tolist())

# %% [markdown]
# Sold data has `latfilled` and `lonfilled`, check what thoes are and what they look like

# %%
print(sold_df["latfilled"].unique())
print(sold_df["lonfilled"].unique())

# %%
sold_df[["latfilled", "lonfilled"]].sample(10)

# %%
print(sold_df["latfilled"].value_counts(dropna=False))
print(sold_df["lonfilled"].value_counts(dropna=False))

# %% [markdown]
# ### Remove helper columns latfilled and lonfilled
# 
# The `latfilled` and `lonfilled` fields act as helper indicators showing whether coordinates may have been filled during earlier processing. They are not substantive property attributes, so they are removed from both datasets to keep the cleaned outputs focused on analytical variables rather than internal workflow metadata.
# 

# %%
helper_cols_to_drop = ["latfilled", "lonfilled"]

sold_df = sold_df.drop(columns=helper_cols_to_drop, errors="ignore")
listing_df = listing_df.drop(columns=helper_cols_to_drop, errors="ignore")

# %%
print("Original Sold shape:", sold.shape, "Original Listing shape:", listing.shape)
print("Sold columns after dropping helper columns:", sold_df.shape, "Listing columns after dropping helper columns:", listing_df.shape)

# %% [markdown]
# ### Standardize blank strings as missing values
# 
# The input files used in this notebook were already processed in a prior missing-value cleaning notebook, where high-missing non-core columns were removed. Here, the transformation is narrower: empty strings and whitespace-only text are converted to `NaN` so that missing values are represented consistently across the two datasets.
# 
# This matters because blank strings are not treated as missing by default, which can lead to misleading counts, inconsistent type conversion, and weaker quality checks later in the notebook.

# %%
# Standardize blank strings as missing values

for df in [sold_df, listing_df]:
    object_cols = df.select_dtypes(include="object").columns
    df[object_cols] = df[object_cols].replace(r"^\s*$", np.nan, regex=True)

# %% [markdown]
# #### Confirm date columns used later
# 
# Date columns were already converted to datetime earlier in the notebook. This section simply keeps the set of date fields explicit before the later timeline checks, so the notebook does not repeat the same transformation twice.

# %%
date_cols = [col for col in date_cols if col in sold_df.columns or col in listing_df.columns]
print("Date columns used in later checks:", date_cols)

# %% [markdown]
# #### Ensure numeric fields are properly typed
# 
# Key numeric columns such as prices, coordinates, square footage, room counts, and lot-size measures are explicitly converted to numeric dtype. This transformation is necessary because values may arrive as strings after ingestion, and numeric validation rules only work reliably when these fields are stored as numbers.
# 
# Again, `errors="coerce"` is used so non-numeric entries become missing rather than remaining as invalid text.

# %%
# Check for numeric columns and convert them to numeric types
numeric_cols = [
    "file_period",
    "OriginalListPrice",
    "ListingKey",
    "ClosePrice",
    "Latitude",
    "Longitude",
    "LivingArea",
    "ListPrice",
    "DaysOnMarket",
    "ListingKeyNumeric",
    "ParkingTotal",
    "LotSizeAcres",
    "YearBuilt",
    "StreetNumberNumeric",
    "BathroomsTotalInteger",
    "BedroomsTotal",
    "Stories",
    "LotSizeArea",
    "MainLevelBedrooms",
    "GarageSpaces",
    "AssociationFee",
    "LotSizeSquareFeet",
    "BuyerAgencyCompensation"
]

for df in [sold_df, listing_df]:
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

# %% [markdown]
# Check numeric dtypes

# %%
print(sold_df[[col for col in numeric_cols if col in sold_df.columns]].dtypes)
print()
print(listing_df[[col for col in numeric_cols if col in listing_df.columns]].dtypes)

# %% [markdown]
# Missing value summary

# %%
print("Sold missing values:")
print(sold_df.isna().sum().sort_values(ascending=False).head(20))

print("\nListing missing values:")
print(listing_df.isna().sum().sort_values(ascending=False).head(20))

# %% [markdown]
# ### *Flag* invalid numeric values
# 
# This step creates boolean flag columns for clearly implausible numeric values, such as nonpositive close price or living area and negative counts for market days, bedrooms, or bathrooms. The goal is to document which rows violate basic domain rules before any records are removed.
# 
# A combined `any_invalid_numeric_flag` is also created so the notebook can later filter records in a transparent and reproducible way.

# %%
for df in [sold_df, listing_df]:
    closeprice = df.get("ClosePrice", pd.Series(np.nan, index=df.index))
    livingarea = df.get("LivingArea", pd.Series(np.nan, index=df.index))
    daysonmarket = df.get("DaysOnMarket", pd.Series(np.nan, index=df.index))
    bedroomstotal = df.get("BedroomsTotal", pd.Series(np.nan, index=df.index))
    bathroomstotal = df.get("BathroomsTotalInteger", pd.Series(np.nan, index=df.index))

    df["invalid_closeprice_flag"] = closeprice.notna() & (closeprice <= 0)
    df["invalid_livingarea_flag"] = livingarea.notna() & (livingarea <= 0)
    df["invalid_daysonmarket_flag"] = daysonmarket.notna() & (daysonmarket < 0)
    df["invalid_bedrooms_flag"] = bedroomstotal.notna() & (bedroomstotal < 0)
    df["invalid_bathrooms_flag"] = bathroomstotal.notna() & (bathroomstotal < 0)

    df["any_invalid_numeric_flag"] = (
        df["invalid_closeprice_flag"]
        | df["invalid_livingarea_flag"]
        | df["invalid_daysonmarket_flag"]
        | df["invalid_bedrooms_flag"]
        | df["invalid_bathrooms_flag"]
    )

# %%
num_invalid_sold = sold_df["any_invalid_numeric_flag"].sum()
num_invalid_listing = listing_df["any_invalid_numeric_flag"].sum()
print(f"Sold dataset has {num_invalid_sold} of invalid numeric values.")
print(f"Listing dataset has {num_invalid_listing} of invalid numeric values.")

# %% [markdown]
# ### *Check* invalid numeric counts

# %%
invalid_cols = [
    "invalid_closeprice_flag",
    "invalid_livingarea_flag",
    "invalid_daysonmarket_flag",
    "invalid_bedrooms_flag",
    "invalid_bathrooms_flag",
    "any_invalid_numeric_flag"
]

print("Sold invalid numeric counts:")
print(sold_df[invalid_cols].sum())

print("\nListing invalid numeric counts:")
print(listing_df[invalid_cols].sum())

# %%
cols_to_look = [
    "ClosePrice", "LivingArea", "DaysOnMarket", "BedroomsTotal", "BathroomsTotalInteger", 
    "any_invalid_numeric_flag"
]

print("Sold sample:")
print(sold_df.loc[sold_df["any_invalid_numeric_flag"], cols_to_look].head(10))
print("\nListing sample:")
print(listing_df.loc[listing_df["any_invalid_numeric_flag"], cols_to_look].head(10))

# %% [markdown]
# ### Date consistency checks
# 
# These transformations add flag columns that test whether the ordering of major transaction dates makes sense. Records are marked when listing dates occur after close dates, when purchase contract dates occur after close dates, or when the purchase contract date appears before the listing contract date.
# 
# The purpose is to preserve an auditable record of temporal inconsistencies instead of dropping suspicious rows without showing why they were considered invalid.
# 
# **Required flags:**
# - listing_after_close_flag
# - purchase_after_close_flag
# - negative_timeline_flag
# - any_date_issue_flag

# %%
for df in [sold_df, listing_df]:
    listing_date = df.get("ListingContractDate", pd.Series(np.nan, index=df.index))
    purchase_date = df.get("PurchaseContractDate", pd.Series(np.nan, index=df.index))
    close_date = df.get("CloseDate", pd.Series(np.nan, index=df.index))

    df["listing_after_close_flag"] = (
        listing_date.notna() &
        close_date.notna() &
        (listing_date > close_date)
    )

    df["purchase_after_close_flag"] = (
        purchase_date.notna() &
        close_date.notna() &
        (purchase_date > close_date)
    )

    df["negative_timeline_flag"] = (
        listing_date.notna() &
        purchase_date.notna() &
        (purchase_date < listing_date)
    )

    df["any_date_issue_flag"] = (
        df["listing_after_close_flag"]
        | df["purchase_after_close_flag"]
        | df["negative_timeline_flag"]
    )

# %% [markdown]
# #### Count date issues

# %%
date_flag_cols = [
    "listing_after_close_flag",
    "purchase_after_close_flag",
    "negative_timeline_flag",
    "any_date_issue_flag"
]

print("Sold date issue counts:")
print(sold_df[date_flag_cols].sum())

print("\nListing date issue counts:")
print(listing_df[date_flag_cols].sum())

# %% [markdown]
# ### Geographic data checks
# 
# This section creates geographic quality flags for missing coordinates, zero coordinates, positive longitude values, and latitude/longitude pairs that fall outside a plausible California range. These checks are used to identify records whose spatial information is incomplete or unrealistic for the study area.
# 
# Creating explicit flags first makes the later filtering step easier to justify and summarize.

# %%
for df in [sold_df, listing_df]:
    latitude = df.get("Latitude", pd.Series(np.nan, index=df.index))
    longitude = df.get("Longitude", pd.Series(np.nan, index=df.index))

    df["missing_coordinates_flag"] = latitude.isna() | longitude.isna()

    df["zero_coordinates_flag"] = ((latitude == 0) | (longitude == 0))

    df["positive_longitude_flag"] = longitude.notna() & (longitude > 0)

    df["out_of_state_or_implausible_flag"] = (
        latitude.notna() & longitude.notna() &
        (
            (latitude < 32) | (latitude > 42.5) |
            (longitude < -125) | (longitude > -114)
        )
    )

    df["any_geo_issue_flag"] = (
        df["missing_coordinates_flag"]
        | df["zero_coordinates_flag"]
        | df["positive_longitude_flag"]
        | df["out_of_state_or_implausible_flag"]
    )

# %% [markdown]
# #### Count geographic issues

# %%
geo_flag_cols = [
    "missing_coordinates_flag",
    "zero_coordinates_flag",
    "positive_longitude_flag",
    "out_of_state_or_implausible_flag",
    "any_geo_issue_flag"
]

print("Sold geographic issue counts:")
print(sold_df[geo_flag_cols].sum())

print("\nListing geographic issue counts:")
print(listing_df[geo_flag_cols].sum())

# %% [markdown]
# View sample flagged records

# %%
# Display rows for sold data
sold_df.loc[
    sold_df["any_invalid_numeric_flag"]
    | sold_df["any_date_issue_flag"]
    | sold_df["any_geo_issue_flag"]
].head(10)

# %%
# Display rows for listing data
listing_df.loc[
    listing_df["any_invalid_numeric_flag"]
    | listing_df["any_date_issue_flag"]
    | listing_df["any_geo_issue_flag"]
].head(10)

# %% [markdown]
# ### Final Filter
# 
# Once all numeric, date, and geographic issue flags have been created, the notebook removes rows with any flagged problem to produce the final cleaned datasets. This transformation is the main record-level filtering step, and it is placed at the end so each removal reason has already been defined and counted.

# %%
# Remove rows with any of the identified issues for both datasets
sold_remove_mask = (
    sold_df["any_invalid_numeric_flag"]
    | sold_df["any_date_issue_flag"]
    | sold_df["any_geo_issue_flag"]
)

listing_remove_mask = (
    listing_df["any_invalid_numeric_flag"]
    | listing_df["any_date_issue_flag"]
    | listing_df["any_geo_issue_flag"]
)

sold_cleaned = sold_df.loc[
    ~sold_remove_mask
].copy()

listing_cleaned = listing_df.loc[
    ~listing_remove_mask
].copy()

# %% [markdown]
# ### Row count summary

# %%
print("Sold rows flagged by issue type:")
print(pd.Series({
    "invalid_numeric": sold_df["any_invalid_numeric_flag"].sum(),
    "date_issue": sold_df["any_date_issue_flag"].sum(),
    "geo_issue": sold_df["any_geo_issue_flag"].sum(),
    "rows_removed_total": sold_remove_mask.sum()
} ))

print("Sold rows before cleaning:", len(sold_df))
print("Sold rows after cleaning:", len(sold_cleaned))
print("Sold rows removed:", len(sold_df) - len(sold_cleaned))

print("\nListing rows flagged by issue type:")
print(pd.Series({
    "invalid_numeric": listing_df["any_invalid_numeric_flag"].sum(),
    "date_issue": listing_df["any_date_issue_flag"].sum(),
    "geo_issue": listing_df["any_geo_issue_flag"].sum(),
    "rows_removed_total": listing_remove_mask.sum()
} ))

print("\nListing rows before cleaning:", len(listing_df))
print("Listing rows after cleaning:", len(listing_cleaned))
print("Listing rows removed:", len(listing_df) - len(listing_cleaned))

# %% [markdown]
# ### Save outputs as csv. file
# 
# The final transformation result is written to `data/processed` as CSV files so the cleaned datasets can be reused in later notebooks or external tools. Saving the outputs here preserves a stable post-cleaning version of both tables for downstream analysis.

# %%
from pathlib import Path

output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

sold_cleaned.to_csv(output_dir / "sold_cleaned_final.csv", index=False)
listing_cleaned.to_csv(output_dir / "listing_cleaned_final.csv", index=False)

# %% [markdown]
# ## Data Cleaning Summary
# 
# This notebook begins with datasets that were previously processed for high-missing columns. In this notebook, redundant duplicate columns were removed, blank strings were standardized as missing values, date columns were converted to datetime format, and key numeric columns were converted to numeric types.
# 
# Invalid numeric values were flagged for records with nonpositive close price or living area, negative days on market, and negative bedroom or bathroom counts. Date consistency flags were created to identify records where listing or purchase dates occurred after the close date, or where the purchase date occurred before the listing date. Geographic quality checks were also applied to flag missing coordinates, zero coordinates, positive longitude values, and implausible out-of-state coordinates.
# 
# After these checks, flagged invalid records were removed to create final cleaned, analysis-ready datasets for downstream analysis. The cleaned outputs were then saved as final CSV files.


