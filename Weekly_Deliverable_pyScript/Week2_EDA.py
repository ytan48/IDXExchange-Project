# %% [markdown]
# # 01 Exploratory Data Analysis
# 
# This notebook performs exploratory data analysis on the raw CRMLS listing and sold files before any cleaning or feature engineering. The goal is to understand the property type mix, summarize key residential sale patterns, inspect days on market and sale-to-list behavior, check for basic date inconsistencies, and compare county-level price patterns.
# 
# ## Imports

# %%
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import sys
from pathlib import Path
PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_functions.merge_crmls_dataset_unfiltered import merge_raw_crmls_data_unfiltered

# %% [markdown]
# Raw Data Loading

# %%
listing_df, sold_df = merge_raw_crmls_data_unfiltered(write_csv=False)
display(listing_df.head())

# %%
display(sold_df.head())

# %%
print("Listing shape:", listing_df.shape)
print("Sold shape:", sold_df.shape)

# %% [markdown]
# key numeric columns:

# %%
numeric_cols = [
    "ClosePrice",
    "ListPrice",
    "OriginalListPrice",
    "LivingArea",
    "LotSizeAcres",
    "BedroomsTotal",
    "BathroomsTotalInteger",
    "DaysOnMarket",
    "YearBuilt"
]

# %%
listing_df[numeric_cols].dtypes

# %%
sold_df[numeric_cols].dtypes

# %% [markdown]
# Convert numerical columns for both dataset

# %%
sold_df[numeric_cols] = sold_df[numeric_cols].apply(pd.to_numeric, errors="coerce")

listing_df[numeric_cols] = listing_df[numeric_cols].apply(pd.to_numeric, errors="coerce")

display(sold_df[numeric_cols].dtypes)

# %% [markdown]
# Convert sold date columns

# %%
date_cols = [
    "CloseDate",
    "PurchaseContractDate",
    "ListingContractDate",
    "ContractStatusChangeDate"
]

# %%
for col in date_cols:
    if col in sold_df.columns:
        sold_df[col] = pd.to_datetime(sold_df[col], errors="coerce")

# %% [markdown]
# ### 1. Residential vs other property type share for **listings**

# %%
listing_property_counts = listing_df["PropertyType"].value_counts(dropna=False)
listing_total = listing_property_counts.sum()

listing_residential_count = listing_property_counts.get("Residential", 0)
listing_other_count = listing_total - listing_residential_count

listing_property_share = pd.DataFrame({
    "group": ["Residential", "Other"],
    "count": [listing_residential_count, listing_other_count],
    "percent": [
        listing_residential_count / listing_total * 100,
        listing_other_count / listing_total * 100
    ]
})

display(listing_property_share)

# %% [markdown]
# ### 2. Residential vs other property type share for **sold**

# %%
sold_property_counts = sold_df["PropertyType"].value_counts(dropna=False)
sold_total = sold_property_counts.sum()

sold_residential_count = sold_property_counts.get("Residential", 0)
sold_other_count = sold_total - sold_residential_count

sold_property_share = pd.DataFrame({
    "group": ["Residential", "Other"],
    "count": [sold_residential_count, sold_other_count],
    "percent": [
        sold_residential_count / sold_total * 100,
        sold_other_count / sold_total * 100
    ]
})

display(sold_property_share)

# %% [markdown]
# ### 3. Create a Residential sold subset

# %%
sold_residential = sold_df[sold_df["PropertyType"] == "Residential"].copy()

print("Residential sold shape:", sold_residential.shape)
display(sold_residential[numeric_cols].dtypes)

# %% [markdown]
# ### 4. Median and average close prices

# %%
close_price_summary = pd.DataFrame({
    "metric": ["median_close_price", "average_close_price"],
    "value": [
        sold_residential["ClosePrice"].median(),
        sold_residential["ClosePrice"].mean()
    ]
})

display(close_price_summary)

# %% [markdown]
# ### 5. Days on Market distribution

# %%
dom = sold_residential["DaysOnMarket"].dropna()

dom_summary = pd.DataFrame({
    "statistic": [
        "count", "mean", "median", "min",
        "p25", "p75", "p95", "p99", "max",
        "negative_count"
    ],
    "value": [
        dom.shape[0],
        dom.mean(),
        dom.median(),
        dom.min(),
        dom.quantile(0.25),
        dom.quantile(0.75),
        dom.quantile(0.95),
        dom.quantile(0.99),
        dom.max(),
        (dom < 0).sum()
    ]
})

display(dom_summary)

# %% [markdown]
# #### 5.1 Trimmed Days on Market histogram

# %%
dom = sold_residential["DaysOnMarket"].dropna()

plt.figure(figsize=(8, 4))
plt.hist(dom, bins=40, edgecolor="black")
plt.title("Histogram of DaysOnMarket (Residential Sold, Raw Data)")
plt.xlabel("DaysOnMarket")
plt.ylabel("Frequency")
plt.show()

# %% [markdown]
# ### 6. Percentage sold above vs below list price

# %%
prices = sold_residential.dropna(subset=["ClosePrice", "ListPrice"])
diff = prices["ClosePrice"] - prices["ListPrice"]

sale_to_list_summary = pd.DataFrame([
    {"group": "Above list", "count": (diff > 0).sum()},
    {"group": "Below list", "count": (diff < 0).sum()},
    {"group": "At list",    "count": (diff == 0).sum()}
])

sale_to_list_summary["percent"] = (sale_to_list_summary["count"] / len(diff)) * 100

display(sale_to_list_summary)

# %% [markdown]
# ### 7. Date consistency checks

# %%
date_issue_summary = pd.DataFrame({
    "check": [
        "CloseDate before ListingContractDate",
        "CloseDate before PurchaseContractDate",
        "PurchaseContractDate before ListingContractDate"
    ],
    "issue_count": [
        (sold_residential["CloseDate"] < sold_residential["ListingContractDate"]).sum(),
        (sold_residential["CloseDate"] < sold_residential["PurchaseContractDate"]).sum(),
        (sold_residential["PurchaseContractDate"] < sold_residential["ListingContractDate"]).sum()
    ]
})

display(date_issue_summary)

# %% [markdown]
# ### 8. Counties with the highest median prices

# %%
county_price_summary = (
    sold_residential.groupby("CountyOrParish")["ClosePrice"]
    .describe()[['50%', 'mean', 'count']].reset_index()
    .rename(columns={"50%": "median_close_price"})
)

display(county_price_summary.nlargest(15, "median_close_price"))

# %% [markdown]
# #### 8.1 Counties with at least 50 sales

# %%
county_price_summary_50 = county_price_summary[
    county_price_summary["count"] >= 50
].sort_values("median_close_price", ascending=False).reset_index(drop=True)

display(county_price_summary_50.head(10))

# %% [markdown]
# ## Conclusion:

# %%
print("Residential share in listings:",
      round(listing_residential_count / listing_total * 100, 2), "%")

print("Residential share in sold:",
      round(sold_residential_count / sold_total * 100, 2), "%")

print("Median residential close price:",
      round(sold_residential["ClosePrice"].median(), 2))

print("Average residential close price:",
      round(sold_residential["ClosePrice"].mean(), 2))

print("Median residential DaysOnMarket:",
      round(dom.median(), 2))

print("Average residential DaysOnMarket:",
      round(dom.mean(), 2))

# %% [markdown]
# - Residential vs. other property type share: Residential properties make up 63.33% of listings and 67.19% of sold properties, so they are the dominant property type in both datasets. This means other property types account for 36.67% of listings and 32.81% of sold properties.
# 
# - Median and average close prices: For sold residential homes, the median close price is $820,000 and the average close price is $1,185,616.36. Since the average is much higher than the median, the close price distribution appears to be right-skewed.
# 
# - Days on Market distribution: For sold residential homes, the median Days on Market is 19 days and the average is 37.34 days. This suggests that most homes sell fairly quickly, but some homes stay on the market much longer, creating a right-skewed distribution.
# 
# - Percentage of homes sold above vs. below list price: Among sold residential homes, 40.11% sold above list price, 42.54% sold below list price, and 17.34% sold exactly at list price. Overall, slightly more homes sold below list than above list.
# 
# - Apparent date consistency issues: There are some records with inconsistent dates, including 58 cases where the close date is before the listing date, 240 cases where the close date is before the purchase contract date, and 261 cases where the purchase contract date is before the listing date.
# 
# - Counties with the highest median prices: The counties with the highest reported median close prices include San Mateo, Santa Clara, Santa Cruz, San Francisco, and Orange. However, some counties with the very highest medians have very small sales counts, so those results should be interpreted carefully.


