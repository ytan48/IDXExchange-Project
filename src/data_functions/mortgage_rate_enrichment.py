# Step 1 – Fetch the mortgage rate data from FRED
import pandas as pd
url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"
mortgage = pd.read_csv(url, parse_dates=['observation_date'])
mortgage.columns = ['date', 'rate_30yr_fixed']


# Step 2 – Resample weekly rates to monthly averages
mortgage['year_month'] = mortgage['date'].dt.to_period('M')
mortgage_monthly = (
 mortgage.groupby('year_month')['rate_30yr_fixed']
 .mean()
 .reset_index()
)


# Step 3 – Create a matching year_month key on the MLS datasets
# Sold dataset — key off CloseDate

sold = pd.read_csv("data\CRMLSSoldMaster.csv", parse_dates=['CloseDate'])
listings = pd.read_csv("data\CRMLSListingMaster.csv", parse_dates=['ListingContractDate'])


sold['year_month'] = pd.to_datetime(sold['CloseDate']).dt.to_period('M')
# Listings dataset — key off ListingContractDate
listings['year_month'] = pd.to_datetime(
 listings['ListingContractDate']
).dt.to_period('M')


# Step 4 – Merge
sold_with_rates = sold.merge(mortgage_monthly, on='year_month', how='left')
listings_with_rates = listings.merge(mortgage_monthly, on='year_month', how='left')


# Step 5 – Validate the merge
# Check for any unmatched rows (rate should not be null)
print("Checking for unmatched rows in sold_with_rates:")
print(sold_with_rates['rate_30yr_fixed'].isnull().sum())
# ---
print("Checking for unmatched rows in listings_with_rates:")
print(listings_with_rates['rate_30yr_fixed'].isnull().sum())

# Preview
print("Preview of sold_with_rates:")
print(
 sold_with_rates[
 ['CloseDate', 'year_month', 'ClosePrice', 'rate_30yr_fixed']
 ].head()
)

# Step 6 – Validation check (strict)
sold_null_count = sold_with_rates['rate_30yr_fixed'].isnull().sum()
listings_null_count = listings_with_rates['rate_30yr_fixed'].isnull().sum()

print("Null mortgage rates in sold_with_rates:", sold_null_count)
print("Null mortgage rates in listings_with_rates:", listings_null_count)

assert sold_null_count == 0, (
    f"Validation failed: sold_with_rates has {sold_null_count} null mortgage rate values."
)

assert listings_null_count == 0, (
    f"Validation failed: listings_with_rates has {listings_null_count} null mortgage rate values."
)

print("Validation passed: no null mortgage rate values exist after either merge.")


# Step 7 – Save enriched datasets as new CSV files
sold_with_rates.to_csv("data/CRMLSSoldMaster_enriched.csv", index=False)
listings_with_rates.to_csv("data/CRMLSListingMaster_enriched.csv", index=False)

print("Saved:")
print(" - data/CRMLSSoldMaster_enriched.csv")
print(" - data/CRMLSListingMaster_enriched.csv")
