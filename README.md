# IDXExchange-Project
IDXExchange Data Analyst Internship Project Repo

## Merge CRMLS master datasets

Use the Python script below to maintain two separate master datasets:

- `data/CRMLSListingMaster.csv`
- `data/CRMLSSoldMaster.csv`

Run:

```powershell
python .\merge_crmls_dataset.py
```

Behavior:

- first run: bootstraps from `raw` and `new_data`
- later runs: scans the project root for newly generated `CRMLSListingYYYYMM.csv` and `CRMLSSoldYYYYMM.csv`
- only includes files from `2024-01` through the most recently completed calendar month
- keeps `listing` and `sold` as two separate output CSVs
- filters both outputs to `PropertyType == Residential`
- prints row-count checkpoints before and after concatenation, and before and after the Residential filter
- if a month already exists and you generate that month again, the script replaces that month inside the corresponding master dataset instead of duplicating it

If your new monthly CSV files are generated into another folder, add that folder during update mode:

```powershell
python .\merge_crmls_dataset.py --scan-dir new_data
```

If you want to rebuild both master files from scratch:

```powershell
python .\merge_crmls_dataset.py --rebuild
```
