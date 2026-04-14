from pathlib import Path
import pandas as pd

# 找到项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 数据文件夹
DATA_DIR = PROJECT_ROOT / "data"

def load_data():
    data_listing = pd.read_csv(DATA_DIR / "CRMLSListingMaster.csv", low_memory=False)
    data_sold = pd.read_csv(DATA_DIR / "CRMLSSoldMaster.csv", low_memory=False)
    return data_listing, data_sold