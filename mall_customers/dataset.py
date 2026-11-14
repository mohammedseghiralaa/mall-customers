import pandas as pd
from mall_customers.config import RAW_DATA_DIR


def load_data(filename="Mall_Customers.csv"):
    return pd.read_csv(RAW_DATA_DIR / filename)


def check_data(df):
    print("Shape:", df.shape)
    print("\nMissing values:\n", df.isna().sum())
    print("\nDuplicates:", df.duplicated().sum())
    print("\nSummary:\n", df.describe().round(1))
    