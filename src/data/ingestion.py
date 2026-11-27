import pandas as pd
import sqlite3
import kagglehub
import shutil

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT_DIR / "data" / "raw"

CSV_PATH = RAW_DIR / "telco_churn.csv"

DB_PATH = RAW_DIR / "customers.db"

RAW_DIR.mkdir(parents=True, exist_ok=True)


def download_dataset():

    path = kagglehub.dataset_download(
        "blastchar/telco-customer-churn"
    )

    source = Path(path) / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

    shutil.copy2(source, CSV_PATH)

    print("Dataset downloaded successfully")


def load_dataset():

    df = pd.read_csv(CSV_PATH)

    print("Dataset shape:", df.shape)

    return df

def validate_dataset(df):

    if df.empty:
        raise ValueError("Dataset is empty")

    if "Churn" not in df.columns:
        raise ValueError("Target column is missing")

    if "customerID" not in df.columns:
        raise ValueError("Customer ID is missing")

    if df["customerID"].duplicated().any():
        raise ValueError("Duplicate customer IDs found")

    print("Data validation successful")

def save_to_sqlite(df):

    conn = sqlite3.connect(DB_PATH)

    try:

        df.to_sql(
            name="customers",
            con=conn,
            if_exists="replace",
            index=False
        )

        print("Data stored in SQLite successfully")

    finally:

        conn.close()


if __name__ == "__main__":

    download_dataset()

    df = load_dataset()

    validate_dataset(df)

    save_to_sqlite(df)

    print("Data ingestion completed successfully")

    