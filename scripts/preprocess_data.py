import pandas as pd
import os
from sklearn.model_selection import train_test_split

# File paths
RAW_DATA_PATH = "data/raw/products.csv"
PROCESSED_DATA_PATH = "data/processed/processed_data.csv"

def preprocess_data():
    
    df = pd.read_csv(RAW_DATA_PATH, header=None)
    df.columns = ["id", "title", "vendor_id", "unknown", "product_name", "category_id", "category"]

    # Basic cleaning: Drop duplicates
    df = df.drop_duplicates(subset="title").reset_index(drop=True)

    # Map categories to IDs for classification
    category_mapping = {cat: idx for idx, cat in enumerate(df["category"].unique())}
    df["category_label"] = df["category"].map(category_mapping)

    # Save the mapping for later use
    with open("data/processed/category_mapping.json", "w") as f:
        import json
        json.dump(category_mapping, f)

    # Split dataset into train and test
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["category_label"])
    
    train.to_csv(PROCESSED_DATA_PATH.replace("processed_data.csv", "train.csv"), index=False)
    test.to_csv(PROCESSED_DATA_PATH.replace("processed_data.csv", "test.csv"), index=False)

    print(f"Preprocessing complete. Data saved to {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    preprocess_data()
