import pandas as pd

def print_parquet_columns(file_path):
    # Read the Parquet file
    df = pd.read_parquet(file_path)
    
    # Print the column names
    print("Columns in the Parquet file:")
    for column in df.columns:
        print(f"- {column}")

if __name__ == "__main__":
    file_path = "data/train.parquet"
    print_parquet_columns(file_path)
