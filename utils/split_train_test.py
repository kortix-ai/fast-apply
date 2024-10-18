# File name: split_train_test.py

import argparse
import random
import os
import json
import pandas as pd

def read_input_file(input_file):
    file_extension = os.path.splitext(input_file)[1].lower()
    if file_extension == '.parquet':
        return pd.read_parquet(input_file)
    elif file_extension == '.json':
        with open(input_file, 'r') as f:
            return pd.DataFrame(json.load(f))
    else:
        raise ValueError("Unsupported file format. Please use .parquet or .json files.")

def write_output_file(data, output_file):
    file_extension = os.path.splitext(output_file)[1].lower()
    if file_extension == '.parquet':
        data.to_parquet(output_file, index=False)
    elif file_extension == '.json':
        data.to_json(output_file, orient='records', lines=True)
    else:
        raise ValueError("Unsupported output format. Please use .parquet or .json extension.")

def split_dataset(input_file, train_file, test_file, num_test=100):
    df = read_input_file(input_file)
    
    # Shuffle the dataframe
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split the data
    test_data = df.head(num_test)
    train_data = df.tail(len(df) - num_test)
    
    # Write test data
    write_output_file(test_data, test_file)
    
    # Write train data
    write_output_file(train_data, train_file)
    
    print(f"Dataset split complete.")
    print(f"Test set: {len(test_data)} examples")
    print(f"Train set: {len(train_data)} examples")

def main():
    parser = argparse.ArgumentParser(description="Split Parquet or JSON dataset into train and test sets")
    parser.add_argument("--input", "-i", required=True, help="Input file (Parquet or JSON) containing the full dataset")
    parser.add_argument("--train", "-tr", help="Output file name for the training set (default: train.[input_extension])")
    parser.add_argument("--test", "-te", help="Output file name for the test set (default: test.[input_extension])")
    parser.add_argument("--num_test", "-n", type=int, default=100, help="Number of examples for the test set (default: 100)")
    
    args = parser.parse_args()
    
    input_path = args.input
    input_dir = os.path.dirname(input_path)
    input_extension = os.path.splitext(input_path)[1]
    
    train_file = args.train if args.train else os.path.join(input_dir, f"train{input_extension}")
    test_file = args.test if args.test else os.path.join(input_dir, f"test{input_extension}")
    
    split_dataset(input_path, train_file, test_file, args.num_test)

if __name__ == "__main__":
    main()