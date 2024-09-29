import argparse
import pandas as pd
import sys
import os

def merge_parquet_files(input_files, output_file):
    try:
        print(f"Starting to merge {len(input_files)} parquet files...")
        
        # Read and concatenate all input parquet files
        dfs = []
        for file in input_files:
            print(f"Reading file: {file}          ", end='')
            df = pd.read_parquet(file)
            print(f"|   {df.shape}")
            dfs.append(df)
        
        print("Concatenating dataframes...")
        merged_df = pd.concat(dfs, ignore_index=True)
        print(f"Merged dataframe shape: {merged_df.shape}")

        # Drop duplicates
        print("Dropping duplicates...")
        merged_df.drop_duplicates(subset=['original_code', 'update_snippet'], inplace=True)
        print(f"Dataframe shape after dropping duplicates: {merged_df.shape}")

        # Shuffle based on original_code and update_snippet
        print("Shuffling dataframe...")
        merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)
        print("Dataframe shuffled.")

        # Write the merged dataframe to a new parquet file
        print(f"Saving merged file as {output_file}")
        merged_df.to_parquet(output_file, index=False)
        print(f"Merged file saved successfully.")
        print(merged_df.columns)

        # Write the merged dataframe to a JSON file
        json_output = os.path.splitext(output_file)[0] + '.json'
        print(f"Saving JSON copy as {json_output}")
        merged_df.to_json(json_output, orient='records', indent=2)
        print(f"JSON copy saved successfully.")
        
        print("Merge process completed.")

        # Check for empty values
        empty_original_code = merged_df['original_code'].isna().sum()
        empty_update_snippet = merged_df['update_snippet'].isna().sum()

        print(f"\nRows with empty 'original_code': {empty_original_code}")
        print(f"Rows with empty 'update_snippet': {empty_update_snippet}")

        print(f"\nFinal dataframe shape: {merged_df.shape}")
    except Exception as e:
        print(f"An error occurred: {str(e)}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Merge multiple parquet files and output the result as parquet and JSON.")
    parser.add_argument("input_files", nargs='+', help="Paths to the input parquet files")
    parser.add_argument("--output", required=True, help="Path to the output merged parquet file (JSON will be created with the same name)")
    
    args = parser.parse_args()
    
    merge_parquet_files(args.input_files, args.output)

if __name__ == "__main__":
    main()
