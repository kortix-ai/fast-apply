import json
import sys
import argparse
import pandas as pd

def count_fixed_entries(filename):
    if filename.endswith('.json'):
        with open(filename, 'r') as file:
            data = json.load(file)
    elif filename.endswith('.parquet'):
        df = pd.read_parquet(filename)
        data = df.to_dict('records')
    else:
        raise ValueError("Unsupported file format. Please use .json or .parquet")
    
    total_entries = len(data)
    fixed_entries = sum(1 for entry in data if entry.get('status') == 'fixed')
    
    return fixed_entries, total_entries

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count fixed entries in JSON or Parquet file")
    parser.add_argument("--input", required=True, help="Path to input file (.json or .parquet)")
    args = parser.parse_args()
    
    filename = args.input
    
    try:
        fixed, total = count_fixed_entries(filename)
        print(f"Number of fixed entries: {fixed}")
        print(f"Total number of entries: {total}")
        print(f"Ratio of fixed entries: {fixed}/{total}")
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: '{filename}' is not a valid JSON file.")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        sys.exit(1)
