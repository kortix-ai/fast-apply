import pandas as pd
import json
import argparse
from tqdm import tqdm

def parquet_to_jsonl(input_file, output_file):
    # Read the Parquet file
    df = pd.read_parquet(input_file)

    # Open the output file in write mode
    with open(output_file, 'w') as f:
        # Iterate through each row in the DataFrame with tqdm progress bar
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Converting"):
            # Create a dictionary for each row
            entry = {
                "instruction": f"Apply changes from <update_snippet> to <original_code>. Output only the complete updated code in <full_updated_code> tag.",
                "context": f"<original_code>{row['original_code']}</original_code>\n<update_snippet>{row['update_snippet']}</update_snippet>",
                "response": f"<full_updated_code>{row['final_code']}</full_updated_code>",
                "category": "closed_qa"
            }
            
            # Write the JSON object to the file, followed by a newline
            json.dump(entry, f)
            f.write('\n')

def main():
    parser = argparse.ArgumentParser(description='Convert Parquet file to JSONL format.')
    parser.add_argument('input_file', help='Input Parquet file path')
    parser.add_argument('output_file', help='Output JSONL file path')
    args = parser.parse_args()

    parquet_to_jsonl(args.input_file, args.output_file)
    print(f"Conversion complete. JSONL file saved as {args.output_file}")

if __name__ == "__main__":
    main()
