import os
import pandas as pd
import json
import glob
import argparse

def calculate_cost(input_tokens, output_tokens):
    input_cost = (input_tokens / 1_000_000) * 1.25  # $1.25 per 1M tokens for input
    output_cost = (output_tokens / 1_000_000) * 5  # $5 per 1M tokens for output
    return input_cost + output_cost

def load_and_pair_batches(batch_dir):
    input_files = sorted(glob.glob(os.path.join(batch_dir, "batch_*.jsonl")))
    output_files = sorted(glob.glob(os.path.join(batch_dir, "output_batch_*.jsonl")))
    
    if len(input_files) != len(output_files):
        raise ValueError("Number of input and output files do not match")
    
    all_data = []
    
    for input_file, output_file in zip(input_files, output_files):
        with open(input_file, 'r') as f_in, open(output_file, 'r') as f_out:
            for input_line, output_line in zip(f_in, f_out):
                try:
                    input_data = json.loads(input_line)
                    output_data = json.loads(output_line)
                    
                    original_code = input_data['body']['messages'][1]['content'].split("<original_code>")[1].split("</original_code>")[0].strip()
                    
                    response_body = output_data['response']['body']
                    output_content = response_body['choices'][0]['message']['content']
                    
                    update_snippet = output_content.split("<update_snippet>")[1].split("</update_snippet>")[0].strip()
                    final_code = output_content.split("<final_code>")[1].split("</final_code>")[0].strip()
                    
                    input_tokens = response_body['usage']['prompt_tokens']
                    output_tokens = response_body['usage']['completion_tokens']
                    total_tokens = response_body['usage']['total_tokens']
                    cost = calculate_cost(input_tokens, output_tokens)
                    
                    all_data.append({
                        'original_code': original_code,
                        'update_snippet': update_snippet,
                        'final_code': final_code
                    })
                    
                    # Keep track of statistics separately
                    all_data[-1]['_input_tokens'] = input_tokens
                    all_data[-1]['_output_tokens'] = output_tokens
                    all_data[-1]['_total_tokens'] = total_tokens
                    all_data[-1]['_cost'] = cost
                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    print(f"Error processing line in {input_file} or {output_file}: {str(e)}")
                    continue
    
    if not all_data:
        raise ValueError("No valid data was processed")
    
    return pd.DataFrame(all_data)

def save_data(df, output_file):
    # Save as parquet
    df.to_parquet(output_file, index=False)
    print(f"Data saved to {output_file}")

    # Save as JSON
    json_file = os.path.splitext(output_file)[0] + '.json'
    with open(json_file, 'w') as f:
        json.dump(df.to_dict(orient='records'), f, ensure_ascii=False, indent=2)
    print(f"Data also saved to {json_file}")

def main():
    parser = argparse.ArgumentParser(description="Process batch files and save results.")
    parser.add_argument("-i", "--input", required=True, help="Input directory containing batch files")
    parser.add_argument("-o", "--output", required=True, help="Output file name (without extension)")
    args = parser.parse_args()

    df = load_and_pair_batches(args.input)
    
    print(f"Processed {len(df)} entries")
    print(df[['original_code', 'update_snippet', 'final_code']].head())
    
    # Calculate statistics
    total_input_tokens = df['_input_tokens'].sum()
    total_output_tokens = df['_output_tokens'].sum()
    total_cost = df['_cost'].sum()
    
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")
    print(f"Total cost: ${total_cost:.2f}")
    
    # Remove statistics columns before saving
    output_df = df[['original_code', 'update_snippet', 'final_code']]
    
    output_file = args.output if args.output.endswith('.parquet') else args.output + '.parquet'
    save_data(output_df, output_file)

if __name__ == "__main__":
    main()
