"""
This script prepares the batch data for OpenAI's Batch API by reading the input data,
generating prompts, and writing the requests to a JSONL file.
"""

import json
import pandas as pd
import argparse
from prompt import generate_prompt
import os
from pathlib import Path
import tiktoken
import math
from data_generation.utils import calculate_cost, load_data, count_tokens


def prepare_batch_requests(df: pd.DataFrame, prompt_func, model: str, batch_limit: int = 90000) -> tuple:
    """
    Prepares batch requests for OpenAI's Batch API, splitting into multiple batches if necessary.
    
    Args:
        df (pd.DataFrame): DataFrame containing the original code.
        prompt_func (function): Function to generate prompts.
        model (str): OpenAI model to use.
        batch_limit (int): Maximum input tokens per batch.
    
    Returns:
        tuple: A tuple containing:
            - list: List of tuples, each containing a list of request dictionaries and total token count for that batch.
            - int: Total input tokens across all batches.
            - int: Total max tokens across all batches.
    """
    all_batches = []
    current_batch = []
    current_batch_tokens = 0
    total_input_tokens = 0
    total_max_tokens = 0

    for index, row in df.iterrows():
        original_code = row.get('original_code')  # Adjust the column name as needed
        if pd.isna(original_code) or not original_code.strip():
            print(f"Warning: Missing or empty original_code at index {index}. Skipping.")
            continue
        
        messages = prompt_func(original_code)
        
        # Count tokens in the original code and messages
        message_tokens = sum(count_tokens(msg['content'], model) for msg in messages)
        request_tokens = message_tokens
        
        # Calculate max_tokens
        original_code_tokens = count_tokens(original_code, model)
        max_tokens = int(original_code_tokens * 1.8)
        
        request = {
            "custom_id": f"task-{index}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": messages,
                "temperature": 0,  
                "max_tokens": max_tokens
            }
        }

        if current_batch_tokens + request_tokens > batch_limit:
            all_batches.append((current_batch, current_batch_tokens))
            current_batch = []
            current_batch_tokens = 0

        current_batch.append(request)
        current_batch_tokens += request_tokens
        total_input_tokens += request_tokens
        total_max_tokens += max_tokens

    if current_batch:
        all_batches.append((current_batch, current_batch_tokens))

    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total max tokens: {total_max_tokens}")
    return all_batches, total_input_tokens, total_max_tokens


def write_jsonl(requests: list, output_dir: str, batch_number: int = 1):
    """
    Writes the list of requests to a .jsonl file.
    
    Args:
        requests (list): List of request dictionaries.
        output_dir (str): Path to the output directory.
        batch_number (int): Batch number to append to the filename.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / f"batch_{batch_number:03d}.jsonl"
    
    if output_file.exists():
        overwrite = input(f"File {output_file} already exists. Overwrite? (y/n): ").lower().strip()
        if overwrite != 'y':
            print(f"Skipping batch {batch_number}")
            return
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + '\n')
    print(f"Batch input file created at {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Prepare batch data for OpenAI's Batch API.")
    parser.add_argument("-i", "--input_file", type=str, required=True, help="Path to the input data file (.parquet or .csv)")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Path to the output directory for .jsonl files")
    parser.add_argument("--model", type=str, default="gpt-4o-2024-08-06", help="OpenAI model to use (default: gpt-4o)")
    parser.add_argument("-n", type=int, help="Number of examples to process (optional)")
    parser.add_argument("--batch_limit", type=int, default=90000, help="Maximum input tokens per batch (default: 90000)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} does not exist.")
        return
    
    df = load_data(args.input_file, args.n)
    print(f"Loaded {len(df)} records from {args.input_file}")
    
    all_batches, total_input_tokens, total_max_tokens = prepare_batch_requests(df, generate_prompt, args.model, args.batch_limit)
    
    if not all_batches:
        print("No valid requests to process. Exiting.")
        return
    
    for batch_number, (batch_requests, batch_tokens) in enumerate(all_batches, start=1):
        write_jsonl(batch_requests, args.output_dir, batch_number)
        print(f"Batch {batch_number}: {len(batch_requests)} requests, {batch_tokens} tokens")

    total_requests = sum(len(batch) for batch, _ in all_batches)
    print(f"Total requests: {total_requests}")
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total max tokens: {total_max_tokens}")
    
    estimated_cost = calculate_cost(total_input_tokens, total_max_tokens)
    print(f"Estimated cost: ${estimated_cost:.2f}")

if __name__ == "__main__":
    main()

