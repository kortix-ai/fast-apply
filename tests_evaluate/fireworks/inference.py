import asyncio
import time
import tiktoken
import json
import pandas as pd
import requests
import argparse
from tqdm.asyncio import tqdm_asyncio
from tests_evaluate.common.inference_prompt import template
import os

# Constants
API_KEY = os.getenv("FIREWORKS_API_KEY")
MAX_TOKENS = 8192
MODEL_PREFIX = "accounts/marko-1d84ff/models/"
DEFAULT_MODEL = "1b-v12"
URL = "https://api.fireworks.ai/inference/v1/completions"

def count_tokens(text):
    """Count the number of tokens in the given text using tiktoken."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))

async def execute_query(model_name, text, max_tokens):
    """Execute a query and return the results."""
    start_time = time.time()
    
    payload = {
        "model": model_name,
        "prompt": text,
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "top_k": 0,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "n": 1,
        "stream": False,
        "context_length_exceeded_behavior": "truncate",
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(URL, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        
        generated_text = result['choices'][0]['text']
        elapsed_time = time.time() - start_time
        total_tokens = count_tokens(text) + count_tokens(generated_text)
        throughput = count_tokens(generated_text) / elapsed_time

        return {
            "generated_text": generated_text,
            "throughput": throughput,
            "total_tokens": total_tokens,
            "elapsed_time": elapsed_time
        }
    except Exception as e:
        print(f"Error executing query: {str(e)}")
        return None

def load_testset(file_path):
    """Load testset from Parquet or JSON file."""
    if file_path.endswith('.parquet'):
        return pd.read_parquet(file_path)
    elif file_path.endswith('.json'):
        return pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file format. Please use .parquet or .json")

async def process_row(model_name, row, max_tokens):
    """Process a single row of the testset."""
    original_code = row.original_code
    update_snippet = row.update_snippet
    text = template.format(original_code=original_code, update_snippet=update_snippet)
    result = await execute_query(model_name, text, max_tokens)
    if result:
        result['model'] = model_name
        result['input'] = text
        result['original_code'] = original_code
        result['update_snippet'] = update_snippet
        result['final_code'] = row.final_code
        result['full_output'] = text + result['generated_text']
    return result

async def process_testset(file_path, model_name, max_tokens, num_queries):
    """Process the testset and generate output asynchronously."""
    df = load_testset(file_path)
    if num_queries and num_queries < len(df):
        df = df.sample(n=num_queries, random_state=42)  # Use a fixed random state for reproducibility
    # tasks = [process_row(model_name, row, max_tokens) for row in df.itertuples(index=False)]
    tasks = [ asyncio.create_task(process_row(model_name, row, max_tokens)) for row in df.itertuples(index=False)]
    
    results = await tqdm_asyncio.gather(*tasks, desc="Processing testset")
    return [result for result in results if result is not None]

def save_results(results, output_file):
    """Save results to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

async def main():
    parser = argparse.ArgumentParser(description="Generate output for a testset using Fireworks API.")
    parser.add_argument("input_file", help="Path to the input Parquet or JSON file")
    parser.add_argument("-m", "--model_name", default=DEFAULT_MODEL, help="Model name (without prefix)")
    parser.add_argument("--max_tokens", type=int, default=MAX_TOKENS, help="Maximum number of tokens for generation")
    parser.add_argument("--num_queries", type=int, help="Number of queries to run (if less than max examples)")
    args = parser.parse_args()

    full_model_name = MODEL_PREFIX + args.model_name
    results = await process_testset(args.input_file, full_model_name, args.max_tokens, args.num_queries)
    output_file = f"data/testset_results_{args.model_name}.json"
    save_results(results, output_file)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())
