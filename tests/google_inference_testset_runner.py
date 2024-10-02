import asyncio
import time
import tiktoken
import json
import pandas as pd
import google.generativeai as genai
from tqdm.asyncio import tqdm_asyncio
from inference_prompt import template
import argparse
import os

def init_google_client(api_key):
    """Initialize and return the Google API client."""
    os.environ["GOOGLE_API_KEY"] = api_key
    genai.configure(api_key=api_key)
    return genai

def count_tokens(text):
    """Count the number of tokens in the given text using tiktoken."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))

async def execute_query(client, model_name, text, max_tokens):
    """Execute a query and return the results."""
    start_time = time.time()
    try:
        model = client.GenerativeModel(model_name)
        response = await asyncio.to_thread(model.generate_content, text)

        generated_text = response.text
        elapsed_time = time.time() - start_time
        total_tokens = count_tokens(generated_text)
        throughput = total_tokens / elapsed_time

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

async def process_row(client, model_name, row, max_tokens):
    """Process a single row of the testset."""
    original_code = row.original_code
    update_snippet = row.update_snippet
    text = template.format(original_code=original_code, update_snippet=update_snippet)
    result = await execute_query(client, model_name, text, max_tokens)
    if result:
        result['model'] = model_name
        result['input'] = text
        result['original_code'] = original_code
        result['update_snippet'] = update_snippet
        result['final_code'] = row.final_code
        result['full_output'] = text + result['generated_text']
    return result

async def process_testset(file_path, api_key, model_name, max_tokens, num_queries):
    """Process the testset and generate output asynchronously."""
    client = init_google_client(api_key)

    df = load_testset(file_path)
    if num_queries and num_queries < len(df):
        df = df.sample(n=num_queries, random_state=42)  # Use a fixed random state for reproducibility
    tasks = [asyncio.create_task(process_row(client, model_name, row, max_tokens)) for row in df.itertuples(index=False)]
    
    results = await tqdm_asyncio.gather(*tasks, desc="Processing testset")
    return [result for result in results if result is not None]

def save_results(results, output_file):
    """Save results to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

async def main():
    parser = argparse.ArgumentParser(description="Generate output for a testset using Google's Generative AI.")
    parser.add_argument("input_file", help="Path to the input Parquet or JSON file")
    parser.add_argument("--api_key", required=True, help="Google API Key")
    parser.add_argument("--model_name", default="gemini-pro", help="Model name")
    parser.add_argument("--max_tokens", type=int, default=8192, help="Maximum number of tokens for generation")
    parser.add_argument("--num_queries", type=int, help="Number of queries to run (if less than max examples)")
    args = parser.parse_args()

    results = await process_testset(args.input_file, args.api_key, args.model_name, args.max_tokens, args.num_queries)
    output_file = f"data/testset_results_{args.model_name}.json"
    save_results(results, output_file)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())
