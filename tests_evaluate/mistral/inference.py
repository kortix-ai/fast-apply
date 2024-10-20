import time
import tiktoken
import json
import pandas as pd
from mistralai import Mistral
from tqdm import tqdm
from tests_evaluate.common.inference_prompt import template
import argparse
import os

def init_mistral_client(api_key):
    """Initialize and return the Mistral client."""
    return Mistral(api_key=api_key)

def count_tokens(text):
    """Count the number of tokens in the given text using tiktoken."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # Using this as a proxy for Mistral
    return len(encoding.encode(text))

def execute_query(client, model, text, max_tokens):
    """Execute a query and return the results."""
    start_time = time.time()
    try:
        response = client.chat.complete(
            model=model,
            messages=[{"role": "user", "content": text}],
            max_tokens=max_tokens
        )

        generated_text = response.choices[0].message.content
        elapsed_time = time.time() - start_time
        total_tokens = count_tokens(text) + count_tokens(generated_text)
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

def process_row(client, model, row, max_tokens, model_name):
    """Process a single row of the testset."""
    original_code = row.original_code
    update_snippet = row.update_snippet
    text = template.format(original_code=original_code, update_snippet=update_snippet)
    result = execute_query(client, model, text, max_tokens)
    if result:
        result['model'] = model_name
        result['input'] = text
        result['original_code'] = original_code
        result['update_snippet'] = update_snippet
        result['final_code'] = row.final_code
        result['full_output'] = text + result['generated_text']
    return result

def process_testset(file_path, api_key, model_name, max_tokens, num_queries):
    """Process the testset and generate output."""
    client = init_mistral_client(api_key)

    df = load_testset(file_path)
    if num_queries and num_queries < len(df):
        df = df.sample(n=num_queries, random_state=42)  # Use a fixed random state for reproducibility
    
    results = []
    for row in tqdm(df.itertuples(index=False), desc="Processing testset", total=len(df)):
        result = process_row(client, model_name, row, max_tokens, model_name)
        if result:
            results.append(result)
    
    return results

def save_results(results, output_file):
    """Save results to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Generate output for a testset using Mistral AI.")
    parser.add_argument("input_file", help="Path to the input Parquet or JSON file")
    parser.add_argument("--model_name", default="mistral-large-latest", help="Model name")
    parser.add_argument("--max_tokens", type=int, default=4192, help="Maximum number of tokens for generation")
    parser.add_argument("--num_queries", type=int, help="Number of queries to run (if less than max examples)")
    args = parser.parse_args()

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY environment variable is not set")

    results = process_testset(args.input_file, api_key, args.model_name, args.max_tokens, args.num_queries)
    output_file = f"data/testset_results_{args.model_name}.json"
    save_results(results, output_file)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
