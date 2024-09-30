import asyncio
import time
import tiktoken
import json
import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm, tqdm_asyncio
from inference_prompt import template
import argparse

def init_openai_client(pod, api_key):
    """Initialize and return the AsyncOpenAI client."""
    url = f"https://api.runpod.ai/v2/{pod}/openai/v1"
    return AsyncOpenAI(base_url=url, api_key=api_key)

def count_tokens(text):
    """Count the number of tokens in the given text using tiktoken."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))

async def execute_query(client, model, text, max_tokens):
    """Execute a query and return the results."""
    #  print(f"Executing query for text: {text[:50]}...")  # Add this line for logging
    start_time = time.time()
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{'role': 'user', 'content': text}],
            max_tokens=max_tokens
        )

        generated_text = response.choices[0].message.content
        elapsed_time = time.time() - start_time
        total_tokens = response.usage.total_tokens
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

async def process_row(client, model, row, max_tokens, model_name):
    """Process a single row of the testset."""
    original_code = row.original_code
    update_snippet = row.update_snippet
    text = template.format(original_code=original_code, update_snippet=update_snippet)
    result = await execute_query(client, model, text, max_tokens)
    if result:
        result['model'] = model_name
        result['input'] = text
        result['original_code'] = original_code
        result['update_snippet'] = update_snippet
        result['final_code'] = row.final_code
        result['full_output'] = text + result['generated_text']
    return result

async def process_testset(file_path, pod, api_key, model_name, max_tokens, num_queries):
    """Process the testset and generate output asynchronously."""
    client = init_openai_client(pod, api_key)
    response = await client.models.list()
    if not response.data:
        raise Exception("No models available")
    model = response.data[0].id

    df = load_testset(file_path)
    if num_queries and num_queries < len(df):
        df = df.sample(n=num_queries, random_state=42)  # Use a fixed random state for reproducibility
    tasks = [asyncio.create_task(process_row(client, model, row, max_tokens, model_name)) for row in df.itertuples(index=False)]
    
    results = await tqdm_asyncio.gather(*tasks, desc="Processing testset")
    return [result for result in results if result is not None]

def save_results(results, output_file):
    """Save results to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

async def main():
    parser = argparse.ArgumentParser(description="Generate output for a testset using serverless vLLM.")
    parser.add_argument("input_file", help="Path to the input Parquet or JSON file")
    parser.add_argument("--pod", default="vllm-s3jk7plkef1ov8", help="RunPod ID")
    parser.add_argument("--api_key", default="PAP4OD5L12KDG6NHBL6UNX3TOOK78C1GH9MKM9UZ", help="API Key")
    parser.add_argument("--model_name", help="Model name")
    parser.add_argument("--max_tokens", type=int, default=4192, help="Maximum number of tokens for generation")
    parser.add_argument("--num_queries", type=int, help="Number of queries to run (if less than max examples)")
    args = parser.parse_args()

    results = await process_testset(args.input_file, args.pod, args.api_key, args.model_name, args.max_tokens, args.num_queries)
    output_file = f"data/testset_results_{args.model_name}.json"
    save_results(results, output_file)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())