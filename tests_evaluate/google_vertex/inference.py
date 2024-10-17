import asyncio
import time
import json
import pandas as pd
import argparse
from tqdm.asyncio import tqdm_asyncio
import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting
from tests_evaluate.common.inference_prompt import template

# Constants
PROJECT_ID = "530422023205"
LOCATION = "us-central1"
MODEL_ID = "projects/530422023205/locations/us-central1/endpoints/2237885494035742720"
MAX_TOKENS = 8192

def init_vertex_ai():
    """Initialize Vertex AI client."""
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    return GenerativeModel(MODEL_ID)

def count_tokens(text):
    """Count the number of tokens in the given text."""
    # Note: This is a placeholder. Vertex AI might have a different way to count tokens.
    return len(text.split())

def retry_with_exponential_backoff(func, *args, **kwargs):
    """Retry a function with exponential backoff."""
    max_retries = 5
    retry_delay = 70  # Initial retry delay in seconds

    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if "429 Quota exceeded" in str(e) and attempt < max_retries - 1:
                print(f"Quota exceeded. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                raise e

async def execute_query(model, text, max_tokens):
    """Execute a query and return the results."""
    start_time = time.time()
    
    chat = model.start_chat()
    
    generation_config = {
        "max_output_tokens": max_tokens,
        "temperature": 0,
        "top_p": 0.95,
    }

    safety_settings = [
        SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=SafetySetting.HarmBlockThreshold.OFF),
        SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=SafetySetting.HarmBlockThreshold.OFF),
        SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=SafetySetting.HarmBlockThreshold.OFF),
        SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=SafetySetting.HarmBlockThreshold.OFF),
    ]

    try:
        def send_message():
            return chat.send_message(
                [text],
                generation_config=generation_config,
                safety_settings=safety_settings
            )

        response = retry_with_exponential_backoff(send_message)
        
        generated_text = response.text
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

async def process_row(model, row, max_tokens):
    """Process a single row of the testset."""
    original_code = row.original_code
    update_snippet = row.update_snippet
    text = template.format(original_code=original_code, update_snippet=update_snippet)
    result = await execute_query(model, text, max_tokens)
    if result:
        result['model'] = MODEL_ID
        result['input'] = text
        result['original_code'] = original_code
        result['update_snippet'] = update_snippet
        result['final_code'] = row.final_code
        result['full_output'] = text + result['generated_text']
    return result

async def process_testset(file_path, model, max_tokens, num_queries):
    """Process the testset and generate output asynchronously."""
    df = load_testset(file_path)
    if num_queries and num_queries < len(df):
        df = df.sample(n=num_queries, random_state=42)  # Use a fixed random state for reproducibility
    tasks = [asyncio.create_task(process_row(model, row, max_tokens)) for row in df.itertuples(index=False)]
    
    results = await tqdm_asyncio.gather(*tasks, desc="Processing testset")
    return [result for result in results if result is not None]

def save_results(results, output_file):
    """Save results to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

async def main():
    parser = argparse.ArgumentParser(description="Generate output for a testset using Google Vertex AI.")
    parser.add_argument("input_file", help="Path to the input Parquet or JSON file")
    parser.add_argument("--max_tokens", type=int, default=MAX_TOKENS, help="Maximum number of tokens for generation")
    parser.add_argument("--num_queries", type=int, help="Number of queries to run (if less than max examples)")
    args = parser.parse_args()

    model = init_vertex_ai()
    results = await process_testset(args.input_file, model, args.max_tokens, args.num_queries)
    output_file = f"data/testset_results_vertex_ai.json"
    save_results(results, output_file)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())
