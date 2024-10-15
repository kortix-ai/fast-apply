import time
import tiktoken
import argparse
import os
import json
import pandas as pd
import google.generativeai as genai
from tqdm import tqdm

# Constants
API_KEY = "AIzaSyDWRqdg15wX03c8V358ipORcaQJqvgqlLo"
# MAX_TOKENS = 
DEFAULT_NUM_TESTS = 1

def init_google_client(api_key):
    """Initialize and return the Google API client."""
    os.environ["GEMINI_KEY"] = api_key
    genai.configure(api_key=api_key)
    return genai

def count_tokens(text):
    """Count the number of tokens in the given text using tiktoken."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))

def execute_query(client, model_name, text, stream_output=False):
    """Execute a query and return the results."""
    start_time = time.time()
    
    generation_config = {
        "max_output_tokens": 8192,
        "temperature": 0,
        "response_mime_type": "text/plain",
    }
    
    model = client.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config,
    )
    
    chat_session = model.start_chat(history=[])
    
    response = chat_session.send_message(text)
    
    generated_text = response.text
    if stream_output:
        print(generated_text)
    
    elapsed_time = time.time() - start_time
    input_tokens = count_tokens(text)
    output_tokens = count_tokens(generated_text)
    total_tokens = input_tokens + output_tokens
    throughput = total_tokens / elapsed_time
    
    return {
        "throughput": throughput,
        "elapsed_time": elapsed_time,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "generated_text": generated_text,
        "response": response
    }

def load_testset(file_path):
    """Load testset from Parquet or JSON file."""
    if file_path.endswith('.parquet'):
        return pd.read_parquet(file_path)
    elif file_path.endswith('.json'):
        return pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file format. Please use .parquet or .json")

def process_row(client, model_name, row, print_prompt=False, use_simple_template=False):
    """Process a single row of the testset."""
    from tests_evaluate.common.inference_prompt import template, simple_template
    text = simple_template if use_simple_template else template
    text = text.format(original_code=row.original_code, update_snippet=row.update_snippet)
    if print_prompt:
        print("Full prompt:")
        print(text)
        print("\nSending query...\n")
    result = execute_query(client, model_name, text)
    result['model'] = model_name
    result['input'] = text
    result['original_code'] = row.original_code
    result['update_snippet'] = row.update_snippet
    result['final_code'] = row.final_code
    result['full_output'] = text + result['generated_text']
    return result

def process_testset(file_path, model_name, print_prompt=False, use_simple_template=False):
    """Process the testset and generate output."""
    client = init_google_client(API_KEY)
    df = load_testset(file_path)
    results = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Processing testset"):
        try:
            result = process_row(client, model_name, row, print_prompt)
            # Convert non-serializable objects to strings
            result['response'] = str(result['response'])
            results.append(result)
        except Exception as e:
            print(f"Error processing row: {e}")
            results.append({
                "error": str(e),
                "model": model_name,
                "original_code": row.original_code,
                "update_snippet": row.update_snippet,
                "final_code": row.final_code
            })
    return results

def save_results(results, output_file):
    """Save results to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

def read_file(file_path):
    """Read and return the contents of a file."""
    with open(file_path, 'r') as file:
        return file.read()

def print_results(results):
    """Print all related information of the response."""
    print("Response Information:")
    print(f"  Throughput: {results['throughput']:.2f} tokens/second")
    print(f"  Elapsed Time: {results['elapsed_time']:.2f} seconds")
    print(f"  Input Tokens: {results['input_tokens']}")
    print(f"  Output Tokens: {results['output_tokens']}")
    print(f"  Total Tokens: {results['total_tokens']}")
    print("  Generated Text:")
    print(f"    {results['generated_text']}")
    print("  Full Response Object:")
    print_nested_dict(results['response'])
    print(f"  Generated Text Characters: {len(results['generated_text'])}")

def print_nested_dict(obj, indent=4):
    """Print a nested dictionary with proper indentation."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            print(" " * indent + str(key) + ":")
            print_nested_dict(value, indent + 4)
    elif isinstance(obj, list):
        for item in obj:
            print_nested_dict(item, indent + 4)
    else:
        print(" " * indent + str(obj))

from tests_evaluate.common.inference_prompt import template, simple_template

def main():
    """Execute queries and save results."""
    parser = argparse.ArgumentParser(description="Run Google API test with a specified model on a testset.")
    parser.add_argument("input_file", nargs='?', help="Path to the input Parquet or JSON file")
    parser.add_argument("--model", required=True, help="The model identifier to use for the test.")
    parser.add_argument("--print-prompt", action="store_true", help="Print the full prompt before sending the query")
    parser.add_argument("-n", "--additional-tests", type=int, default=DEFAULT_NUM_TESTS, help="Number of additional tests to run (default: 1)")
    parser.add_argument("--prompt-template", default="tests_evaluate/common/inference_prompt.py", help="File path for the prompt template.")
    parser.add_argument("--single-test-prompt", default="tests_evaluate/common/single_test_prompt.py", help="File path for the single test prompt.")
    parser.add_argument("--simple", action="store_true", help="Use simple template for prompt input")
    args = parser.parse_args()
    
    model_name = args.model
    print(f"Running tests with model: {model_name}")
    
    if args.input_file:
        # Process testset
        results = []
        try:
            results = process_testset(args.input_file, model_name, args.print_prompt, args.simple)
            
            # Run additional tests if specified
            for i in range(1, args.additional_tests + 1):
                print(f"\nRunning additional test {i}:")
                additional_results = process_testset(args.input_file, model_name, args.print_prompt, args.simple)
                results.extend(additional_results)
        except Exception as e:
            print(f"An error occurred during processing: {e}")
        finally:
            if results:
                output_file = f"data/testset_results_{args.model.replace('/', '_')}.json"
                save_results(results, output_file)
                print(f"Results saved to {output_file}")
            else:
                print("No results were generated.")
    else:
        # Run single test
        try:
            client = init_google_client(API_KEY)
            
            template = read_file(args.prompt_template)
            single_test_prompt = read_file(args.single_test_prompt)
            
            # Create a new namespace for executing the single test prompt
            namespace = {}
            exec(single_test_prompt, namespace)
            
            # Extract original_code and update_snippet from the namespace
            original_code = namespace.get('original_code', '')
            update_snippet = namespace.get('update_snippet', '')
            
            text = simple_template if args.simple else template
            text = text.format(original_code=original_code, update_snippet=update_snippet)

            if args.print_prompt:
                print("Full prompt:")
                print(text)
                print("\nSending query...\n")
            
            print("Test Query (Streaming):")
            results = execute_query(client, model_name, text, stream_output=True)
            print_results(results)
            
            for i in range(1, args.additional_tests + 1):
                print(f"\nQuery {i}:")
                results = execute_query(client, model_name, text)
                print_results(results)
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
