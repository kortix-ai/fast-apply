import time
import tiktoken
import argparse
import os
import requests
import json
from tests_evaluate.single_test_prompt import original_code, update_snippet
from tests_evaluate.inference_prompt import template

# Constants
API_KEY = "fw_3ZhfovPCeNKHpHcEnS9D9HmX"
MAX_TOKENS = 8192
DEFAULT_PATTERN = "accounts/marko-1d84ff/models/"
DEFAULT_MODEL = "8b-v12"
URL = "https://api.fireworks.ai/inference/v1/completions"

def count_tokens(text):
    """Count the number of tokens in the given text using tiktoken."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))

def execute_query(model_name, text, stream_output=False):
    """Execute a query and return the results."""
    start_time = time.time()
    
    payload = {
        "model": model_name,
        "prompt": text,
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "top_p": 1,
        "top_k": 0,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "n": 1,
        "stream": True,
        "context_length_exceeded_behavior": "truncate",
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    generated_text = ""
    with requests.post(URL, json=payload, headers=headers, stream=True) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    json_str = line[6:]  # Remove 'data: ' prefix
                    if json_str.strip() == '[DONE]':
                        break
                    try:
                        json_data = json.loads(json_str)
                        text = json_data['choices'][0]['text']
                        generated_text += text
                        if stream_output:
                            print(text, end='', flush=True)
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON: {json_str}")
    
    if stream_output:
        print()  # Print a newline at the end
    
    elapsed_time = time.time() - start_time
    total_tokens = count_tokens(generated_text)
    throughput = total_tokens / elapsed_time
    
    return {
        "throughput": throughput,
        "generated_text": generated_text
    }

def main():
    """Execute queries and print their results."""
    parser = argparse.ArgumentParser(description="Run Fireworks API test with a specified model.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="The model identifier to use for the test.")
    args = parser.parse_args()
    
    try:
        model_name = DEFAULT_PATTERN + args.model
        
        text = template.format(original_code=original_code, update_snippet=update_snippet)
        
        print(f"Running test with model: {model_name}")
        print("Test Query (Streaming):")
        results = execute_query(model_name, text, stream_output=True)
        print(f"\n\nTest Query Throughput: {results['throughput']:.2f} tokens/second")
        
        for i in range(1, 5):
            print(f"\nQuery {i}:")
            results = execute_query(model_name, text)
            print(f"Throughput: {results['throughput']:.2f} tokens/second")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
