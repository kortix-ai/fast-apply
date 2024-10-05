import time
import tiktoken
import argparse
import os
import google.generativeai as genai
from tests.single_test_prompt import original_code, update_snippet
from tests.inference_prompt import template

# Constants
API_KEY = "AIzaSyD0HiVoAPUNzh3MWNHtuBZby4SWTqxnSvU"
MAX_TOKENS = 8192
DEFAULT_MODEL = "tunedModels/train-4gaullhp8hak"

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
        "max_output_tokens": MAX_TOKENS,
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
    total_tokens = count_tokens(generated_text)
    throughput = total_tokens / elapsed_time
    
    return {
        "throughput": throughput
    }

def main():
    """Execute queries and print their results."""
    parser = argparse.ArgumentParser(description="Run Google API test with a specified model.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="The model identifier to use for the test.")
    args = parser.parse_args()
    
    try:
        client = init_google_client(API_KEY)
        model_name = args.model
        
        text = template.format(original_code=original_code, update_snippet=update_snippet)
        
        print(f"Running test with model: {model_name}")
        print("Test Query (Streaming):")
        results = execute_query(client, model_name, text, stream_output=True)
        print(f"\n\nTest Query Throughput: {results['throughput']:.2f} tokens/second")
        
        for i in range(1, 5):
            print(f"\nQuery {i}:")
            results = execute_query(client, model_name, text)
            print(f"Throughput: {results['throughput']:.2f} tokens/second")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
