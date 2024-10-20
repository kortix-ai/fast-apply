import time
import tiktoken
import argparse
from mistralai import Mistral
import os

# Constants
MAX_TOKENS = 8192

def init_mistral_client(api_key):
    """Initialize and return the Mistral client."""
    return Mistral(api_key=api_key)

def count_tokens(text):
    """Count the number of tokens in the given text using tiktoken."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # Using this as a proxy for Mistral
    return len(encoding.encode(text))

def execute_query(client, model, text, stream_output=False):
    """Execute a query and return the results."""
    start_time = time.time()
    response = client.chat.complete(
        model=model,
        messages=[{"role": "user", "content": text}],
        max_tokens=MAX_TOKENS
    )

    generated_text = response.choices[0].message.content
    if stream_output:
        print(generated_text)

    elapsed_time = time.time() - start_time
    total_tokens = count_tokens(generated_text)
    throughput = total_tokens / elapsed_time

    return {
        "throughput": throughput,
        "generated_text": generated_text
    }

def main():
    """Execute queries and print their results."""
    parser = argparse.ArgumentParser(description="Run Mistral AI test.")
    parser.add_argument("--model", default="ministral-3b-latest", help="The model to use for the test.")
    args = parser.parse_args()

    try:
        from tests_evaluate.common.inference_prompt import template
        from tests_evaluate.common.single_test_prompt import original_code, update_snippet

        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable is not set")

        client = init_mistral_client(api_key)

        text = template.format(original_code=original_code, update_snippet=update_snippet)

        print(f"Running test with model: {args.model}")
        print("Test Query (Streaming):")
        results = execute_query(client, args.model, text, stream_output=True)
        print(f"\n\nTest Query Throughput: {results['throughput']:.2f} tokens/second")

        for i in range(1, 3):
            print(f"\nQuery {i}:")
            results = execute_query(client, args.model, text)
            print(f"Throughput: {results['throughput']:.2f} tokens/second")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
