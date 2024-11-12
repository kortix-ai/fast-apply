import argparse
import os
import time

import tiktoken
from openai import OpenAI

# Constants
API_KEY = os.getenv("OPENROUTER_API_KEY")
MAX_TOKENS = 4000


def init_openrouter_client():
    """Initialize and return the OpenRouter client."""
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)


def count_tokens(text):
    """Count the number of tokens in the given text using tiktoken. WARNING: This is not accurate for OpenRouter."""
    print(
        "\nWARNING: Number of tokens are estimated using tiktoken, not accurate for OpenRouter.",
    )
    encoding = tiktoken.encoding_for_model("gpt-4")
    return len(encoding.encode(text))


def execute_query(
    client, model, text, original_code=None, use_prediction=True, stream_output=False
):
    """Execute a query and return the results."""
    start_time = time.time()

    # Prepare API call parameters
    api_params = {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "stream": stream_output,  # Now correctly set based on stream_output
    }

    # Add prediction if enabled and original code is provided
    if use_prediction and original_code:
        api_params["prediction"] = {
            "type": "content",
            "content": f"<updated-code>\n{original_code}\n</updated-code>",
        }

    generated_text = ""
    if stream_output:
        # Handle streaming response
        stream = client.chat.completions.create(**api_params)
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                generated_text += content
    else:
        # Handle non-streaming response
        response = client.chat.completions.create(**api_params)
        generated_text = response.choices[0].message.content

    elapsed_time = time.time() - start_time
    total_tokens = count_tokens(generated_text)
    throughput = total_tokens / elapsed_time

    return {
        "throughput": throughput,
        "total_tokens": total_tokens,
        "elapsed_time": elapsed_time,
    }


def main():
    """Execute queries and print their results."""
    parser = argparse.ArgumentParser(
        description="Test OpenRouter models with streaming output."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/llama-3.1-70b-instruct:free",
        help="Model to use",
    )
    parser.add_argument(
        "--no-prediction", action="store_true", help="Disable prediction parameter"
    )
    parser.add_argument(
        "--no-stream", action="store_true", help="Disable streaming output"
    )
    args = parser.parse_args()

    try:
        from tests_evaluate.common.inference_prompt import template
        from tests_evaluate.common.single_test_prompt import (
            original_code,
            update_snippet,
        )

        model = args.model
        client = init_openrouter_client()
        text = template.format(
            original_code=original_code, update_snippet=update_snippet
        )

        print(f"Running test with model: {model}")
        print("Prediction mode:", "Disabled" if args.no_prediction else "Enabled")
        print("\nTest Query (Streaming):")
        results = execute_query(
            client,
            model,
            text,
            original_code=original_code if not args.no_prediction else None,
            use_prediction=not args.no_prediction,
            stream_output=not args.no_stream,
        )
        print(f"\n\nTest Query Results:")
        print(f"Throughput: {results['throughput']:.2f} tokens/second")
        print(f"Total Tokens: {results['total_tokens']}")
        print(f"Elapsed Time: {results['elapsed_time']:.2f} seconds")

        print("\nRunning additional queries for throughput testing...")
        throughputs = []
        for i in range(1, 3):
            print(f"\nQuery {i}:")
            results = execute_query(
                client,
                model,
                text,
                original_code=original_code if not args.no_prediction else None,
                use_prediction=not args.no_prediction,
                stream_output=False,  # Disable streaming for throughput tests
            )
            throughputs.append(results["throughput"])
            print(f"Throughput: {results['throughput']:.2f} tokens/second")
            print(f"Total Tokens: {results['total_tokens']}")
            print(f"Elapsed Time: {results['elapsed_time']:.2f} seconds")

        avg_throughput = sum(throughputs) / len(throughputs)
        print(f"\nAverage Throughput: {avg_throughput:.2f} tokens/second")

    except Exception as e:
        print(f"An error occurred: {e}")
        if "api_key" in str(e).lower():
            print("\nMake sure to set the OPENROUTER_API_KEY environment variable.")


if __name__ == "__main__":
    main()
