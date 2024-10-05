import time
import tiktoken
import argparse
from openai import OpenAI
from single_test_prompt import original_code, update_snippet
from inference_prompt import template

# Constants
API_KEY = "PAP4OD5L12KDG6NHBL6UNX3TOOK78C1GH9MKM9UZ"
MAX_TOKENS = 4192

# template="""Merge all changes from the <update_snippet> into the <original_code>, producing a fully updated version of the code. Only modify the parts of the original_code that are specified in the update_snippet; keep all other parts unchanged. Provide the result as <updated_code>.

# ### Original Code:
# <original_code>{original_code}</original_code>

# ### Update Snippet:
# <update_snippet>{update_snippet}</update_snippet>

# ### Updated Code:
# """


def init_openai_client(pod):
    """Initialize and return the OpenAI client."""
    url = f"https://api.runpod.ai/v2/{pod}/openai/v1"
    return OpenAI(base_url=url, api_key=API_KEY)

def count_tokens(text):
    """Count the number of tokens in the given text using tiktoken."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))

def execute_query(client, model, text, stream_output=False):
    """Execute a query and return the results."""
    start_time = time.time()
    stream = client.chat.completions.create(
        model=model,
        messages=[{'role': 'user', 'content': text}],
        max_tokens=MAX_TOKENS,
        stream=True
    )

    generated_text = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            if stream_output:
                print(content, end="")
            generated_text += content

    elapsed_time = time.time() - start_time
    total_tokens = count_tokens(generated_text)
    throughput = total_tokens / elapsed_time

    return {
        "throughput": throughput
    }

def main():
    """Execute queries and print their results."""
    parser = argparse.ArgumentParser(description="Run serverless vLLM test with a specified POD.")
    parser.add_argument("--pod", required=True, help="The POD identifier to use for the test.")
    args = parser.parse_args()

    try:
        client = init_openai_client(args.pod)
        response = client.models.list()
        if not response.data:
            raise Exception("No models available")
        model = response.data[0].id

        text = template.format(original_code=original_code, update_snippet=update_snippet)

        print(f"Running test with POD: {args.pod}")
        print("Test Query (Streaming):")
        results = execute_query(client, model, text, stream_output=True)
        print(f"\n\nTest Query Throughput: {results['throughput']:.2f} tokens/second")

        for i in range(1, 3):
            print(f"\nQuery {i}:")
            results = execute_query(client, model, text)
            print(f"Throughput: {results['throughput']:.2f} tokens/second")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

