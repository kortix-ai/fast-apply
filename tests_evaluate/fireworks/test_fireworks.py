import time
import tiktoken
import argparse
from fireworks.client import Fireworks
from tests_evaluate.common.single_test_prompt import original_code, update_snippet


SYSTEM_PROMPT = """You are an coding assistant that helps merge code updates, ensuring every modification is fully integrated."""

USER_PROMPT = """Merge all changes from the <update> snippet into the <code> below.
- Preserve the code's structure, order, comments, and indentation exactly.
- Output only the updated code, enclosed within <updated-code> and </updated-code> tags.
- Do not include any additional text, explanations, placeholders, ellipses, or code fences.

<code>{original_code}</code>

<update>{update_snippet}</update>

Provide the complete updated code.
"""


# Constants
API_KEY = "fw_3ZhfovPCeNKHpHcEnS9D9HmX"
MAX_TOKENS = 8192
DEFAULT_PATTERN = "accounts/marko-1d84ff/models/"
DEFAULT_MODEL = "fast-apply-v16-1p5b-instruct"
# DEFAULT_MODEL = "8b-v12"
# DEFAULT_MODEL = "fast-apply-v16-7b-instruct"
# DEFAULT_MODEL  =  "accounts/fireworks/models/qwen2p5-32b-instruct"

def count_tokens(text):
    """Count the number of tokens in the given text using tiktoken."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))

def execute_query(client, model_name, original_code, update_snippet, stream_output=False):
    """Execute a query and return the results."""
    start_time = time.time()
    
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": USER_PROMPT.format(original_code=original_code, update_snippet=update_snippet),
        }
    ]
    # print(SYSTEM_PROMPT + USER_PROMPT.format(original_code=original_code, update_snippet=update_snippet))

    
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=0,
        stream=True,
    )
    
    generated_text = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            text = chunk.choices[0].delta.content
            generated_text += text
            if stream_output:
                print(text, end='', flush=True)
    
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
        # model_name = 'accounts/fireworks/models/qwen2p5-32b-instruct'
        # model_name = "qwen2p5-32b-instruct-e8fb1bf2"
        client = Fireworks(api_key=API_KEY)
        
        print(f"Running test with model: {model_name}")
        print("Test Query (Streaming):")
        results = execute_query(client, model_name, original_code, update_snippet, stream_output=True)
        print(f"\n\nTest Query Throughput: {results['throughput']:.2f} tokens/second")
        
        for i in range(1, 5):
            print(f"\nQuery {i}:")
            results = execute_query(client, model_name, original_code, update_snippet)
            print(f"Throughput: {results['throughput']:.2f} tokens/second")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
