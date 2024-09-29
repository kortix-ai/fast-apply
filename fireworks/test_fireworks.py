import os
import time
import tiktoken
from fireworks.client import Fireworks

def count_tokens(text):
    """Count the number of tokens in the given text using tiktoken."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # Using this as a default
    return len(encoding.encode(text))

client = Fireworks(api_key=os.getenv("FIREWORKS_API_KEY"))

start_time = time.time()
response = client.chat.completions.create(
    model="accounts/fireworks/models/llama-v3p1-8b-instruct",
    messages=[{
        "role": "user",
        "content": "Tell me a story",
    }],
    stream=True,
)

generated_text = ""
for chunk in response:
    if chunk.choices[0].delta.content:
        content = chunk.choices[0].delta.content
        print(content, end="", flush=True)
        generated_text += content

end_time = time.time()
elapsed_time = end_time - start_time

total_tokens = count_tokens(generated_text)
throughput = total_tokens / elapsed_time

print(f"\n\nThroughput: {throughput:.2f} tokens/second")

