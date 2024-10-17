import base64
import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting, Part
import argparse
import time
from tests_evaluate.common.single_test_prompt import original_code, update_snippet
from tests_evaluate.common.inference_prompt import template

# Constants
PROJECT_ID = "530422023205"
LOCATION = "us-central1"
MODEL_ID = "projects/530422023205/locations/us-central1/endpoints/2237885494035742720"

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

def execute_query(model, text, stream_output=False):
    """Execute a query and return the results."""
    chat = model.start_chat()
    
    generation_config = {
        "max_output_tokens": 8192,
        "temperature": 0,
        "top_p": 0.95,
    }

    safety_settings = [
        SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=SafetySetting.HarmBlockThreshold.OFF),
        SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=SafetySetting.HarmBlockThreshold.OFF),
        SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=SafetySetting.HarmBlockThreshold.OFF),
        SafetySetting(category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=SafetySetting.HarmBlockThreshold.OFF),
    ]

    start_time = time.time()
    
    def send_message():
        return chat.send_message(
            [text],
            generation_config=generation_config,
            safety_settings=safety_settings
        )

    response = retry_with_exponential_backoff(send_message)
    end_time = time.time()

    generated_text = response.text
    
    if stream_output:
        print(generated_text)

    total_tokens = count_tokens(text) + count_tokens(generated_text)
    elapsed_time = end_time - start_time
    
    return {
        "throughput": total_tokens / elapsed_time,
        "generated_text": generated_text
    }

def main():
    """Execute queries and print their results."""
    parser = argparse.ArgumentParser(description="Run Google Vertex AI test.")
    args = parser.parse_args()
    
    try:
        model = init_vertex_ai()
        
        text = template.format(original_code=original_code, update_snippet=update_snippet)
        
        print("Test Query (Streaming):")
        results = execute_query(model, text, stream_output=True)
        print(f"\n\nTest Query Throughput: {results['throughput']:.2f} tokens/second")
        
        for i in range(1, 5):
            print(f"\nQuery {i}:")
            results = execute_query(model, text)
            print(f"Throughput: {results['throughput']:.2f} tokens/second")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
