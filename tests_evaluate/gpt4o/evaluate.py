import os
import json
import time
import asyncio
import pandas as pd
import tiktoken
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from aiolimiter import AsyncLimiter
from ..evaluate import parse_generated_text, calculate_diff, print_statistics
from ..common.inference_prompt import template, simple_template, SYSTEM_PROMPT, USER_PROMPT

# Set rate limiter for OpenAI API - 10000 TPM / 200 RPM for most paid tiers
rate_limiter = AsyncLimiter(180, 60)  # Conservative limit of 180 requests per minute

def count_tokens(text):
    """Count the number of tokens in the given text using tiktoken."""
    encoding = tiktoken.encoding_for_model("gpt-4")
    return len(encoding.encode(text))

def load_testset(file_path):
    """Load testset from Parquet or JSON file."""
    if file_path.endswith('.parquet'):
        return pd.read_parquet(file_path)
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError("Unsupported file format. Please use .parquet or .json")

async def evaluate_with_gpt4(data, limit=None, use_simple_template=False, use_system_user_prompt=False, use_mini=False, use_prediction=False):
    """
    Evaluate code updates using OpenAI's GPT-4 model.
    
    Parameters:
    - data: List of entries containing code examples
    - limit: Optional limit on number of entries to process
    - use_simple_template: Boolean indicating whether to use simple template
    - use_system_user_prompt: Boolean indicating whether to use system-user prompt format
    - use_mini: Boolean indicating whether to use gpt-4o-mini model
    - use_prediction: Boolean indicating whether to use prediction parameter
    
    Returns:
    - List of evaluation results
    """
    # Initialize OpenAI client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    client = AsyncOpenAI(api_key=api_key)
    
    # Select model based on use_mini flag
    model_name = "gpt-4o-mini" if use_mini else "gpt-4o"
    
    if limit is not None:
        data = data[:limit]
    
    results = []
    
    async def process_entry(entry):
        try:
            async with rate_limiter:
                # Extract code from entry
                original_code = entry.get('original_code', '')
                update_snippet = entry.get('update_snippet', '')
                
                # Prepare messages based on template type
                if use_system_user_prompt:
                    system_message = SYSTEM_PROMPT
                    user_message = USER_PROMPT.format(original_code=original_code, update_snippet=update_snippet)
                    messages = [
                        {'role': 'system', 'content': system_message},
                        {'role': 'user', 'content': user_message}
                    ]
                    prompt_text = user_message
                elif use_simple_template:
                    prompt_text = simple_template.format(original_code=original_code, update_snippet=update_snippet)
                    messages = [{'role': 'user', 'content': prompt_text}]
                else:
                    prompt_text = template.format(original_code=original_code, update_snippet=update_snippet)
                    messages = [{'role': 'user', 'content': prompt_text}]

                # Start timing
                start_time = time.time()
                
                # Prepare API call parameters
                api_params = {
                    'model': model_name,
                    'messages': messages,
                    'temperature': 0,
                    'max_tokens': 4000
                }
                
                # Add prediction if enabled
                if use_prediction:
                    api_params['prediction'] = {
                        'type': 'content',
                        'content': f'<updated-code>\n{original_code}\n</updated-code>'
                    }
                
                # Call GPT-4
                response = await client.chat.completions.create(**api_params)
                
                # Calculate timing and tokens
                elapsed_time = time.time() - start_time
                total_tokens = response.usage.total_tokens
                throughput = total_tokens / elapsed_time
                
                # Extract generated code
                generated_text = response.choices[0].message.content
                
                # Parse the generated text
                parsed_text = parse_generated_text(generated_text, use_simple_template)
                
                # Create result entry with generated text and metrics
                entry_with_generated = {
                    'original_code': original_code,
                    'update_snippet': update_snippet,
                    'final_code': entry.get('final_code', ''),
                    'generated_text': parsed_text,
                    'throughput': throughput,
                    'total_tokens': total_tokens,
                    'elapsed_time': elapsed_time,
                    'model': model_name,
                    'input': prompt_text,
                    'full_output': prompt_text + generated_text
                }
                
                # Calculate diffs
                diff_results = calculate_diff([entry_with_generated], limit=1, use_simple_template=use_simple_template)[0]
                
                # Merge metrics with diff results
                result = {**diff_results, **{
                    'throughput': throughput,
                    'total_tokens': total_tokens,
                    'elapsed_time': elapsed_time,
                    'model': model_name
                }}
                
                return result
                
        except Exception as e:
            print(f"Error processing entry: {str(e)}")
            return None
    
    # Process entries with progress bar
    tasks = [process_entry(entry) for entry in data]
    for future in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc=f"Evaluating with {model_name}"
    ):
        result = await future
        if result is not None:
            results.append(result)
    
    return results

async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate code updates using GPT-4")
    parser.add_argument("input_files", nargs='+', help="Paths to input JSON or Parquet files")
    parser.add_argument("--output_file", help="Path to output JSON file (optional)")
    parser.add_argument("-n", type=int, help="Number of examples to process (optional)")
    parser.add_argument("--simple", action="store_true", help="Use simple template without tags")
    parser.add_argument("--system-user", action="store_true", help="Use system-user prompt format")
    parser.add_argument("--mini", action="store_true", help="Use gpt-4o-mini model instead of gpt-4o")
    parser.add_argument("--prediction", action="store_true", help="Use prediction parameter with original code")
    args = parser.parse_args()
    
    all_results = {}
    
    for input_file in args.input_files:
        try:
            data = load_testset(input_file)
            if isinstance(data, pd.DataFrame):
                data = data.to_dict('records')
            
            results = await evaluate_with_gpt4(
                data, 
                limit=args.n, 
                use_simple_template=args.simple,
                use_system_user_prompt=args.system_user,
                use_mini=args.mini,
                use_prediction=args.prediction
            )
            all_results[input_file] = results
            
        except Exception as e:
            print(f"Error processing {input_file}: {e}")
            continue
    
    if args.output_file:
        try:
            with open(args.output_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"Results saved to {args.output_file}")
        except Exception as e:
            print(f"Error writing to {args.output_file}: {e}")
    
    for input_file, results in all_results.items():
        print_statistics(input_file, results)

if __name__ == "__main__":
    asyncio.run(main())