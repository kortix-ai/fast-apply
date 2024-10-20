import os
import aiosqlite
import pandas as pd
import anthropic
from anthropic import BadRequestError
from anthropic.types.beta.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.beta.messages.batch_create_params import Request
import tiktoken
import argparse
import asyncio
import json
import random
from dotenv import load_dotenv
from prompt_template import ONLY_UPDATE_PROMPT, GOAL
from tqdm.asyncio import tqdm
from aiolimiter import AsyncLimiter
from data_generation.utils import load_parquet, save_parquet, save_json, display_parquet_info

# Load environment variables
load_dotenv()

# Initialize Anthropic client with beta features
client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Initialize tiktoken encoder
enc = tiktoken.get_encoding("cl100k_base")  # Use cl100k_base encoding for Claude models

# Initialize token counters
input_tokens = 0
output_tokens = 0
token_lock = asyncio.Lock()  # Lock for token counters

# Database file
DB_FILE = 'query_cache.db'

# Rate limiter: 50 requests per minute
rate_limiter = AsyncLimiter(50, 60)

# Updated GOAL string with weights
# Parse GOAL string to extract goals and weights
GOAL_LIST = []
GOAL_WEIGHTS = []
for line in GOAL.strip().split('\n'):
    if line.strip():
        goal_part, weight_part = line.rsplit(':', 1)
        goal = goal_part.strip()
        weight = int(weight_part.strip())
        GOAL_LIST.append(goal)
        GOAL_WEIGHTS.append(weight)

# Batch configuration
BATCH_SIZE = 1000  # Adjust based on your needs and API limits

async def init_db(db):
    """Initialize the SQLite database and create the cache table if it doesn't exist."""
    await db.execute('''
        CREATE TABLE IF NOT EXISTS cache (
            original_code TEXT PRIMARY KEY,
            generated_content TEXT
        )
    ''')
    await db.commit()

async def clear_cache(db):
    """Clear all entries from the cache table."""
    await db.execute('DELETE FROM cache')
    await db.commit()
    print("Cache cleared successfully.")

async def get_from_cache(db, original_code):
    """Retrieve a result from the cache."""
    async with db.execute("SELECT generated_content FROM cache WHERE original_code = ?", (original_code,)) as cursor:
        result = await cursor.fetchone()
    return result[0] if result else None

async def add_to_cache(db, original_code, generated_content):
    """Add a result to the cache."""
    await db.execute("INSERT OR REPLACE INTO cache (original_code, generated_content) VALUES (?, ?)",
                     (original_code, generated_content))
    await db.commit()

def update_token_count_json(input_tokens, output_tokens):
    """Update or create a JSON file with token counts."""
    json_file = '.token_counts.json'
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
    else:
        data = {"input_tokens": 0, "output_tokens": 0}

    data["input_tokens"] += input_tokens
    data["output_tokens"] += output_tokens

    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Token counts updated in {json_file}")

async def process_cached_result(idx, cached_content, df):
    """Process cached result and update DataFrame."""
    try:
        update_snippet = cached_content.split('<update-snippet>')[1].split('</update-snippet>')[0].strip()
        final_code = pd.NA  # Modify if final_code is also generated
        df.at[idx, 'update_snippet'] = update_snippet
        df.at[idx, 'final_code'] = final_code
        df.at[idx, 'error'] = pd.NA
    except IndexError:
        print(f"Error processing cached result for idx {idx}. Content doesn't match expected pattern.")
        df.at[idx, 'error'] = cached_content

async def create_batches(df, db):
    """Create batches of requests based on the Batch API limitations."""
    batches = []
    total_requests = len(df)
    for i in range(0, total_requests, BATCH_SIZE):
        batch_df = df.iloc[i:i+BATCH_SIZE]
        requests = []
        for idx, row in batch_df.iterrows():
            if pd.isna(row['update_snippet']) or pd.isna(row['final_code']):
                custom_id = f"request_{idx}"
                selected_goal = random.choices(GOAL_LIST, weights=GOAL_WEIGHTS, k=1)[0]
                prompt = ONLY_UPDATE_PROMPT.format(original_code=row['original_code'], goal=selected_goal)
                # Check cache
                cached_result = await get_from_cache(db, row['original_code'])
                if cached_result:
                    print(f"Using cached result for idx {idx}")
                    # Process cached result directly
                    await process_cached_result(idx, cached_result, df)
                    continue
                params = MessageCreateParamsNonStreaming(
                    model="claude-3-5-sonnet-20240620",
                    temperature=0,
                    max_tokens=8192,
                    messages=[{"role": "user", "content": prompt}]
                )
                request = Request(
                    custom_id=custom_id,
                    params=params
                )
                requests.append(request)
        if requests:
            batches.append(requests)
    return batches

async def send_batch(batch_requests):
    """Send a single batch to the Message Batches API."""
    try:
        message_batch = await client.beta.messages.batches.create(
            requests=batch_requests,
            betas=["message-batches-2024-09-24"]
        )
        return message_batch.id
    except Exception as e:
        print(f"Failed to create batch: {e}")
        return None

async def poll_batch_status(batch_id):
    """Poll the status of a batch until it is completed or expired."""
    while True:
        batch = await client.beta.messages.batches.retrieve(batch_id)
        status = batch.processing_status
        print(f"Batch {batch_id} status: {status}")
        if status == "ended":
            return batch
        elif status in ["canceled", "expired"]:
            print(f"Batch {batch_id} ended with status: {status}")
            return batch
        await asyncio.sleep(30)  # Wait for 30 seconds before polling again

async def retrieve_and_process_results(batch_id, df, db):
    """Retrieve results of a batch and update the DataFrame and cache accordingly."""
    global input_tokens, output_tokens

    try:
        async for result in client.beta.messages.batches.results(batch_id):
            custom_id = result.custom_id
            idx = int(custom_id.split("_")[1])

            if result.result.type == "succeeded":
                message = result.result.message
                content = message.content[0]['text']
                # Update token counts
                async with token_lock:
                    input_tokens += message.usage.input_tokens
                    output_tokens += message.usage.output_tokens
                # Cache the result
                await add_to_cache(db, df.at[idx, 'original_code'], content)
                try:
                    update_snippet = content.split('<update-snippet>')[1].split('</update-snippet>')[0].strip()
                    final_code = pd.NA  # Modify if final_code is also generated
                    df.at[idx, 'update_snippet'] = update_snippet
                    df.at[idx, 'final_code'] = final_code
                    df.at[idx, 'error'] = pd.NA
                except IndexError:
                    print(f"Error processing result for idx {idx}. Content doesn't match expected pattern.")
                    df.at[idx, 'error'] = content
            elif result.result.type == "errored":
                error_type = result.result.error.type
                print(f"Error in request {custom_id}: {error_type}")
                df.at[idx, 'error'] = error_type
            elif result.result.type == "canceled":
                print(f"Request {custom_id} was canceled.")
                df.at[idx, 'error'] = "canceled"
            elif result.result.type == "expired":
                print(f"Request {custom_id} expired.")
                df.at[idx, 'error'] = "expired"
    except Exception as e:
        print(f"Failed to retrieve results for batch {batch_id}: {e}")

async def process_batches(batches, df, db):
    """Process all batches sequentially."""
    for batch_requests in tqdm(batches, desc="Processing batches"):
        batch_id = await send_batch(batch_requests)
        if batch_id:
            batch = await poll_batch_status(batch_id)
            if batch.processing_status == "ended":
                await retrieve_and_process_results(batch_id, df, db)
            else:
                print(f"Batch {batch_id} did not complete successfully.")
        else:
            print("Skipping batch due to creation failure.")

async def main(parquet_file, test_mode=False, should_clear_cache=False):
    global input_tokens, output_tokens
    df = load_parquet(parquet_file)
    display_parquet_info(df)

    if test_mode:
        df = df.head(5)
        print("Test mode: Processing only the first 5 rows")

    async with aiosqlite.connect(DB_FILE) as db:
        # Initialize the database
        await init_db(db)

        if should_clear_cache:
            await clear_cache(db)

        # Create batches
        batches = await create_batches(df, db)

        if not batches:
            print("No batches to process.")
            return

        # Process all batches
        await process_batches(batches, df, db)

        if not test_mode:
            # Save final progress
            save_parquet(df, parquet_file)
        
        # Save as JSON
        json_file = parquet_file.rsplit('.', 1)[0] + '.json'
        save_json(df, json_file)

        print("Processing complete")

        # Update token count JSON file
        update_token_count_json(input_tokens, output_tokens)

    if test_mode:
        print("Test mode: JSON output:")
        print(df.to_json(orient='records', indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a Parquet file and generate code updates using Batch API.")
    parser.add_argument("--parquet_file", type=str, default="data/output.parquet",
                        help="Path to the input Parquet file (default: data/output.parquet)")
    parser.add_argument("--test", action="store_true", help="Run in test mode (process only 5 prompts)")
    parser.add_argument("--clear-cache", action="store_true", help="Clear the cache before processing")
    args = parser.parse_args()

    asyncio.run(main(args.parquet_file, test_mode=args.test, should_clear_cache=args.clear_cache))
