import os
import aiosqlite
import pandas as pd
import anthropic
from anthropic import BadRequestError
import tiktoken
import argparse
import asyncio
import json
from dotenv import load_dotenv
from prompt_template import PROMPT
from tqdm.asyncio import tqdm
from aiolimiter import AsyncLimiter
import sys
from data_generation.utils import load_parquet, save_parquet, save_json, display_parquet_info

# Load environment variables
load_dotenv()

# Initialize Anthropic client
client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Initialize tiktoken encoder
enc = tiktoken.get_encoding("cl100k_base")  # Use cl100k_base encoding for Claude models

# Initialize token counters
input_tokens = 0
output_tokens = 0
token_lock = asyncio.Lock()  # Lock for token counters

# Database file
DB_FILE = 'query_cache.db'

# Rate limiter: 40 requests per minute
rate_limiter = AsyncLimiter(50, 60)

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


async def generate_update(db, original_code):
    """Generate update snippet and final code using Anthropic API or cache."""
    global input_tokens, output_tokens

    # Check if the result is already in the cache
    cached_result = await get_from_cache(db, original_code)
    if cached_result:
        print("Using cached result")
        return cached_result

    prompt = PROMPT.format(original_code=original_code)
    token_count = len(enc.encode(prompt))
    async with token_lock:
        input_tokens += token_count

    try:
        async with rate_limiter:
            message = await client.messages.create(
                model="claude-3-5-sonnet-20240620",
                temperature=0,
                max_tokens=8192,
                messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            )

        content = message.content[0].text
        token_count = len(enc.encode(content))
        async with token_lock:
            output_tokens += token_count

        # Cache the result
        await add_to_cache(db, original_code, content)

        return content
    except BadRequestError as e:
        print(f"Error: {e}")
        return "DELETE_ROW"

async def process_row(db, idx, row):
    """Process a single row of the DataFrame."""
    if pd.isna(row['update_snippet']) or pd.isna(row['final_code']):
        print(f"Processing file: {row['File Name']}")
        generated_content = await generate_update(db, row['original_code'])
        if generated_content == "DELETE_ROW":
            print(f"Deleting row for file: {row['File Name']}")
            return idx, None
        try:
            update_snippet = generated_content.split('<update_snippet>')[1].split('</update_snippet>')[0].strip()
            final_code = generated_content.split('<final_code>')[1].split('</final_code>')[0].strip()
            return idx, {'update_snippet': update_snippet, 'final_code': final_code, 'error': pd.NA}
        except IndexError:
            print(f"Error processing file: {row['File Name']}. Output doesn't match expected pattern.")
            return idx, {'update_snippet': pd.NA, 'final_code': pd.NA, 'error': generated_content}
    else:
        print(f"File already processed, skipping: {row['File Name']}")
        return idx, {'update_snippet': row['update_snippet'], 'final_code': row['final_code'], 'error': row['error']}

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

        # Prepare tasks
        tasks = [asyncio.create_task(process_row(db, idx, row)) for idx, row in df.iterrows()]

        # Process tasks as they complete
        rows_to_delete = []
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing files"):
            idx, result = await future
            if result is None:
                rows_to_delete.append(idx)
            else:
                df.at[idx, 'update_snippet'] = result['update_snippet']
                df.at[idx, 'final_code'] = result['final_code']
                df.at[idx, 'error'] = result['error']

        # Remove rows that encountered the BadRequestError
        if rows_to_delete:
            print(f"Removing {len(rows_to_delete)} rows due to content filtering errors")
            df = df.drop(rows_to_delete)

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
    parser = argparse.ArgumentParser(description="Process a Parquet file and generate code updates.")
    parser.add_argument("--parquet_file", type=str, default="data/output.parquet",
                        help="Path to the input Parquet file (default: data/output.parquet)")
    parser.add_argument("--test", action="store_true", help="Run in test mode (process only 5 prompts)")
    parser.add_argument("--clear-cache", action="store_true", help="Clear the cache before processing")
    args = parser.parse_args()

    asyncio.run(main(args.parquet_file, test_mode=args.test, should_clear_cache=args.clear_cache))

