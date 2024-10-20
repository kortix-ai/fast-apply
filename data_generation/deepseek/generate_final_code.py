import os
import aiosqlite
import pandas as pd
import asyncio
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
from aiolimiter import AsyncLimiter
import argparse
import logging
from openai import AsyncOpenAI
from aiohttp import ClientError, ClientResponseError
from data_generation.utils import load_parquet, save_parquet, save_json, display_parquet_info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()

# Set DeepSeek API key and base URL for 8K tokens support
client = AsyncOpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/beta",
)

# Database file
DB_FILE = 'fix_query_cache.db'

# It limits the rate of asynchronous operations to 120 requests per 60 seconds
rate_limiter = AsyncLimiter(120, 60)

# Prompt template
PROMPT_TEMPLATE = """
You are an AI code reviewer tasked with updating and correcting code based on provided changes. Here's what you need to do:

First, you will be given two pieces of code:

1. The original code:
2. The update snippet containing changes:

Your task is to apply all changes from the update snippet to the original code and provide the corrected final code. Follow these guidelines:

1. Apply all changes from the update snippet to the original code.
2. Maintain the original formatting and structure of the code.
3. Ensure that all necessary parts of the code are included.
4. Enclose the complete, corrected code in <final_code> tags.

Important notes:
- Focus on accuracy and completeness when applying the changes.
- Only apply the changes specified in the update snippet.
- Preserve the original coding style and structure as much as possible.
- Do not add any explanations or comments about the changes.

Your output should contain only the final, corrected code enclosed in <final_code> tags. Do not include any additional text, explanations, or justifications for the changes made.
""".strip()

async def init_db(db):
    """Initialize the SQLite database and ensure the cache table has the correct schema."""
    # Check if 'cache' table exists
    async with db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='cache'") as cursor:
        table_exists = await cursor.fetchone()

    if table_exists:
        # Check if 'update_snippet' column exists
        async with db.execute("PRAGMA table_info(cache)") as cursor:
            columns = await cursor.fetchall()
            column_names = [column[1] for column in columns]

        if 'update_snippet' not in column_names:
            # Since SQLite does not support IF NOT EXISTS in ALTER TABLE, we catch the exception if the column exists
            try:
                await db.execute("ALTER TABLE cache ADD COLUMN update_snippet TEXT")
                await db.commit()
            except Exception as e:
                logging.error(f"Failed to add 'update_snippet' column: {e}")
    else:
        # Create the table with the correct schema
        await db.execute('''
            CREATE TABLE cache (
                original_code TEXT,
                update_snippet TEXT,
                generated_content TEXT,
                PRIMARY KEY (original_code, update_snippet)
            )
        ''')
        await db.commit()


async def clear_cache(db):
    """Clear all entries from the cache table."""
    await db.execute('DELETE FROM cache')
    await db.commit()
    logging.info("Cache cleared successfully.")


async def get_from_cache(db, original_code, update_snippet):
    """Retrieve a result from the cache."""
    async with db.execute("SELECT generated_content FROM cache WHERE original_code = ? AND update_snippet = ?", (original_code, update_snippet)) as cursor:
        result = await cursor.fetchone()
    return result[0] if result else None


async def add_to_cache(db, original_code, update_snippet, generated_content):
    """Add a result to the cache."""
    await db.execute("INSERT OR REPLACE INTO cache (original_code, update_snippet, generated_content) VALUES (?, ?, ?)",
                     (original_code, update_snippet, generated_content))
    await db.commit()


async def generate_update(db, original_code, update_snippet, existing_final_code):
    """Generate final code using DeepSeek API or cache."""
    # Check if the result is already in the cache
    cached_result = await get_from_cache(db, original_code, update_snippet)
    if cached_result:
        logging.info("Using cached result")
        return cached_result

    messages = [
        {"role": "system", "content": PROMPT_TEMPLATE},
    ]

    snippets_content = f"<original_code>\n{original_code}\n</original_code>"
    snippets_content += f"\n<update_snippet>\n{update_snippet}\n</update_snippet>"
    # snippets_content += f"\n<existing_final_code>\n{existing_final_code}\n</existing_final_code>"
    messages.append({"role": "user", "content": snippets_content.strip()})

    max_retries = 2
    attempt = 0
    backoff_factor = 2  # Exponential backoff

    while attempt <= max_retries:
        try:
            async with rate_limiter:
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model="deepseek-chat",
                        messages=messages,
                        stream=False,
                        temperature=0,
                        max_tokens=8192
                    ),
                    timeout=1000
                )
                content = response.choices[0].message.content
                # Cache the result
                await add_to_cache(db, original_code, update_snippet, content)
                return content
        except asyncio.TimeoutError:
            logging.error(f"Timeout after 15 minutes for original_code: {original_code[:30]}...")
            return "DELETE_ROW"
        except ClientResponseError as cre:
            status = cre.status
            logging.error(f"HTTP error {status} for original_code: {original_code[:30]}... - {cre.message}")
            # Handle specific error codes
            if status == 400:
                logging.error("400 - Invalid Format: Modify request body format.")
                return "DELETE_ROW"
            elif status == 401:
                logging.error("401 - Authentication Fails: Check API key.")
                return "DELETE_ROW"
            elif status == 402:
                logging.error("402 - Insufficient Balance: Top up your account.")
                return "DELETE_ROW"
            elif status == 422:
                logging.error("422 - Invalid Parameters: Modify request parameters.")
                return "DELETE_ROW"
            elif status == 429:
                logging.warning("429 - Rate Limit Reached: Pacing requests.")
                # Optionally, implement a wait before retrying
            elif status in [500, 503]:
                logging.warning(f"{status} - Server Error: Retrying after backoff.")
            else:
                logging.error(f"Unhandled HTTP error {status}.")
                return "DELETE_ROW"
        except ClientError as ce:
            logging.error(f"Client error: {ce} for original_code: {original_code[:30]}...")
        except Exception as e:
            logging.error(f"Unexpected error: {e} for original_code: {original_code[:30]}...")

        attempt += 1
        if attempt <= max_retries:
            wait_time = backoff_factor ** attempt + 10
            logging.info(f"Retrying in {wait_time} seconds... (Attempt {attempt} of {max_retries})")
            await asyncio.sleep(wait_time)
    logging.error(f"All retries failed for original_code: {original_code[:30]}...")
    return "DELETE_ROW"


async def process_row(db, idx, row):
    """Process a single row of the DataFrame."""
    file_name = row.get('File Name', idx)
    logging.info(f"Processing file: {file_name}")
    original_code = row['original_code']
    update_snippet = row['update_snippet'] if not pd.isna(row['update_snippet']) else ""
    existing_final_code = row['final_code'] if not pd.isna(row['final_code']) else ""

    generated_content = await generate_update(
        db,
        original_code,
        update_snippet,
        existing_final_code
    )
    if generated_content == "DELETE_ROW":
        logging.info(f"Deleting row for file: {file_name}")
        return idx, None

    if "The provided final code is complete and requires no changes." in generated_content:
        # Code is complete, no changes needed
        return idx, {
            'final_code': existing_final_code,
            'error': pd.NA,
            'status': 'complete'
        }

    if '<final_code>' in generated_content:
        try:
            final_code = generated_content.split('<final_code>')[1].split('</final_code>')[0].strip()
            return idx, {
                'final_code': final_code if final_code else existing_final_code,
                'old_final_code': existing_final_code if existing_final_code else pd.NA,
                'error': pd.NA,
                'status': 'fixed'
            }
        except IndexError:
            logging.error(f"Error processing file: {file_name}. Output doesn't match expected pattern.")
            return idx, {
                'final_code': existing_final_code,
                'error': generated_content,
                'status': 'error'
            }
    else:
        logging.error(f"Response missing expected tags for file: {file_name}")
        return idx, {
            'final_code': existing_final_code,
            'error': generated_content,
            'status': 'missing_tags'
        }


async def main(parquet_file, test_mode=False, should_clear_cache=False, test_samples=5):
    df = load_parquet(parquet_file)
    display_parquet_info(df)

    if test_mode:
        df = df.head(test_samples)
        logging.info(f"Test mode: Processing only the first {test_samples} rows")

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
                df.at[idx, 'final_code'] = result['final_code']
                if 'old_final_code' in result:
                    df.at[idx, 'old_final_code'] = result.get('old_final_code', pd.NA)
                df.at[idx, 'error'] = result['error']
                df.at[idx, 'status'] = result['status']

        # Remove rows that encountered errors
        if rows_to_delete:
            logging.info(f"Removing {len(rows_to_delete)} rows due to errors")
            df = df.drop(rows_to_delete)

        # Save final progress
        save_parquet(df, parquet_file)

        # Save as JSON
        json_file = parquet_file.rsplit('.', 1)[0] + '.json'
        save_json(df, json_file)

        logging.info("Processing complete")

    if test_mode:
        pass
    else:
        logging.info("Full processing completed. Check the 'fixed_' output files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a Parquet file and check/fix code examples.")
    parser.add_argument("--parquet_file", type=str, default="data/output.parquet",
                        help="Path to the input Parquet file (default: data/output.parquet)")
    parser.add_argument("--test", nargs='?', const=5, type=int, metavar='N', help="Run in test mode (process N prompts, default 5)")
    parser.add_argument("--clear-cache", action="store_true", help="Clear the cache before processing")
    args = parser.parse_args()

    asyncio.run(main(args.parquet_file, test_mode=bool(args.test), should_clear_cache=args.clear_cache, test_samples=args.test or 5))
