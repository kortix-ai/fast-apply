import os
import aiosqlite
import pyarrow.parquet as pq
import pandas as pd
import asyncio
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
from aiolimiter import AsyncLimiter
import argparse
import logging
from openai import AsyncOpenAI
from aiohttp import ClientError, ClientResponseError

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
As an AI coding assistant, you will be provided with:

- **Original Code**
- **Update Snippet**
- **Existing Final Code**

Your task is to:

1. **Verify Completeness**: Check if the **Existing Final Code** includes all changes specified in the **Update Snippet** applied to the **Original Code**.

2. **Output**:
   - **If the Final Code is complete** (all changes are correctly applied):
     - Output only: `The provided final code is complete and requires no changes.`
   - **If the Final Code is incomplete or incorrect** (trucated, not all changes are applied or applied incorrectly):
     - Generate the corrected full final code with all changes applied.
     - Enclose the corrected code within `<final_code>` and `</final_code>` tags, like so:

       ```
       <final_code>[Provide the complete final code here]</final_code>
       ```
""".strip()


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
    logging.info("Cache cleared successfully.")


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


def load_parquet(file_path):
    """Load a Parquet file and return a DataFrame with necessary columns."""
    df = pq.read_table(file_path).to_pandas()
    if 'Content' in df.columns:
        df = df.rename(columns={'Content': 'original_code'})
    for column in ['update_snippet', 'final_code', 'error']:
        if column not in df.columns:
            df[column] = pd.NA
    return df


def display_parquet_info(df):
    """Display information about the Parquet file."""
    logging.info("Parquet File Information:")
    logging.info(f"Number of files: {len(df)}")
    logging.info("\nSchema:")
    logging.info(df.dtypes)
    logging.info("\nFirst few rows:")
    logging.info(df[['File Name']].head())


async def generate_update(db, original_code, existing_update_snippet, existing_final_code):
    """Generate update snippet and final code using DeepSeek API or cache."""
    # Check if the result is already in the cache
    cached_result = await get_from_cache(db, original_code)
    if cached_result:
        logging.info("Using cached result")
        return cached_result

    messages = [
        {"role": "system", "content": PROMPT_TEMPLATE},
    ]

    snippets_content = f"<original_code>\n{original_code}\n</original_code>"
    if existing_update_snippet or existing_final_code:
        if existing_update_snippet:
            snippets_content += f"<update_snippet>\n{existing_update_snippet}\n</update_snippet>\n\n"
        if existing_final_code:
            snippets_content += f"<existing_final_code>\n{existing_final_code}\n</existing_final_code>"
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
                await add_to_cache(db, original_code, content)
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
    generated_content = await generate_update(
        db,
        row['original_code'],
        row['update_snippet'] if not pd.isna(row['update_snippet']) else "",
        row['final_code'] if not pd.isna(row['final_code']) else ""
    )
    if generated_content == "DELETE_ROW":
        logging.info(f"Deleting row for file: {file_name}")
        return idx, None
    
    # Parse describe-changes
    describe_changes = pd.NA
    if '<describe_changes>' in generated_content and '</describe_changes>' in generated_content:
        describe_changes = generated_content.split('<describe_changes>')[1].split('</describe_changes>')[0].strip()
    
    if "The provided update snippet and final code are correct and require no changes." in generated_content:
        # Code is correct, no changes needed
        return idx, {
            'update_snippet': row['update_snippet'] if not pd.isna(row['update_snippet']) else pd.NA,
            'final_code': row['final_code'] if not pd.isna(row['final_code']) else pd.NA,
            'error': pd.NA,
            'status': 'correct',
            'describe_changes': describe_changes
        }
    
    if '<update_snippet>' in generated_content and '<final_code>' in generated_content:
        try:
            update_snippet = generated_content.split('<update_snippet>')[1].split('</update_snippet>')[0].strip()
            final_code = generated_content.split('<final_code>')[1].split('</final_code>')[0].strip()
            return idx, {
                'update_snippet': update_snippet if update_snippet else row['update_snippet'],
                'final_code': final_code if final_code else row['final_code'],
                'old_update_snippet': row['update_snippet'] if not pd.isna(row['update_snippet']) else pd.NA,
                'old_final_code': row['final_code'] if not pd.isna(row['final_code']) else pd.NA,
                'error': pd.NA,
                'status': 'fixed',
                'describe_changes': describe_changes
            }
        except IndexError:
            logging.error(f"Error processing file: {file_name}. Output doesn't match expected pattern.")
            return idx, {
                'update_snippet': row['update_snippet'],
                'final_code': row['final_code'],
                'error': generated_content,
                'status': 'error',
                'describe_changes': describe_changes
            }
    else:
        logging.error(f"Response missing expected tags for file: {file_name}")
        return idx, {
            'update_snippet': row['update_snippet'],
            'final_code': row['final_code'],
            'error': generated_content,
            'status': 'missing_tags',
            'describe_changes': describe_changes
        }


def save_parquet(df, parquet_file):
    """Save the DataFrame to a Parquet file."""
    output_file = os.path.join(os.path.dirname(parquet_file), f"fixed_{os.path.basename(parquet_file)}")
    df.to_parquet(output_file, index=False)
    logging.info(f"Updated Parquet file saved to {output_file}")


def save_json(df, json_file):
    """Save the DataFrame to a JSON file."""
    output_file = os.path.join(os.path.dirname(json_file), f"fixed_{os.path.basename(json_file)}")
    df.to_json(output_file, orient='records', indent=2)
    logging.info(f"JSON file saved to {output_file}")


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
                df.at[idx, 'update_snippet'] = result['update_snippet']
                df.at[idx, 'final_code'] = result['final_code']
                df.at[idx, 'old_update_snippet'] = result.get('old_update_snippet', pd.NA)
                df.at[idx, 'old_final_code'] = result.get('old_final_code', pd.NA)
                df.at[idx, 'error'] = result['error']
                df.at[idx, 'status'] = result['status']
                df.at[idx, 'describe_changes'] = result['describe_changes']

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