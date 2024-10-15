import os
import aiosqlite
import pyarrow.parquet as pq
import pandas as pd
import asyncio
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
from aiolimiter import AsyncLimiter
import argparse
# from openai import OpenAI
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()

# Set OpenAI API key and base URL for 8K tokens support
client = AsyncOpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/beta",
)

# Database file
DB_FILE = 'fix_query_cache.db'

rate_limiter = AsyncLimiter(60, 60)  # 60 requests per minute

# Prompt template
PROMPT_TEMPLATE = """
You are an AI assistant specialized in reviewing and correcting synthetic code update examples for model training purposes. Your task is to ensure that each data example is accurate, complete, and follows the specified format. Based on the original code, existing update snippet, and final code provided, please perform the following steps:

1. **Create an Update Snippet**
   - Modify the original code as specified (e.g., add features, remove code).
   - Include only the new or changed code.
   - Use the exact ellipsis comment `// ... existing code ...` to represent omitted unchanged lines.
   - Focus only on the relevant parts; do not include the entire code.
   - Ensure the update snippet is concise and clearly shows where changes are applied.
   - Enclose the update snippet within `<update_snippet>` tags.

2. **Provide the Final Updated Code**
   - Start with the original code.
   - Apply only the changes from your update snippet.
   - Do not make any additional modifications beyond what is specified in the update snippet.
   - Retain all original formatting and structure.
   - Enclose the final updated code within `<final_code>` tags.

**Instructions**
- If the provided code is correct and requires no changes, simply state "The provided code is correct and requires no changes." and finish.
- Do not include any explanations or commentary outside of the specified tags.
- Begin your response with the update snippet, followed immediately by the final updated code (if applicable).

**Example Output:**

*If corrections are needed:*

<update_snippet>
// ... existing code ...

// Add after initializePlugins function
function loadCustomPlugins(pluginConfigs) {{
  return pluginConfigs.map(({{
    name, options
  }}) => require(name)(options));
}}

// Update the plugins array in the configuration
export default ({{ mode }}) => {{
  const env = loadEnv(mode, process.cwd(), "");

  return defineConfig({{
    plugins: [
      ...initializePlugins(),
      ...loadCustomPlugins([
        {{ name: 'vite-plugin-pwa', options: {{}} }},
        {{ name: 'vite-plugin-svgr', options: {{}} }}
      ])
    ],
    // ... existing code ...
  }});
}}</update_snippet>

<final_code>
import {{ defineConfig }} from 'vite';
import {{ loadEnv }} from 'vite';
import react from '@vitejs/plugin-react';

function initializePlugins() {{
  return [
    react(),
  ];
}}

function loadCustomPlugins(pluginConfigs) {{
  return pluginConfigs.map(({{
    name, options
  }}) => require(name)(options));
}}

export default ({{ mode }}) => {{
  const env = loadEnv(mode, process.cwd(), "");

  return defineConfig({{
    plugins: [
      ...initializePlugins(),
      ...loadCustomPlugins([
        {{ name: 'vite-plugin-pwa', options: {{}} }},
        {{ name: 'vite-plugin-svgr', options: {{}} }}
      ])
    ],
    build: {{
      sourcemap: true,
      rollupOptions: {{
        output: {{
          manualChunks: {{
            vendor: ['react', 'react-dom'],
          }},
        }},
      }},
    }},
    server: {{
      port: 3000,
      proxy: {{
        '/api': 'http://localhost:8080',
      }},
    }},
    resolve: {{
      alias: {{ '@': '/src' }},
    }},
  }});
}};
</final_code>

*If no corrections are needed:*

The provided code is correct and requires no changes.
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
    print("Parquet File Information:")
    print(f"Number of files: {len(df)}")
    print("\nSchema:")
    print(df.dtypes)
    print("\nFirst few rows:")
    print(df[['File Name']].head())

async def generate_update(db, original_code, existing_update_snippet, existing_final_code):
    """Generate update snippet and final code using OpenAI API or cache."""
    # Check if the result is already in the cache
    cached_result = await get_from_cache(db, original_code)
    if cached_result:
        print("Using cached result")
        return cached_result

    messages = [
        {"role": "system", "content": PROMPT_TEMPLATE},
    ]

    snippets_content = f"<original_code>\n{original_code}\n</original_code>"
    if existing_update_snippet or existing_final_code:
        if existing_update_snippet:
            snippets_content += f"<update_snippet>\n{existing_update_snippet}\n</update_snippet>\n\n"
        if existing_final_code:
            snippets_content += f"<final_code>\n{existing_final_code}\n</final_code>"
    messages.append({"role": "user", "content": snippets_content.strip()})

    try:
        async with rate_limiter:
            response = await client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                stream=False,
                temperature=0,
                max_tokens=8192
            )
            content = response.choices[0].message.content
            # Cache the result
            await add_to_cache(db, original_code, content)
            return content
    except Exception as e:
        print(f"Exception during API call: {e}")
        return "DELETE_ROW"

async def process_row(db, idx, row):
    """Process a single row of the DataFrame."""
    print(f"Processing file: {row.get('File Name', idx)}")
    generated_content = await generate_update(
        db,
        row['original_code'],
        row['update_snippet'] if not pd.isna(row['update_snippet']) else "",
        row['final_code'] if not pd.isna(row['final_code']) else ""
    )
    if generated_content == "DELETE_ROW":
        print(f"Deleting row for file: {row.get('File Name', idx)}")
        return idx, None
    if "The provided code is correct and requires no changes." in generated_content:
        # Code is correct, no changes needed
        return idx, {'update_snippet': row['update_snippet'] if not pd.isna(row['update_snippet']) else pd.NA,
                    'final_code': row['final_code'] if not pd.isna(row['final_code']) else pd.NA,
                    'error': pd.NA,
                    'status': 'correct'}
    if '<update_snippet>' in generated_content and '<final_code>' in generated_content:
        try:
            update_snippet = generated_content.split('<update_snippet>')[1].split('</update_snippet>')[0].strip()
            final_code = generated_content.split('<final_code>')[1].split('</final_code>')[0].strip()
            return idx, {'update_snippet': update_snippet if update_snippet else row['update_snippet'],
                        'final_code': final_code if final_code else row['final_code'],
                        'error': pd.NA,
                        'status': 'fixed'}
        except IndexError:
            print(f"Error processing file: {row.get('File Name', idx)}. Output doesn't match expected pattern.")
            return idx, {'update_snippet': row['update_snippet'],
                        'final_code': row['final_code'],
                        'error': generated_content,
                        'status': 'error'}
    else:
        print(f"Response missing expected tags for file: {row.get('File Name', idx)}")
        return idx, {'update_snippet': row['update_snippet'],
                    'final_code': row['final_code'],
                    'error': generated_content,
                    'status': 'missing_tags'}

def save_parquet(df, parquet_file):
    output_file = f"fixed_{parquet_file.split('/')[-1]}"
    df.to_parquet(output_file, index=False)
    print(f"Updated Parquet file saved to {output_file}")

def save_json(df, json_file):
    output_file = f"fixed_{json_file.split('/')[-1]}"
    df.to_json(output_file, orient='records', indent=2)
    print(f"JSON file saved to {output_file}")

async def main(parquet_file, test_mode=False, should_clear_cache=False):
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
                df.at[idx, 'status'] = result['status']

        # Remove rows that encountered errors
        if rows_to_delete:
            print(f"Removing {len(rows_to_delete)} rows due to errors")
            df = df.drop(rows_to_delete)

        # Save final progress
        save_parquet(df, parquet_file)

        # Save as JSON
        json_file = parquet_file.rsplit('.', 1)[0] + '.json'
        save_json(df, json_file)

        print("Processing complete")

    if test_mode:
        print("Test mode: JSON output:")
        print(df.to_json(orient='records', indent=2))
    else:
        print("Full processing completed. Check the 'fixed_' output files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a Parquet file and check/fix code examples.")
    parser.add_argument("--parquet_file", type=str, default="data/output.parquet",
                        help="Path to the input Parquet file (default: data/output.parquet)")
    parser.add_argument("--test", action="store_true", help="Run in test mode (process only 5 prompts)")
    parser.add_argument("--clear-cache", action="store_true", help="Clear the cache before processing")
    args = parser.parse_args()

    asyncio.run(main(args.parquet_file, test_mode=args.test, should_clear_cache=args.clear_cache))
