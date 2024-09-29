import os
import argparse
import time
import pandas as pd
import numpy as np
import os
import pyarrow as pa
import pyarrow.parquet as pq
import tiktoken
from tqdm import tqdm
from collections import Counter
import sys
import io

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def should_ignore(path, is_dir=False):
    ignore_list = [
        '.git', '__pycache__', 'node_modules', 'venv', 'env',
        'build', 'dist', 'target', 'bin', 'obj',
        '.idea', '.vscode', '.gradle', 'LICENSE', '.github', 'CODEOWNERS'
        '.prettierignore',  '.dockerignore', 'prettierignore', '.gitignore', '.cursorignore',
    ]
    ignore_extensions = [
        '.pyc', '.pyo', '.pyd', '.so', '.dll', '.class',
        '.md', '.markdown', '.yaml', '.yml', '.json', '.xml',
        '.log', '.lock', '.cfg', '.ini', '.toml', '.parquet',
        '.webm', '.png', '.gif', '.jpg', '.jpeg', '.bmp', '.tiff',
        '.mp3', '.mp4', '.avi', '.mov', '.flv', '.wav',
        '.zip', '.tar', '.gz', '.rar', '.7z',
        '.exe', '.bin', '.iso',
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
        '.svg', '.ico', '.ttf', '.woff', '.woff2',
        '.min.js', '.min.css',
        '.cjs', '.example', '.hbs', 
        '.map', '.otf', '.snap', '.svelte', '.template',
        '.tpl', '.txt', '.webp',
        '.mdx', '.snapshot', '.pem', '.pic', '.config', '.patch',
        '.alt', '.approvers', '.avif', '.bak', '.default', '.dev', '.development', '.empty', '.eot', '.glb', '.i18n-images', '.icns', '.local', '.new', '.plist', '.po', '.production', '.sample', '.skip', '.stderr', '.test', '.webmanifest', '.xyz', '.drawio'
    ]
    
    name = os.path.basename(path)
    
    if is_dir:
        return name in ignore_list
    else:
        file_extension = os.path.splitext(name)[1].lower()
        return name in ignore_list or file_extension in ignore_extensions

def count_lines_and_tokens(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        content = file.read()
        lines = content.split('\n')
        
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(content)
    
    return len(lines), len(tokens)

def process_directory(path, debug=False):
    results = []
    file_data = []
    all_extensions = set()
    total_files = 0
    no_extension_files = []

    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if not should_ignore(os.path.join(root, d), is_dir=True)]
        total_files += len([f for f in files if not should_ignore(os.path.join(root, f))])

    with tqdm(total=total_files, desc="Processing files", unit="file") as pbar:
        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if not should_ignore(os.path.join(root, d), is_dir=True)]
            
            for file in files:
                file_path = os.path.join(root, file)
                if not should_ignore(file_path):
                    file_extension = os.path.splitext(file)[1].lower()
                    all_extensions.add(file_extension)
                    
                    if file_extension == '':
                        no_extension_files.append(file_path)
                    
                    line_count, token_count = count_lines_and_tokens(file_path)
                    
                    if token_count >= 1000:
                        content = read_file_content(file_path)
                        results.append((file_path, content))

                    file_data.append({
                        'file_path': file_path,
                        'line_count': line_count,
                        'token_count': token_count
                    })

                    pbar.update(1)

    df = pd.DataFrame(file_data)
    
    total_lines = df['line_count'].sum()
    total_tokens = df['token_count'].sum()
    included_files = len(results)
    max_lines = df['line_count'].max()
    max_tokens = df['token_count'].max()
    file_with_max_lines = df.loc[df['line_count'].idxmax(), 'file_path']
    file_with_max_tokens = df.loc[df['token_count'].idxmax(), 'file_path']

    # Updated bins for token distribution
    token_distribution = pd.cut(df['token_count'], 
                                bins=[0, 200, 1000, 2000, 3000, 4000, 5000, 10000, np.inf], 
                                labels=['<200 tokens', '200-999 tokens', '1000-1999 tokens', '2000-2999 tokens', 
                                        '3000-3999 tokens', '4000-4999 tokens', '5000-9999 tokens', 
                                        '10000+ tokens'])

    token_dist_dict = token_distribution.value_counts().to_dict()

    return (results, total_files, total_lines, total_tokens, included_files, 
            max_lines, max_tokens, file_with_max_lines, file_with_max_tokens, 
            token_dist_dict, all_extensions, df, no_extension_files)

def read_file_content(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        return file.read()

def save_to_parquet(results, output_file):
    df = pd.DataFrame(results, columns=['File Name', 'original_code', 'Line Count', 'Token Count'])
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_file)

def sample_dataset(df, sample_sizes):
    # Updated bins for sampling based on tokens
    bins = {
        '<200 tokens': (0, 200),
        '200-999 tokens': (200, 1000),
        '1000-1999 tokens': (1000, 2000),
        '2000-2999 tokens': (2000, 3000),
        '3000-3999 tokens': (3000, 4000),
        '4000-4999 tokens': (4000, 5000),
        '5000-9999 tokens': (5000, 10000),
        '10000+ tokens': (10000, np.inf)
    }
    
    sampled_dfs = []
    for bin_name, size in sample_sizes.items():
        if size > 0:
            lower, upper = bins[bin_name]
            bin_df = df[(df['token_count'] >= lower) & (df['token_count'] < upper)]
            if len(bin_df) > 0:
                if len(bin_df) > size:
                    sampled_dfs.append(bin_df.sample(size))
                else:
                    sampled_dfs.append(bin_df)
                print(f"Sampled {len(sampled_dfs[-1])} files from {bin_name} (requested: {size})")
            else:
                print(f"No files found in the {bin_name} range")
    
    if not sampled_dfs:
        print("No samples were selected. Please check your sampling parameters and the content of your dataset.")
        return None
    
    return pd.concat(sampled_dfs, ignore_index=True)

def print_sample_statistics(sampled_df):
    print("\nSampled Dataset Statistics:")
    print(f"Total files in sample: {len(sampled_df)}")
    print(f"Total lines in sample: {sampled_df['line_count'].sum()}")
    print(f"Total tokens in sample: {sampled_df['token_count'].sum()}")
    
    highest_token_count = sampled_df['token_count'].max()
    file_with_highest_tokens = sampled_df.loc[sampled_df['token_count'].idxmax(), 'file_path']
    print(f"\nHighest token count in sample: {highest_token_count}")
    print(f"File with highest token count: {file_with_highest_tokens}")
    
    extensions = Counter(sampled_df['file_path'].apply(lambda x: os.path.splitext(x)[1].lower()))
    print("\nFile Extensions in Sample:")
    for ext, count in extensions.most_common():
        print(f"  {ext}: {count}")
    
    # Updated bins for token distribution in sample statistics
    token_distribution = pd.cut(sampled_df['token_count'], 
                                bins=[0, 200, 1000, 2000, 3000, 4000, 5000, 10000, np.inf], 
                                labels=['<200 tokens', '200-999 tokens', '1000-1999 tokens', '2000-2999 tokens', 
                                        '3000-3999 tokens', '4000-4999 tokens', '5000-9999 tokens', 
                                        '10000+ tokens'])
    print("\nToken Distribution in Sample:")
    for category, count in token_distribution.value_counts().items():
        print(f"  {category}: {count}")

def main():
    parser = argparse.ArgumentParser(description='Recursively read files, sample, and save to Parquet file.')
    parser.add_argument('path', help='Path to the directory to process')
    parser.add_argument('--output', default='output.parquet', help='Output file name (default: output.parquet)')
    parser.add_argument('--log', default='repo.log', help='Log file name (default: repo.log)')
    parser.add_argument('--sample-lt-200', type=int, default=50, help='Number of samples for files with <200 tokens')
    parser.add_argument('--sample-200-999', type=int, default=50, help='Number of samples for files with 200-999 tokens')
    parser.add_argument('--sample-1000-1999', type=int, default=300, help='Number of samples for files with 1000-1999 tokens')
    parser.add_argument('--sample-2000-2999', type=int, default=0, help='Number of samples for files with 2000-2999 tokens')
    parser.add_argument('--sample-3000-3999', type=int, default=0, help='Number of samples for files with 3000-3999 tokens')
    parser.add_argument('--sample-4000-4999', type=int, default=0, help='Number of samples for files with 4000-4999 tokens')
    parser.add_argument('--sample-5000-9999', type=int, default=0, help='Number of samples for files with 5000-9999 tokens')
    parser.add_argument('--sample-10000-plus', type=int, default=0, help='Number of samples for files with 10000+ tokens')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    # Set up logging
    sys.stdout = Logger(args.log)

    start_time = time.time()
    (results, total_files, total_lines, total_tokens, included_files,
     max_lines, max_tokens, file_with_max_lines, file_with_max_tokens,
     token_distribution, all_extensions, df, no_extension_files) = process_directory(args.path, args.debug)

    sample_sizes = {
        '<200 tokens': args.sample_lt_200,
        '200-999 tokens': args.sample_200_999,
        '1000-1999 tokens': args.sample_1000_1999,
        '2000-2999 tokens': args.sample_2000_2999,
        '3000-3999 tokens': args.sample_3000_3999,
        '4000-4999 tokens': args.sample_4000_4999,
        '5000-9999 tokens': args.sample_5000_9999,
        '10000+ tokens': args.sample_10000_plus
    }

    sampled_df = sample_dataset(df, sample_sizes)
    if sampled_df is not None and not sampled_df.empty:
        sampled_results = [(row['file_path'], read_file_content(row['file_path']), row['line_count'], row['token_count'])
                           for _, row in sampled_df.iterrows()]
        save_to_parquet(sampled_results, args.output)
        end_time = time.time()

        print(f"\nResults saved to {args.output}")
        print("\nOriginal Dataset Statistics:")
        print(f"Total files processed: {total_files}")
        print(f"Total lines processed: {total_lines}")
        print(f"Total tokens processed: {total_tokens}")
        print(f"Files included in output: {included_files}")
        print(f"Files ignored: {total_files - included_files}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        print(f"\nHighest number of lines in a file: {max_lines}")
        print(f"File with the most lines: {file_with_max_lines}")
        print(f"Maximum number of tokens in a file: {max_tokens}")
        print(f"File with the most tokens: {file_with_max_tokens}")
        print("\nToken Distribution:")
        for category, count in token_distribution.items():
            print(f"  {category}: {count}")
        print("\nDistinct File Extensions:")
        for ext in sorted(all_extensions):
            print(f"  {ext}")

        print_sample_statistics(sampled_df)

        if args.debug:
            print("\nFiles with no extension:")
            for file in no_extension_files:
                print(f"  {file}")
    else:
        print("\nNo samples were selected. The output file was not created.")
        print("Please check your sampling parameters and the content of your dataset.")

    print(f"\nLog saved to {args.log}")

if __name__ == "__main__":
    main()
