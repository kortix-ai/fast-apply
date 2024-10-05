import argparse
import json
import csv
import sys
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser(description='Convert JSON to CSV with prompt processing.')
    parser.add_argument('-i', '--input', required=True, help='Path to input JSON file')
    parser.add_argument('-o', '--output', required=True, help='Path to output CSV file')
    parser.add_argument('-m', '--max-output-chars', type=int, default=None, help='Maximum number of characters in the output (default: no limit)')
    return parser.parse_args()

def load_json(input_path):
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON file must contain a list of objects.")
            return data
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        sys.exit(1)

def format_prompt_input(update_snippet, original_code):
    prompt_input = f"""Merge all changes from the update snippet to the code below, ensuring that every modification is fully integrated. 
Maintain the code's structure, order, comments, and indentation precisely. 
Do not use any placeholders, ellipses, or omit any sections in <updated-code>.
Only output the updated code; do not include any additional text, explanations, or fences.
\n
<update>{update_snippet}</update>
\n
<code>{original_code}</code>
\n
The updated code MUST be enclosed in <updated-code> tags.
Here's the updated-code with fully integrated changes, start the tag now:
"""
    return prompt_input

def format_prompt_output(final_code):
    prompt_output = f"""<updated-code>{final_code}</updated-code>"""
    return prompt_output

def write_csv(data, output_path, max_output_chars=None):
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['Input', 'Output'], quoting=csv.QUOTE_ALL)
            writer.writeheader()
            for item in data:
                update_snippet = item.get('update_snippet', '')
                original_code = item.get('original_code', '')
                final_code = item.get('final_code', '')

                input_text = format_prompt_input(update_snippet, original_code)
                output_text = format_prompt_output(final_code)

                if max_output_chars is None or len(output_text) <= max_output_chars:
                    writer.writerow({'Input': input_text, 'Output': output_text})
    except Exception as e:
        print(f"Error writing CSV file: {e}")
        sys.exit(1)

def main():
    args = parse_arguments()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        print(f"Input file does not exist: {input_path}")
        sys.exit(1)

    data = load_json(input_path)
    write_csv(data, output_path, args.max_output_chars)
    print(f"Successfully converted {input_path} to {output_path}")

if __name__ == "__main__":
    main()
