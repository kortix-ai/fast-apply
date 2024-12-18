import argparse
import json
import csv
import sys
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser(description='Convert JSON to CSV or JSONL with prompt processing.')
    parser.add_argument('-i', '--input', required=True, help='Path to input JSON file')
    parser.add_argument('-o', '--output', required=True, help='Path to output file (CSV or JSONL)')
    parser.add_argument('-m', '--max-output-chars', type=int, default=None, help='Maximum number of characters in the output (default: no limit)')
    parser.add_argument('-s', '--simple', action='store_true', help='Use simple template for prompt input')
    parser.add_argument('-f', '--format', choices=['csv', 'jsonl'], default='csv', help='Output format (csv or jsonl)')
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

def format_prompt_input(update_snippet, original_code, use_simple_template=False):
    if use_simple_template:
        return f"""<update>{update_snippet}</update>\n<code>{original_code}</code>"""
    else:
        return f"""Merge all changes from the update snippet to the code below, ensuring that every modification is fully integrated. 
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

def format_prompt_output(final_code):
    prompt_output = f"""<updated-code>{final_code}</updated-code>"""
    return prompt_output

def write_output(data, output_path, max_output_chars=None, use_simple_template=False, output_format='csv'):
    try:
        if output_format == 'csv':
            return write_csv(data, output_path, max_output_chars, use_simple_template)
        elif output_format == 'jsonl':
            return write_jsonl(data, output_path, max_output_chars, use_simple_template)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    except Exception as e:
        print(f"Error writing output file: {e}")
        sys.exit(1)

def write_csv(data, output_path, max_output_chars=None, use_simple_template=False):
    processed_lines = 0
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Input', 'Output'], quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for item in data:
            row = process_item(item, max_output_chars, use_simple_template)
            if row:
                writer.writerow(row)
                processed_lines += 1
    return processed_lines

def write_jsonl(data, output_path, max_output_chars=None, use_simple_template=False):
    processed_lines = 0
    with open(output_path, 'w', encoding='utf-8') as jsonl_file:
        for item in data:
            row = process_item(item, max_output_chars, use_simple_template)
            if row:
                jsonl_entry = {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": row['Input']}]
                        },
                        {
                            "role": "model",
                            "parts": [{"text": row['Output']}]
                        }
                    ]
                }
                json.dump(jsonl_entry, jsonl_file)
                jsonl_file.write('\n')
                processed_lines += 1
    return processed_lines

def process_item(item, max_output_chars=None, use_simple_template=False):
    update_snippet = item.get('update_snippet', '')
    original_code = item.get('original_code', '')
    final_code = item.get('final_code', '')

    input_text = format_prompt_input(update_snippet, original_code, use_simple_template)
    output_text = format_prompt_output(final_code)

    if max_output_chars is None or len(output_text) <= max_output_chars:
        return {'Input': input_text, 'Output': output_text}
    return None

def format_prompt_output(final_code):
    return final_code

def main():
    args = parse_arguments()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        print(f"Input file does not exist: {input_path}")
        sys.exit(1)

    data = load_json(input_path)
    processed_lines = write_output(data, output_path, args.max_output_chars, args.simple, args.format)
    print(f"Successfully converted {input_path} to {output_path}")
    print(f"Total lines processed: {processed_lines}")

if __name__ == "__main__":
    main()
