import json
import difflib
import argparse
import statistics
import sys
import os
import openai

def count_diff_lines(S1, S2):
    """
    Counts the number of differing lines between two multi-line strings S1 and S2
    using the unified diff algorithm.

    Parameters:
    - S1: First multi-line string
    - S2: Second multi-line string

    Returns:
    - Number of differing lines, added lines, removed lines
    """
    # Split the input strings into lists of lines
    lines1 = S1.strip().splitlines()
    lines2 = S2.strip().splitlines()

    # Generate the unified diff
    diff = difflib.unified_diff(lines1, lines2, lineterm='')

    # Initialize counters
    added = 0
    removed = 0

    # Iterate over the diff output
    for line in diff:
        if line.startswith('+') and not line.startswith('+++'):
            added += 1
        elif line.startswith('-') and not line.startswith('---'):
            removed += 1

    total_diff = added + removed
    return total_diff, added, removed

def parse_generated_text(text, use_simple_template=False):
    """
    Parse the generated text to extract the code within <updated-code> or <update-code> tags,
    or return the entire text if using the simple template.

    Parameters:
    - text: The generated text containing code within specific tags or the entire updated code
    - use_simple_template: Boolean indicating whether the simple template was used

    Returns:
    - Extracted code as a string
    """
    if use_simple_template:
        return text.strip()
    
    try:
        tag_pairs = [
            ("<updated-code>", "</updated-code>"),
            ("<update-code>", "</update-code>")
        ]

        for start_tag, end_tag in tag_pairs:
            start_index = text.find(start_tag)
            end_index = text.find(end_tag, start_index + len(start_tag))

            if start_index != -1 and end_index != -1:
                extracted_code = text[start_index + len(start_tag):end_index].strip()
                if extracted_code:
                    return extracted_code

        # If no tags found or empty content, return the original text
        return text.strip()
    except Exception as e:
        print(f"Error parsing generated text: {e}", file=sys.stderr)
        return text.strip()

def calculate_diff(data, limit=None, use_simple_template=False, deepseek_activated=False):
    """
    Calculate the diff between final_code and generated_text for each entry.
    Additionally, calculate diffs based on sorted lines for an alternative accuracy metric.

    Parameters:
    - data: List of entries containing 'final_code' and 'generated_text'
    - limit: Optional limit on the number of entries to process
    - use_simple_template: Boolean indicating whether the simple template was used

    Returns:
    - List of dictionaries with original and sorted diff results
    """
    if limit is not None:
        data = data[:limit]

    results = []
    for entry in data:
        final_code = entry.get('final_code', '')
        generated_text = parse_generated_text(entry.get('generated_text', ''), use_simple_template)

        # Original diff
        total_diff, added, removed = count_diff_lines(final_code, generated_text)

        # Sorted diff
        final_lines_sorted = sorted(final_code.strip().splitlines())
        generated_lines_sorted = sorted(generated_text.strip().splitlines())
        generated_text_sorted = '\n'.join(generated_lines_sorted)
        final_text_sorted = '\n'.join(final_lines_sorted)

        total_diff_sorted, added_sorted, removed_sorted = count_diff_lines(final_text_sorted, generated_text_sorted)

        # Create result dictionary
        result = {
            'total_diff': total_diff,
            'added_lines': added,
            'removed_lines': removed,
            'total_diff_sorted': total_diff_sorted,
            'added_lines_sorted': added_sorted,
            'removed_lines_sorted': removed_sorted
        }

        # Add DeepSeek evaluation if activated and there are differences
        if total_diff > 0 and deepseek_activated:
            deepseek_result = evaluate_with_deepseek(entry)
            if deepseek_result:
                result['deepseek_score'] = deepseek_result.get('score')
                result['deepseek_analysis'] = deepseek_result.get('analysis')

        results.append(result)

    return results

def calculate_accuracy(results, key='total_diff'):
    """
    Calculate the accuracy score based on fully corrected examples.

    Parameters:
    - results: List of dictionaries with diff results
    - key: The key to determine correctness ('total_diff' or 'total_diff_sorted')

    Returns:
    - Accuracy score as a float
    """
    fully_corrected = sum(1 for r in results if r.get(key, 0) == 0)
    total_examples = len(results)
    return fully_corrected / total_examples if total_examples > 0 else 0

def evaluate_with_deepseek(entry):
    """
    Evaluate mismatched code using DeepSeek API.
    
    Parameters:
    - entry: Dictionary containing original_code, update_snippet, and final_code
    
    Returns:
    - Dictionary containing score and analysis from DeepSeek
    """
    original_code = entry.get('original_code', '')
    update_snippet = entry.get('update_snippet', '')
    final_code = entry.get('final_code', '')

    prompt = """
You are an AI code evaluator. You will be provided with:

- Original code
- Update snippet
- Final code

Your task is to:

1. Analyze whether the final code correctly applies the update snippet to the original code.
2. Check for any bugs or issues in the final code.
3. Provide a brief analysis of any issues found.
4. Give a score between 0 and 1 (1 means the final code correctly applies the update snippet and is bug-free).

Output:

- Your analysis enclosed within <analysis></analysis> tags.
- The score enclosed within <score></score> tags.

Do not include any other text.
""".strip()

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"<original_code>\n{original_code}\n</original_code>\n<update_snippet>\n{update_snippet}\n</update_snippet>\n<final_code>\n{final_code}\n</final_code>"}
    ]

    try:
        response = openai.ChatCompletion.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0,
            max_tokens=1000
        )
        content = response['choices'][0]['message']['content']

        # Parse the content to extract <score> and <analysis>
        score = None
        analysis = None
        if '<score>' in content and '</score>' in content:
            score = float(content.split('<score>')[1].split('</score>')[0].strip())
        if '<analysis>' in content and '</analysis>' in content:
            analysis = content.split('<analysis>')[1].split('</analysis>')[0].strip()
        return {'score': score, 'analysis': analysis}
    except Exception as e:
        print(f"Error during DeepSeek evaluation: {e}", file=sys.stderr)
        return None

def print_statistics(file_name, results):
    """
    Print statistics for a single input file, including both original and sorted accuracy.

    Parameters:
    - file_name: Name of the input file
    - results: List of dictionaries with diff results
    """
    # Original diffs
    total_diffs = [r['total_diff'] for r in results]
    added_lines = [r['added_lines'] for r in results]
    removed_lines = [r['removed_lines'] for r in results]
    accuracy_original = calculate_accuracy(results, key='total_diff')

    # Sorted diffs
    total_diffs_sorted = [r['total_diff_sorted'] for r in results]
    added_lines_sorted = [r['added_lines_sorted'] for r in results]
    removed_lines_sorted = [r['removed_lines_sorted'] for r in results]
    accuracy_sorted = calculate_accuracy(results, key='total_diff_sorted')

    print(f"\nStatistics for {file_name}:")
    print(f"Total entries: {len(results)}")

    # Original diffs statistics
    if total_diffs:
        print("\nOriginal Diff Statistics:")
        print(f"  Average total diff: {statistics.mean(total_diffs):.2f}")
        print(f"  Median total diff: {statistics.median(total_diffs):.2f}")
    else:
        print("\nOriginal Diff Statistics:")
        print("  No diffs to calculate statistics.")

    if added_lines:
        print(f"  Average added lines: {statistics.mean(added_lines):.2f}")
    else:
        print("  No added lines to calculate statistics.")

    if removed_lines:
        print(f"  Average removed lines: {statistics.mean(removed_lines):.2f}")
    else:
        print("  No removed lines to calculate statistics.")

    print(f"  Accuracy score: {accuracy_original:.2%}")

    # Sorted diffs statistics
    if total_diffs_sorted:
        print("\nSorted Diff Statistics:")
        print(f"  Average total diff (sorted): {statistics.mean(total_diffs_sorted):.2f}")
        print(f"  Median total diff (sorted): {statistics.median(total_diffs_sorted):.2f}")
    else:
        print("\nSorted Diff Statistics:")
        print("  No diffs to calculate statistics.")

    if added_lines_sorted:
        print(f"  Average added lines (sorted): {statistics.mean(added_lines_sorted):.2f}")
    else:
        print("  No added lines to calculate statistics.")

    if removed_lines_sorted:
        print(f"  Average removed lines (sorted): {statistics.mean(removed_lines_sorted):.2f}")
    else:
        print("  No removed lines to calculate statistics.")

    print(f"  Sorted Accuracy score: {accuracy_sorted:.2%}")

    # Print DeepSeek evaluation results if available
    deepseek_scores = [r.get('deepseek_score') for r in results if 'deepseek_score' in r]
    if deepseek_scores:
        print("\nDeepSeek Evaluation Results:")
        print(f"  Average DeepSeek score: {statistics.mean(deepseek_scores):.2f}")
        print(f"  Median DeepSeek score: {statistics.median(deepseek_scores):.2f}")

def main():
    parser = argparse.ArgumentParser(description="Calculate diff between final_code and generated_text.")
    parser.add_argument("input_files", nargs='+', help="Paths to the input JSON files")
    parser.add_argument("--output_file", help="Path to the output JSON file (optional)")
    parser.add_argument("-n", type=int, help="Number of examples to process (optional)")
    parser.add_argument("--simple", action="store_true", help="Use simple template without tags")
    parser.add_argument("--deepseek", action="store_true", help="Use DeepSeek for evaluation of mismatched code")
    args = parser.parse_args()

    all_results = {}
    # Configure DeepSeek if enabled
    if args.deepseek:
        openai.api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not openai.api_key:
            print("Error: DEEPSEEK_API_KEY environment variable not set", file=sys.stderr)
            sys.exit(1)
        openai.api_base = "https://api.deepseek.com/beta"

    for input_file in args.input_files:
        try:
            with open(input_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {input_file}: {e}", file=sys.stderr)
            continue

        results = calculate_diff(data, limit=args.n, use_simple_template=args.simple, 
                               deepseek_activated=args.deepseek)
        all_results[input_file] = results

    if args.output_file:
        try:
            with open(args.output_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\nDiff results saved to {args.output_file}")
        except Exception as e:
            print(f"Error writing to {args.output_file}: {e}", file=sys.stderr)

    for input_file, results in all_results.items():
        print_statistics(input_file, results)

if __name__ == "__main__":
    main()
