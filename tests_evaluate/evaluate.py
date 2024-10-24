import json
import difflib
import argparse
import statistics
import sys

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

def calculate_diff(data, limit=None, use_simple_template=False):
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

        # Append both results
        results.append({
            'total_diff': total_diff,
            'added_lines': added,
            'removed_lines': removed,
            'total_diff_sorted': total_diff_sorted,
            'added_lines_sorted': added_sorted,
            'removed_lines_sorted': removed_sorted
        })

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

def main():
    parser = argparse.ArgumentParser(description="Calculate diff between final_code and generated_text.")
    parser.add_argument("input_files", nargs='+', help="Paths to the input JSON files")
    parser.add_argument("--output_file", help="Path to the output JSON file (optional)")
    parser.add_argument("-n", type=int, help="Number of examples to process (optional)")
    parser.add_argument("--simple", action="store_true", help="Use simple template without tags")
    args = parser.parse_args()

    all_results = {}
    for input_file in args.input_files:
        try:
            with open(input_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {input_file}: {e}", file=sys.stderr)
            continue

        results = calculate_diff(data, limit=args.n, use_simple_template=args.simple)
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
