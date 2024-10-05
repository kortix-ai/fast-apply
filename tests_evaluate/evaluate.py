import json
import xml.etree.ElementTree as ET
import argparse
import statistics
import difflib

def count_diff_lines(S1, S2):
    """
    Counts the number of differing lines between two multi-line strings S1 and S2
    using the unified diff algorithm.

    Parameters:
    - S1: First multi-line string
    - S2: Second multi-line string

    Returns:
    - Number of differing lines
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
        #  print(line)
        if line.startswith('+') and not line.startswith('+++'):
            added += 1
        elif line.startswith('-') and not line.startswith('---'):
            removed += 1

    total_diff = added + removed
    return total_diff, added, removed

def parse_generated_text(text):
    """Parse the generated text to extract the code within <updated-code> or <update-code> tags."""
    #  print(text)
    try:
        tag_pairs = [
            ("<updated-code>", "</updated-code>"),
            ("<updated-code>", "<updated-code>")
        ]
        
        for start_tag, end_tag in tag_pairs:
            start_index = text.find(start_tag)
            end_index = text.rfind(end_tag)
            
            if start_index != -1 and end_index != -1:
                extracted_code = text[start_index + len(start_tag):end_index].strip()
                if extracted_code:
                    return extracted_code
        
        # If no tags found or empty content, return the original text
        return text.strip()
    except Exception as e:
        print(f"Error parsing generated text: {e}")
        return text.strip()

def calculate_diff(input_file):
    """Calculate the diff between final_code and generated_text for each entry."""
    with open(input_file, 'r') as f:
        data = json.load(f)

    results = []
    for entry in data:
        final_code = entry['final_code']
        generated_text = parse_generated_text(entry['generated_text'])
        
        total_diff, added, removed = count_diff_lines(final_code, generated_text)
        
        results.append({
            'total_diff': total_diff,
            'added_lines': added,
            'removed_lines': removed
        })

    return results

def calculate_accuracy(results):
    """Calculate the accuracy score based on fully corrected examples."""
    fully_corrected = sum(1 for r in results if r['total_diff'] == 0)
    total_examples = len(results)
    return fully_corrected / total_examples if total_examples > 0 else 0

def print_statistics(all_results):
    """Print statistics for all input files."""
    for file_name, results in all_results.items():
        total_diffs = [r['total_diff'] for r in results]
        added_lines = [r['added_lines'] for r in results]
        removed_lines = [r['removed_lines'] for r in results]
        accuracy = calculate_accuracy(results)

        print(f"\nStatistics for {file_name}:")
        print(f"Total entries: {len(results)}")
        print(f"Average total diff: {statistics.mean(total_diffs):.2f}")
        print(f"Median total diff: {statistics.median(total_diffs):.2f}")
        print(f"Average added lines: {statistics.mean(added_lines):.2f}")
        print(f"Average removed lines: {statistics.mean(removed_lines):.2f}")
        print(f"Accuracy score: {accuracy:.2%}")

def main():
    parser = argparse.ArgumentParser(description="Calculate diff between final_code and generated_text.")
    parser.add_argument("input_files", nargs='+', help="Paths to the input JSON files")
    parser.add_argument("--output_file", help="Path to the output JSON file (optional)")
    parser.add_argument("-n", type=int, help="Number of examples to process (optional)")
    args = parser.parse_args()

    all_results = {}
    for input_file in args.input_files:
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        if args.n:
            data = data[:args.n]
        
        results = []
        for entry in data:
            final_code = entry['final_code']
            generated_text = parse_generated_text(entry['generated_text'])
            
            total_diff, added, removed = count_diff_lines(final_code, generated_text)
            
            results.append({
                'total_diff': total_diff,
                'added_lines': added,
                'removed_lines': removed
            })
        
        all_results[input_file] = results

    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Diff results saved to {args.output_file}")
    
    for input_file in args.input_files:
        print(f"\nStatistics for {input_file}:")
        print_statistics({input_file: all_results[input_file]})

if __name__ == "__main__":
    main()
