import argparse
import json
import os
from collections import defaultdict

def split_jsonl_by_task_type(input_jsonl, output_dir):
    """
    Splits a JSONL file into multiple JSONL files based on task_type.

    Args:
        input_jsonl (str): Path to the input JSONL file.
        output_dir (str): Path to the directory where output files will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_handlers = {}

    try:
        with open(input_jsonl, 'r') as f_in:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON line: {line}")
                    continue

                if 'id' not in data:
                    print(f"Warning: Skipping line with no 'id' field: {line}")
                    continue

                parts = data['id'].split('/')
                if len(parts) < 2:
                    print(f"Warning: Skipping line with unexpected ID format: {data['id']}")
                    continue

                task_type = '/'.join(parts[:-1])
                sanitized_task_type = task_type.replace('/', '_')
                output_filename = f"{sanitized_task_type}.jsonl"
                output_path = os.path.join(output_dir, output_filename)

                if output_path not in file_handlers:
                    file_handlers[output_path] = open(output_path, 'w')

                file_handlers[output_path].write(line + '\n')

        print(f"Successfully split {input_jsonl} into {len(file_handlers)} files in {output_dir}")

    finally:
        for handler in file_handlers.values():
            handler.close()

def main():
    parser = argparse.ArgumentParser(description="Split a JSONL file by task_type.")
    parser.add_argument('--input_jsonl', type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the split JSONL files.")
    args = parser.parse_args()

    split_jsonl_by_task_type(args.input_jsonl, args.output_dir)

if __name__ == '__main__':
    main()
