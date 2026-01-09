#!/usr/bin/env python3
"""
Automatically apply all n/a support modifications to remaining run_*.py files.
This script reads the completed files and applies the same changes to the remaining ones.
"""

import os
import re
import sys

def read_helper_functions():
    """Extract helper functions from completed file"""
    with open('run_text_demo_nothink.py', 'r') as f:
        content = f.read()

    # Extract calculate_false_positives and calculate_voc_metrics
    match = re.search(
        r'(def calculate_false_positives.*?return ref_fp, score_fp\n\n\n'
        r'def calculate_voc_metrics.*?return \{\n.*?\'voc_values\': \[\]\n.*?\})',
        content,
        re.DOTALL
    )

    if match:
        return '\n\n' + match.group(1) + '\n\n'
    return None

def apply_modifications(filepath):
    """Apply all modifications to a file"""
    print(f"\nProcessing: {os.path.basename(filepath)}")

    with open(filepath, 'r') as f:
        content = f.read()

    original = content
    changes = []

    # 1. Update imports
    if 'from scipy.stats import spearmanr' not in content:
        content = content.replace(
            'from typing import List, Dict, Any, Optional',
            'from typing import List, Dict, Any, Optional, Tuple'
        )
        content = content.replace(
            'from multiprocessing import Manager, Process, Queue\n',
            'from multiprocessing import Manager, Process, Queue\nfrom scipy.stats import spearmanr\nimport numpy as np\n'
        )
        changes.append("imports")

    # 2. Update ref parsing - add n/a check
    ref_old = r'(# Extract ref \(now expects integer 1-based (?:index|image number)\))\n(\s+ref_match = re\.search\(r\'<ref>\(\.\*\?\)</ref>\', response, re\.DOTALL\)\n)(\s+if ref_match:\n)(\s+ref_str = ref_match\.group\(1\)\.strip\(\)\n)(\s+try:\n)'

    if re.search(ref_old, content) and 'if ref_str.lower() in ["n/a"' not in content:
        content = re.sub(
            ref_old,
            r'\1 or "n/a")\n\2\3\4            # Check for "n/a" first\n            if ref_str.lower() in ["n/a", "na"]:\n                result[\'ref\'] = "n/a"\n            else:\n\5',
            content
        )
        # Fix indentation for the else block
        content = re.sub(
            r'(if ref_str\.lower\(\) in \["n/a".*?\n.*?else:\n\s+try:\n)(                # Extract)',
            r'\1    \2',
            content
        )
        content = re.sub(
            r'(else:\n\s+try:\n\s+# Extract.*?\n)(                ref_num)',
            r'\1    \2',
            content
        )
        content = re.sub(
            r'(else:\n\s+try:\n.*?\n\s+ref_num.*?\n)(                if ref_num:)',
            r'\1    \2',
            content
        )
        content = re.sub(
            r'(else:\n\s+try:\n.*?\n.*?\n\s+if ref_num:\n)(                    result\[)',
            r'\1        \2',
            content
        )
        content = re.sub(
            r'(else:\n\s+try:\n.*?\n.*?\n.*?\n.*?\n)(                else:)',
            r'\1    \2',
            content
        )
        content = re.sub(
            r'(else:\n\s+try:\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n)(                    result\[)',
            r'\1        \2',
            content
        )
        content = re.sub(
            r'(else:\n\s+try:\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n)(            except)',
            r'\1    \2',
            content
        )
        content = re.sub(
            r'(else:\n\s+try:\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n)(                result\[)',
            r'\1        \2',
            content
        )
        changes.append("ref_parsing")

    # 3. Update score parsing - add n/a check
    score_old = r'(# Extract score \(supports (?:both )?"(?:\d+%|8%)" (?:and|or) "0\.\d+"\))\n(\s+score_match = re\.search\(r\'<score>\(\.\*\?\)</score>\', response, re\.DOTALL\)\n)(\s+if score_match:\n)(\s+score_str = score_match\.group\(1\)\.strip\(\)\n)(\s+try:\n)'

    if re.search(score_old, content) and 'if score_str.lower() in ["n/a"' not in content:
        content = re.sub(
            score_old,
            r'\1, or "n/a")\n\2\3\4            # Check for "n/a" first\n            if score_str.lower() in ["n/a", "na"]:\n                result[\'score\'] = "n/a"\n            else:\n\5',
            content
        )
        # Fix indentation
        content = re.sub(
            r'(if score_str\.lower\(\) in \["n/a".*?\n.*?else:\n\s+try:\n)(                # Remove % sign)',
            r'\1    \2',
            content
        )
        content = re.sub(
            r'(else:\n\s+try:\n\s+# Remove.*?\n)(                if score_str\.endswith)',
            r'\1    \2',
            content
        )
        content = re.sub(
            r'(else:\n\s+try:\n.*?\n\s+if score_str\.endswith.*?\n)(                    score_value = float)',
            r'\1        \2',
            content
        )
        content = re.sub(
            r'(else:\n\s+try:\n.*?\n.*?\n.*?\n)(                else:)',
            r'\1    \2',
            content
        )
        content = re.sub(
            r'(else:\n\s+try:\n.*?\n.*?\n.*?\n.*?\n)(                    score_value = float\(score_str\))',
            r'\1        \2',
            content
        )
        content = re.sub(
            r'(else:\n\s+try:\n.*?\n.*?\n.*?\n.*?\n.*?\n)(                    # If > 1\.0)',
            r'\1        \2',
            content
        )
        content = re.sub(
            r'(else:\n\s+try:\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n)(                    if score_value > 1\.0:)',
            r'\1        \2',
            content
        )
        content = re.sub(
            r'(else:\n\s+try:\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n)(                        score_value = score_value / 100)',
            r'\1            \2',
            content
        )
        content = re.sub(
            r'(else:\n\s+try:\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n)(                # Clamp)',
            r'\1        \2',
            content
        )
        content = re.sub(
            r'(else:\n\s+try:\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n)(                result\[.score.\] = max)',
            r'\1        \2',
            content
        )
        content = re.sub(
            r'(else:\n\s+try:\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n.*?\n)(            except ValueError:)',
            r'\1    \2',
            content
        )
        changes.append("score_parsing")

    # 4. Add helper functions if not present
    if 'def calculate_false_positives' not in content:
        helper_funcs = read_helper_functions()
        if helper_funcs:
            # Find insertion point after calculate_ref_error
            pattern = r'(def calculate_ref_error.*?return float\(absolute_error\))\n\n\ndef worker_process'
            if re.search(pattern, content, re.DOTALL):
                content = re.sub(
                    pattern,
                    r'\1' + helper_funcs + '\ndef worker_process',
                    content,
                    flags=re.DOTALL
                )
                changes.append("helper_functions")

    # Write back if changes were made
    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  ✓ Applied: {', '.join(changes)}")
        return True
    else:
        print(f"  - No changes needed or already updated")
        return False

def main():
    """Main function"""
    files_to_process = [
        'run_text_demo.py',
        'run_visual_demo.py',
        'run_text_demo_single.py',
        'run_visual_demo_single.py',
        'run_text_demo_72B_nothink.py',
        'run_visual_demo_72B_nothink.py',
    ]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    success_count = 0
    for filename in files_to_process:
        if os.path.exists(filename):
            try:
                if apply_modifications(filename):
                    success_count += 1
            except Exception as e:
                print(f"  ✗ Error: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"\n✗ File not found: {filename}")

    print(f"\n{'='*70}")
    print(f"Phase 1 complete: {success_count}/{len(files_to_process)} files updated")
    print(f"{'='*70}")
    print("\nNote: Phase 1 only updates parsing and adds helper functions.")
    print("You still need to manually update for each file:")
    print("  - worker_process result building")
    print("  - Error handling blocks")
    print("  - tqdm monitoring")
    print("  - Summary statistics")
    print("\nThese require more context-specific changes.")
    print("Refer to run_text_demo_nothink.py for the complete pattern.")

if __name__ == '__main__':
    main()
