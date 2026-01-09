#!/usr/bin/env python3
"""
检测JSONL文件中ref/closest_idx和score/ground_truth_score不一致或包含n/a的样本
可选择删除这些不一致的样本

用法:
    python check_consistency.py <jsonl_file> [--remove]

参数:
    jsonl_file: 要检查的JSONL文件路径
    --remove: 可选，如果指定则删除不一致的样本
"""

import json
import argparse
import sys
from pathlib import Path


def check_consistency(file_path: str, remove: bool = False) -> dict:
    """
    检查JSONL文件中的一致性问题

    Args:
        file_path: JSONL文件路径
        remove: 是否删除不一致的样本

    Returns:
        包含检查结果的字典
    """
    valid_lines = []
    inconsistent_samples = []
    total_lines = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                ref = data.get('ref')
                closest_idx = data.get('closest_idx')
                score = data.get('score')
                ground_truth_score = data.get('ground_truth_score')

                # 检查是否有n/a
                has_na = any(val == 'n/a' for val in [ref, closest_idx, score, ground_truth_score])

                # 检查不一致
                ref_mismatch = ref != closest_idx
                score_mismatch = score != ground_truth_score

                if has_na or ref_mismatch or score_mismatch:
                    inconsistent_samples.append({
                        'line': line_num,
                        'ref': ref,
                        'closest_idx': closest_idx,
                        'score': score,
                        'ground_truth_score': ground_truth_score,
                        'has_na': has_na,
                        'ref_mismatch': ref_mismatch,
                        'score_mismatch': score_mismatch
                    })
                else:
                    valid_lines.append(line + '\n')

            except json.JSONDecodeError as e:
                inconsistent_samples.append({
                    'line': line_num,
                    'error': f'JSON decode error: {e}'
                })

    # 如果需要删除不一致样本
    if remove and inconsistent_samples:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(valid_lines)

    return {
        'file_path': file_path,
        'total_lines': total_lines,
        'valid_count': len(valid_lines),
        'inconsistent_count': len(inconsistent_samples),
        'inconsistent_samples': inconsistent_samples,
        'removed': remove
    }


def print_report(result: dict):
    """打印检查报告"""
    print("=" * 80)
    print(f"文件: {result['file_path']}")
    print("=" * 80)
    print(f"总行数: {result['total_lines']}")
    print(f"有效样本数: {result['valid_count']}")
    print(f"不一致样本数: {result['inconsistent_count']}")

    if result['removed']:
        print(f"\n已删除 {result['inconsistent_count']} 个不一致样本")

    if result['inconsistent_samples']:
        print("\n不一致样本详情:")
        print("-" * 80)
        for sample in result['inconsistent_samples']:
            print(f"行号: {sample['line']}")
            if 'error' in sample:
                print(f"  错误: {sample['error']}")
            else:
                issues = []
                if sample.get('has_na'):
                    issues.append("包含n/a")
                if sample.get('ref_mismatch'):
                    issues.append(f"ref({sample['ref']}) != closest_idx({sample['closest_idx']})")
                if sample.get('score_mismatch'):
                    issues.append(f"score({sample['score']}) != ground_truth_score({sample['ground_truth_score']})")
                print(f"  问题: {'; '.join(issues)}")
            print()
    else:
        print("\n所有样本一致，无问题!")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='检测JSONL文件中ref/closest_idx和score/ground_truth_score的一致性'
    )
    parser.add_argument('jsonl_file', help='要检查的JSONL文件路径')
    parser.add_argument('--remove', action='store_true',
                        help='删除不一致的样本')

    args = parser.parse_args()

    if not Path(args.jsonl_file).exists():
        print(f"错误: 文件不存在 - {args.jsonl_file}")
        sys.exit(1)

    result = check_consistency(args.jsonl_file, args.remove)
    print_report(result)

    # 返回状态码: 0=全部一致, 1=存在不一致
    sys.exit(0 if result['inconsistent_count'] == 0 else 1)


if __name__ == '__main__':
    main()
