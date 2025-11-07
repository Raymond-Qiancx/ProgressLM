#!/usr/bin/env python3
"""
修复 JSONL 文件中 stage_to_estimate 字段类型不一致的问题
将所有字符串类型的 stage_to_estimate 转换为数组类型
"""

import json
import sys

def fix_jsonl_file(input_file, output_file):
    """修复JSONL文件中的数据类型不一致问题"""
    fixed_count = 0
    total_count = 0

    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:

        for line_num, line in enumerate(fin, 1):
            try:
                data = json.loads(line)
                total_count += 1

                # 检查 stage_to_estimate 字段
                if 'stage_to_estimate' in data:
                    stage_val = data['stage_to_estimate']

                    # 如果是字符串，转换为数组
                    if isinstance(stage_val, str):
                        data['stage_to_estimate'] = [stage_val]
                        fixed_count += 1
                        if fixed_count <= 5:  # 只打印前5个修复的行
                            print(f"行 {line_num}: 修复 '{stage_val}' -> ['{stage_val}']")
                    elif not isinstance(stage_val, list):
                        print(f"警告：行 {line_num} 的 stage_to_estimate 既不是字符串也不是数组: {type(stage_val)}")

                # 写入修复后的数据
                fout.write(json.dumps(data, ensure_ascii=False) + '\n')

            except json.JSONDecodeError as e:
                print(f"错误：行 {line_num} JSON 解析失败: {e}")
                sys.exit(1)

    print(f"\n修复完成！")
    print(f"总行数: {total_count}")
    print(f"修复行数: {fixed_count}")
    print(f"输出文件: {output_file}")

if __name__ == '__main__':
    input_file = '/projects/p32958/chengxuan/ProgressLM/data/train/rl/new/new_rl_35k_final.jsonl'
    output_file = '/projects/p32958/chengxuan/ProgressLM/data/train/rl/new/new_rl_35k_final_fixed.jsonl'

    print(f"开始修复文件: {input_file}")
    print(f"输出文件: {output_file}\n")

    fix_jsonl_file(input_file, output_file)
