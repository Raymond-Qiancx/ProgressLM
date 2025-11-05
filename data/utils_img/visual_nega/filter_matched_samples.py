import json

def filter_matched_samples(input_file, output_file):
    """
    过滤掉 id 中间部分和 stage_to_estimate 中包含相同字符串的样本
    """
    total_count = 0
    filtered_count = 0
    kept_count = 0
    error_count = 0

    print("开始过滤文件...")
    print(f"输入文件：{input_file}")
    print(f"输出文件：{output_file}")
    print("="*60)

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                total_count += 1

                # 获取 id 和 stage_to_estimate
                sample_id = data.get('id', '')
                stage_images = data.get('stage_to_estimate', [])

                if not sample_id or not stage_images:
                    # 如果缺少字段，保留样本
                    f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                    kept_count += 1
                    continue

                # 提取 id 中间部分
                id_parts = sample_id.split('/')
                if len(id_parts) >= 2:
                    middle_part = id_parts[1] if len(id_parts) == 3 else '/'.join(id_parts[1:-1])
                else:
                    # 如果 id 格式不符合预期，保留样本
                    f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                    kept_count += 1
                    continue

                # 检查 stage_to_estimate 中是否包含 middle_part
                has_match = False
                for img_name in stage_images:
                    if middle_part in img_name:
                        has_match = True
                        break

                if has_match:
                    # 匹配的样本，过滤掉（不写入输出文件）
                    filtered_count += 1
                    if filtered_count <= 5:
                        print(f"[过滤] 第 {line_num} 行: id={sample_id}")
                else:
                    # 不匹配的样本，保留
                    f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                    kept_count += 1

                # 每处理1000个样本打印一次进度
                if total_count % 1000 == 0:
                    print(f"[进度] 已处理 {total_count} 个样本，保留 {kept_count} 个，过滤 {filtered_count} 个...")

            except json.JSONDecodeError as e:
                error_count += 1
                print(f"[错误] 第 {line_num} 行 JSON 解析失败: {e}")
                continue
            except Exception as e:
                error_count += 1
                print(f"[错误] 第 {line_num} 行处理失败: {e}")
                continue

    # 打印统计结果
    print("\n" + "="*60)
    print("过滤结果")
    print("="*60)
    print(f"原始样本数：{total_count}")
    print(f"保留样本数：{kept_count}")
    print(f"过滤样本数：{filtered_count}")
    print(f"错误样本数：{error_count}")
    print(f"保留比例：{kept_count/total_count*100:.2f}%" if total_count > 0 else "N/A")
    print(f"过滤比例：{filtered_count/total_count*100:.2f}%" if total_count > 0 else "N/A")
    print("="*60)
    print("过滤完成！")


if __name__ == '__main__':
    input_file = '/projects/p32958/chengxuan/ProgressLM/data/train/visual_demo/visual_negative_trans_img_replaced_processed.jsonl'
    output_file = '/projects/p32958/chengxuan/ProgressLM/data/train/visual_demo/visual_negative_trans_img_replaced_filtered.jsonl'

    filter_matched_samples(input_file, output_file)
