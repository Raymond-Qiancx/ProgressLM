import json

def check_id_stage_match(jsonl_file):
    """
    检测 id 中间部分和 stage_to_estimate 中是否包含相同字符串的样本数量
    """
    match_count = 0
    total_count = 0
    match_samples = []

    print("开始检测文件...")
    print("="*60)

    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
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
                    continue

                # 提取 id 中间部分
                # 例如：h5_ur_1rgb/put_yellow_pepper_in_top_drawer/1017_180145
                # 分割后：['h5_ur_1rgb', 'put_yellow_pepper_in_top_drawer', '1017_180145']
                # 中间部分：put_yellow_pepper_in_top_drawer
                id_parts = sample_id.split('/')
                if len(id_parts) >= 2:
                    # 取中间部分（跳过第一个和最后一个）
                    middle_part = id_parts[1] if len(id_parts) == 3 else '/'.join(id_parts[1:-1])
                else:
                    continue

                # 检查 stage_to_estimate 中的每个图片名是否包含 middle_part
                has_match = False
                for img_name in stage_images:
                    if middle_part in img_name:
                        has_match = True
                        break

                if has_match:
                    match_count += 1
                    if match_count <= 10:  # 保存前10个匹配的样本
                        match_samples.append({
                            'line': line_num,
                            'id': sample_id,
                            'middle_part': middle_part,
                            'stage_to_estimate': stage_images
                        })

                # 每处理1000个样本打印一次进度
                if total_count % 1000 == 0:
                    print(f"[进度] 已处理 {total_count} 个样本，找到 {match_count} 个匹配...")

            except json.JSONDecodeError as e:
                print(f"[错误] 第 {line_num} 行 JSON 解析失败: {e}")
                continue
            except Exception as e:
                print(f"[错误] 第 {line_num} 行处理失败: {e}")
                continue

    # 打印统计结果
    print("\n" + "="*60)
    print("检测结果")
    print("="*60)
    print(f"总样本数：{total_count}")
    print(f"匹配样本数：{match_count}")
    print(f"匹配比例：{match_count/total_count*100:.2f}%" if total_count > 0 else "N/A")

    # 显示前10个匹配的样本示例
    if match_samples:
        print("\n前10个匹配样本示例：")
        print("="*60)
        for i, sample in enumerate(match_samples, 1):
            print(f"\n样本 {i}:")
            print(f"  行号: {sample['line']}")
            print(f"  ID: {sample['id']}")
            print(f"  中间部分: {sample['middle_part']}")
            print(f"  stage_to_estimate: {sample['stage_to_estimate']}")

    print("\n" + "="*60)
    print("检测完成！")


if __name__ == '__main__':
    jsonl_file = '/projects/p32958/chengxuan/ProgressLM/data/train/visual_demo/visual_negative_trans_img_replaced_processed.jsonl'
    check_id_stage_match(jsonl_file)
