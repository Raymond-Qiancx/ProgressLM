#!/usr/bin/env python3
"""
进度恢复脚本
从 annotated_output.jsonl 和统计文件重建 annotation_progress.json
"""

import json
import os

def restore_progress():
    """从已保存的输出恢复进度"""

    # 文件路径
    original_jsonl = "/gpfs/projects/p32958/chengxuan/ProgressLM/data/raw/edit_imgs/edited_raw_all.jsonl"
    annotated_output = "/gpfs/projects/p32958/chengxuan/ProgressLM/data/utils_img/visual_nega/annotated_output.jsonl"
    stats_file = "/gpfs/projects/p32958/chengxuan/ProgressLM/data/utils_img/visual_nega/annotated_output_stats.txt"
    progress_file = "/gpfs/projects/p32958/chengxuan/ProgressLM/data/utils_img/visual_nega/annotation_progress.json"

    print("=" * 60)
    print("进度恢复脚本")
    print("=" * 60)

    # 1. 读取原始数据
    print("\n[1/4] 读取原始数据...")
    all_records = []
    with open(original_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                all_records.append(json.loads(line))
    print(f"✓ 加载了 {len(all_records)} 条原始记录")

    # 2. 读取已标注为YES的记录
    print("\n[2/4] 读取已标注的记录...")
    yes_records = []
    with open(annotated_output, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yes_records.append(json.loads(line))
    print(f"✓ 找到 {len(yes_records)} 条 YES 记录")

    # 3. 读取统计信息获取已标注总数
    print("\n[3/4] 读取统计信息...")
    with open(stats_file, 'r', encoding='utf-8') as f:
        content = f.read()
        for line in content.split('\n'):
            if line.startswith('已标注数:'):
                total_annotated = int(line.split(':')[1].strip())
                print(f"✓ 已标注总数: {total_annotated}")
                break

    # 4. 重建进度信息
    print("\n[4/4] 重建进度文件...")

    # 创建一个集合来快速查找YES记录
    # 使用记录的唯一标识来匹配（这里使用整个记录的JSON字符串）
    yes_records_set = set()
    for record in yes_records:
        # 使用meta_data中的id和image作为唯一标识
        meta = record.get('meta_data', {})
        key = (meta.get('id', ''), meta.get('image', ''))
        yes_records_set.add(key)

    # 重建annotations字典
    annotations = {}
    for idx in range(total_annotated):
        if idx >= len(all_records):
            break

        record = all_records[idx]
        meta = record.get('meta_data', {})
        key = (meta.get('id', ''), meta.get('image', ''))

        # 如果在YES记录中，标记为True，否则为False
        annotations[idx] = key in yes_records_set

    # 当前索引设置为已标注的数量
    current_index = total_annotated

    # 保存进度文件
    progress = {
        'current_index': current_index,
        'annotations': annotations
    }

    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2)

    print(f"✓ 进度文件已创建: {progress_file}")

    # 5. 显示恢复的统计信息
    print("\n" + "=" * 60)
    print("恢复完成！统计信息：")
    print("=" * 60)
    yes_count = sum(1 for v in annotations.values() if v)
    no_count = sum(1 for v in annotations.values() if not v)
    print(f"当前进度: {current_index} / {len(all_records)}")
    print(f"已标注: {len(annotations)} 条")
    print(f"  - YES (保留): {yes_count}")
    print(f"  - NO (删除): {no_count}")
    print(f"剩余未标注: {len(all_records) - current_index}")
    print("=" * 60)
    print("\n✅ 现在可以重新启动标注工具，它将从第 {} 条记录继续！\n".format(current_index + 1))

    return True

if __name__ == '__main__':
    try:
        restore_progress()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
