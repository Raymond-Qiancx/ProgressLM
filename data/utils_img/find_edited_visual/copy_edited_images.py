#!/usr/bin/env python3
"""
复制 edited_visual_transfer_raw.jsonl 中的 stage_to_estimate 图片
从源目录复制到目标目录，保持相同的目录结构
"""

import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# 路径配置
# JSONL_FILE = "/projects/p32958/chengxuan/ProgressLM/data/negative/final_edited/edited_visual_transfer_raw.jsonl"
# SOURCE_BASE = "/projects/p32958/chengxuan/results/progresslm/negative/image"
# TARGET_BASE = "/projects/p32958/chengxuan/new_extracted_images/images"

JSONL_FILE = "/projects/p32958/chengxuan/ProgressLM/data/benchmark/tiny-bench/visual-nega.jsonl"
SOURCE_BASE = "/projects/p32958/chengxuan/results/progresslm/negative/image"
TARGET_BASE = "/projects/p32958/chengxuan/data/images"

# 失败记录
failures = {
    "source_not_exist": [],
    "target_dir_not_exist": [],
    "copy_error": []
}

# 成功计数
success_count = 0
total_count = 0


def load_jsonl():
    """读取 jsonl 文件并返回所有记录"""
    records = []
    with open(JSONL_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def copy_image(record_id, image_name):
    """复制单张图片"""
    global success_count, total_count
    total_count += 1

    # 构建源路径和目标路径
    source_path = os.path.join(SOURCE_BASE, record_id, image_name)
    target_dir = os.path.join(TARGET_BASE, record_id)
    target_path = os.path.join(target_dir, image_name)

    # 检查源文件是否存在
    if not os.path.exists(source_path):
        failures["source_not_exist"].append({
            "id": record_id,
            "image": image_name,
            "source_path": source_path
        })
        return False

    # 检查目标目录是否存在（不存在则报错，不创建）
    if not os.path.exists(target_dir):
        failures["target_dir_not_exist"].append({
            "id": record_id,
            "image": image_name,
            "target_dir": target_dir
        })
        return False

    # 复制文件
    try:
        shutil.copy2(source_path, target_path)
        success_count += 1
        return True
    except Exception as e:
        failures["copy_error"].append({
            "id": record_id,
            "image": image_name,
            "source_path": source_path,
            "target_path": target_path,
            "error": str(e)
        })
        return False


def print_report():
    """打印详细报告"""
    print("\n" + "="*80)
    print("复制结果报告".center(80))
    print("="*80)

    print(f"\n总计: {total_count} 张图片")
    print(f"成功: {success_count} 张")
    print(f"失败: {total_count - success_count} 张")
    print(f"成功率: {success_count/total_count*100:.2f}%")

    # 源文件不存在
    if failures["source_not_exist"]:
        print(f"\n❌ 源文件不存在 ({len(failures['source_not_exist'])} 个):")
        for item in failures["source_not_exist"][:10]:  # 只显示前10个
            print(f"  - ID: {item['id']}")
            print(f"    图片: {item['image']}")
            print(f"    源路径: {item['source_path']}")
        if len(failures["source_not_exist"]) > 10:
            print(f"  ... 还有 {len(failures['source_not_exist']) - 10} 个")

    # 目标目录不存在
    if failures["target_dir_not_exist"]:
        print(f"\n❌ 目标目录不存在 ({len(failures['target_dir_not_exist'])} 个):")
        for item in failures["target_dir_not_exist"][:10]:
            print(f"  - ID: {item['id']}")
            print(f"    图片: {item['image']}")
            print(f"    目标目录: {item['target_dir']}")
        if len(failures["target_dir_not_exist"]) > 10:
            print(f"  ... 还有 {len(failures['target_dir_not_exist']) - 10} 个")

    # 复制错误
    if failures["copy_error"]:
        print(f"\n❌ 复制错误 ({len(failures['copy_error'])} 个):")
        for item in failures["copy_error"][:10]:
            print(f"  - ID: {item['id']}")
            print(f"    图片: {item['image']}")
            print(f"    错误: {item['error']}")
        if len(failures["copy_error"]) > 10:
            print(f"  ... 还有 {len(failures['copy_error']) - 10} 个")

    # 保存详细失败日志
    if total_count - success_count > 0:
        log_file = "/projects/p32958/chengxuan/ProgressLM/data/utils_img/find_edited_visual/copy_failures.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(failures, f, indent=2, ensure_ascii=False)
        print(f"\n完整失败日志已保存到: {log_file}")

    print("\n" + "="*80)


def main():
    print("开始读取 jsonl 文件...")
    records = load_jsonl()
    print(f"共读取 {len(records)} 条记录")

    # 统计总图片数
    total_images = sum(len(r.get("stage_to_estimate", [])) for r in records)
    print(f"共需复制 {total_images} 张图片\n")

    # 复制图片（带进度条）
    print("开始复制图片...")
    with tqdm(total=total_images, desc="复制进度", unit="张") as pbar:
        for record in records:
            record_id = record.get("id")
            stage_images = record.get("stage_to_estimate", [])

            for image_name in stage_images:
                copy_image(record_id, image_name)
                pbar.update(1)

    # 打印报告
    print_report()


if __name__ == "__main__":
    main()
