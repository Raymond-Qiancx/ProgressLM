import json
from pathlib import Path
from tqdm import tqdm

def check_images_exist(jsonl_path, image_root):
    """
    检查JSONL文件中所有图片是否存在
    兼容两种数据格式:
    1. visual_demo格式: visual_demo是图片列表, stage_to_estimate是图片列表
    2. text_demo格式: text_demo是文本列表, stage_to_estimate是单个图片字符串
    
    Args:
        jsonl_path: JSONL文件路径
        image_root: 图片根目录
    """
    image_root = Path(image_root)
    
    # 统计信息
    total_samples = 0
    total_images = 0
    missing_count = 0
    samples_with_missing = set()
    
    # 先计算总行数
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    # 使用tqdm显示进度
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in tqdm(enumerate(f, 1), total=total_lines, desc="检查图片"):
            try:
                data = json.loads(line.strip())
                total_samples += 1
                
                sample_id = data.get('id', '')
                sample_has_missing = False
                
                # 检查 visual_demo 中的图片（如果存在）
                visual_demo = data.get('visual_demo', [])
                if isinstance(visual_demo, list):
                    for img_name in visual_demo:
                        img_path = image_root / sample_id / img_name
                        total_images += 1
                        if not img_path.exists():
                            missing_count += 1
                            sample_has_missing = True

                # 检查 stage_to_estimate 中的图片（兼容字符串和列表格式）
                stage_to_estimate = data.get('stage_to_estimate', [])

                # 如果是字符串，转为列表处理
                if isinstance(stage_to_estimate, str):
                    stage_to_estimate = [stage_to_estimate] if stage_to_estimate else []

                # 如果是列表，遍历检查
                if isinstance(stage_to_estimate, list):
                    for img_name in stage_to_estimate:
                        if img_name and img_name != "n/a":  # 跳过空值和n/a
                            img_path = image_root / sample_id / img_name
                            total_images += 1
                            if not img_path.exists():
                                missing_count += 1
                                sample_has_missing = True
                
                if sample_has_missing:
                    samples_with_missing.add(sample_id)
                    
            except:
                pass
    
    # 打印统计结果
    print(f"\n{'='*80}")
    print(f"检查完成!")
    print(f"{'='*80}")
    print(f"总样本数: {total_samples}")
    print(f"总图片数: {total_images}")
    print(f"缺失图片数: {missing_count}")
    print(f"有缺失图片的样本数: {len(samples_with_missing)}")
    
    if total_images > 0:
        completeness = (total_images - missing_count) / total_images * 100
        print(f"完整性: {completeness:.2f}%")
        
    if missing_count == 0:
        print(f"\n✅ 所有图片都存在!")
    else:
        print(f"\n⚠️  发现 {missing_count} 张图片缺失")


# if __name__ == '__main__':
#     # 设置路径
#     JSONL_PATH = '/projects/p32958/chengxuan/ProgressLM/data/train/rl/new/new_rl_35k_final.jsonl'  # 替换为你的JSONL文件路径 clean
#     IMAGE_ROOT = '/projects/p32958/chengxuan/new_extracted_images/images'  # 替换为你的图片根目录
    
#     # 运行检查
#     check_images_exist(JSONL_PATH, IMAGE_ROOT)

if __name__ == '__main__':
    # 设置路径
    JSONL_PATH = '/home/vcj9002/jianshu/chengxuan/ProgressLM/data/eval/visual/visual_franka_multi_view_3k.jsonl'  # 替换为你的JSONL文件路径 clean
    IMAGE_ROOT = '/home/vcj9002/jianshu/chengxuan/Data/ProgressLM-data/images'  # 替换为你的图片根目录
    
    # 运行检查
    check_images_exist(JSONL_PATH, IMAGE_ROOT)