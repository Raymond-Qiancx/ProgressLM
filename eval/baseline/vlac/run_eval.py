#!/usr/bin/env python3
"""
VLAC Image Sequence Evaluation Script
使用VLAC模型评估图像序列轨迹的质量

用法:
    python run_eval.py --model_path ./models/VLAC --data_dir ./images --task "Pick up bowl"
"""

import argparse
import os
import sys
import json
import glob
from pathlib import Path
from PIL import Image
import time

# 添加当前目录到路径以导入utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import GAC_model


def load_images_from_dir(image_dir, pattern="*.jpg", max_images=None):
    """
    从目录加载图像序列

    Args:
        image_dir: 图像目录路径
        pattern: 文件匹配模式 (*.jpg, *.png, etc.)
        max_images: 最大加载图像数量，None表示加载全部

    Returns:
        List[PIL.Image]: 图像列表
    """
    # 支持多种图像格式
    patterns = pattern.split(',') if ',' in pattern else [pattern]
    image_paths = []
    for pat in patterns:
        image_paths.extend(glob.glob(os.path.join(image_dir, pat.strip())))

    image_paths = sorted(image_paths)

    if max_images is not None:
        image_paths = image_paths[:max_images]

    if len(image_paths) == 0:
        raise ValueError(f"No images found in {image_dir} with pattern {pattern}")

    print(f"Loading {len(image_paths)} images from {image_dir}")
    images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert('RGB')
            images.append(img)
        except Exception as e:
            print(f"Warning: Failed to load {path}: {e}")
            continue

    return images


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="VLAC Image Sequence Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 必需参数
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to VLAC model directory')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to input images directory')
    parser.add_argument('--task', type=str, required=True,
                        help='Task description (e.g., "Pick up the bowl")')

    # 可选参数 - 参考轨迹
    parser.add_argument('--ref_dir', type=str, default=None,
                        help='Path to reference images directory (optional)')

    # 可选参数 - 模型配置
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run model on (cuda:0, cpu, etc.)')
    parser.add_argument('--model_type', type=str, default='internvl2',
                        help='Model type')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=1,
                        help='Top-k sampling')

    # 可选参数 - 评估配置
    parser.add_argument('--batch_num', type=int, default=5,
                        help='Batch size for inference')
    parser.add_argument('--ref_num', type=int, default=6,
                        help='Number of reference images to sample')
    parser.add_argument('--skip', type=int, default=1,
                        help='Frame skip step for pair-wise evaluation')
    parser.add_argument('--rich', action='store_true',
                        help='Enable rich mode (output decimal values)')
    parser.add_argument('--think', action='store_true',
                        help='Enable Chain-of-Thought reasoning')
    parser.add_argument('--reverse_eval', action='store_true',
                        help='Enable reverse evaluation (for VROC)')

    # 可选参数 - 输入输出
    parser.add_argument('--image_pattern', type=str, default='*.jpg,*.png',
                        help='Image file pattern (comma-separated)')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to load')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for results')
    parser.add_argument('--output_name', type=str, default=None,
                        help='Output filename (default: auto-generated)')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print("VLAC Image Sequence Evaluation")
    print("="*80)
    print(f"Model Path: {args.model_path}")
    print(f"Data Directory: {args.data_dir}")
    print(f"Task: {args.task}")
    print(f"Device: {args.device}")
    print(f"Batch Size: {args.batch_num}")
    print(f"Skip: {args.skip}")
    print("="*80)

    # ========== 1. 加载图像 ==========
    print("\n[Step 1/4] Loading images...")
    start_time = time.time()

    test_images = load_images_from_dir(
        args.data_dir,
        pattern=args.image_pattern,
        max_images=args.max_images
    )

    ref_images = None
    if args.ref_dir and os.path.exists(args.ref_dir):
        print(f"Loading reference images from {args.ref_dir}")
        ref_images = load_images_from_dir(
            args.ref_dir,
            pattern=args.image_pattern
        )
        print(f"Loaded {len(ref_images)} reference images")

    load_time = time.time() - start_time
    print(f"Image loading completed in {load_time:.2f}s")

    # ========== 2. 初始化模型 ==========
    print("\n[Step 2/4] Initializing VLAC model...")
    start_time = time.time()

    critic = GAC_model(tag='critic')
    critic.init_model(
        model_path=args.model_path,
        model_type=args.model_type,
        device_map=args.device
    )
    critic.temperature = args.temperature
    critic.top_k = args.top_k
    critic.set_config()
    critic.set_system_prompt()

    init_time = time.time() - start_time
    print(f"Model initialization completed in {init_time:.2f}s")

    # ========== 3. 运行评估 ==========
    print("\n[Step 3/4] Running trajectory evaluation...")
    print(f"Evaluating {len(test_images)} images with skip={args.skip}")
    print(f"This will generate {len(range(args.skip, len(test_images), args.skip))} critic scores")

    start_time = time.time()

    critic_list, value_list = critic.get_trajectory_critic(
        task=args.task,
        image_list=test_images,
        ref_image_list=ref_images,
        batch_num=args.batch_num,
        ref_num=args.ref_num if ref_images else 0,
        skip=args.skip,
        rich=args.rich,
        think=args.think,
        reverse_eval=args.reverse_eval,
        frame_skip=True
    )

    eval_time = time.time() - start_time
    print(f"Evaluation completed in {eval_time:.2f}s")

    # ========== 4. 计算质量指标 ==========
    print("\n[Step 4/4] Computing quality metrics...")

    voc = critic.compute_voc(value_list)
    negative_rate = critic.compute_negative_rate(critic_list)

    # ========== 输出结果 ==========
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Task: {args.task}")
    print(f"Total Frames: {len(test_images)}")
    print(f"Evaluated Steps: {len(critic_list)}")
    print(f"\nQuality Metrics:")
    print(f"  VOC (Value-Order Correlation): {voc:.4f}")
    print(f"    → Range: -1 to +1, higher is better")
    print(f"    → Interpretation: {'Excellent (>0.5)' if voc > 0.5 else 'Good (0.3-0.5)' if voc > 0.3 else 'Poor (<0.3)'}")
    print(f"  Negative Rate: {negative_rate:.4f}")
    print(f"    → Range: 0 to 1, lower is better")
    print(f"    → Interpretation: {'Excellent (<0.2)' if negative_rate < 0.2 else 'Fair (0.2-0.4)' if negative_rate < 0.4 else 'Poor (>0.4)'}")

    print(f"\nValue List (Task Progress %):")
    print(f"  Start: {value_list[0]:.2f}%")
    print(f"  End: {value_list[-1]:.2f}%")
    print(f"  Full: {[f'{v:.2f}' for v in value_list]}")

    print(f"\nCritic List (Step-wise Rewards):")
    print(f"  Positive steps: {sum(1 for c in critic_list if float(c) > 0)}/{len(critic_list)}")
    print(f"  Negative steps: {sum(1 for c in critic_list if float(c) < 0)}/{len(critic_list)}")
    print(f"  Full: {critic_list}")

    # ========== 保存结果 ==========
    if args.output_name:
        output_filename = args.output_name if args.output_name.endswith('.json') else f"{args.output_name}.json"
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_filename = f"vlac_eval_{timestamp}.json"

    output_path = os.path.join(args.output_dir, output_filename)

    results = {
        'config': {
            'model_path': args.model_path,
            'data_dir': args.data_dir,
            'ref_dir': args.ref_dir,
            'task': args.task,
            'batch_num': args.batch_num,
            'skip': args.skip,
            'ref_num': args.ref_num,
            'device': args.device,
            'rich': args.rich,
            'think': args.think
        },
        'metrics': {
            'voc': float(voc),
            'negative_rate': float(negative_rate),
            'num_frames': len(test_images),
            'num_steps': len(critic_list),
            'positive_steps': sum(1 for c in critic_list if float(c) > 0),
            'negative_steps': sum(1 for c in critic_list if float(c) < 0)
        },
        'results': {
            'value_list': [float(v) for v in value_list],
            'critic_list': [str(c) for c in critic_list]
        },
        'timing': {
            'load_time': load_time,
            'init_time': init_time,
            'eval_time': eval_time,
            'total_time': load_time + init_time + eval_time
        }
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print("="*80)

    # 返回质量指标用于脚本判断
    return voc, negative_rate


if __name__ == "__main__":
    main()
