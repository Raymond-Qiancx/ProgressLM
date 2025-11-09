#!/usr/bin/env python3
"""
跨相机视角数据增强脚本
对JSONL中的每个样本随机替换相机角度，确保demo和stage使用不同的相机视角
- 组合1: demo=camera_left, stage=camera_top
- 组合2: demo=camera_right, stage=camera_top
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from collections import defaultdict


class CrossCameraAugmenter:
    """跨相机视角数据增强器"""

    # 可用的相机组合 (demo_camera, stage_camera)
    CAMERA_COMBINATIONS = [
        ('camera_left', 'camera_top'),
        ('camera_right', 'camera_top'),
    ]

    def __init__(self, input_jsonl: str, output_jsonl: str,
                 base_image_dir: str, random_seed: int = 42,
                 verify_images: bool = True):
        """
        初始化增强器

        Args:
            input_jsonl: 输入JSONL文件路径
            output_jsonl: 输出JSONL文件路径
            base_image_dir: 图像基础目录路径
            random_seed: 随机种子
            verify_images: 是否验证图像文件存在
        """
        self.input_jsonl = Path(input_jsonl)
        self.output_jsonl = Path(output_jsonl)
        self.base_image_dir = Path(base_image_dir)
        self.verify_images = verify_images

        # 设置随机种子
        random.seed(random_seed)

        # 统计信息
        self.total_samples = 0
        self.success_samples = 0
        self.failed_samples = 0
        self.camera_combo_stats = defaultdict(int)  # 统计每种组合的使用次数
        self.failed_records = []  # 记录失败的样本

    def parse_image_filename(self, filename: str) -> Tuple[str, str, str]:
        """
        解析图像文件名，提取相机类型和帧号

        Args:
            filename: 图像文件名，如 "camera_top_0020.jpg"

        Returns:
            (camera_type, frame_number, extension) 如 ("camera_top", "0020", "jpg")
        """
        # 移除扩展名
        name_without_ext = filename.rsplit('.', 1)[0]
        extension = filename.rsplit('.', 1)[1] if '.' in filename else 'jpg'

        # 分割获取相机类型和帧号
        parts = name_without_ext.rsplit('_', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid filename format: {filename}")

        camera_prefix, frame_number = parts
        return camera_prefix, frame_number, extension

    def replace_camera_in_filename(self, filename: str, new_camera: str) -> str:
        """
        替换文件名中的相机类型

        Args:
            filename: 原始文件名
            new_camera: 新的相机类型

        Returns:
            新的文件名
        """
        old_camera, frame_number, extension = self.parse_image_filename(filename)
        return f"{new_camera}_{frame_number}.{extension}"

    def build_image_path(self, sample_id: str, camera_type: str, filename: str) -> Path:
        """
        构建图像完整路径

        Args:
            sample_id: 样本ID，格式为 "data_source/action_type/trajectory_id"
            camera_type: 相机类型
            filename: 文件名

        Returns:
            完整图像路径
        """
        parts = sample_id.split('/')
        if len(parts) != 3:
            raise ValueError(f"Invalid sample_id format: {sample_id}")

        data_source, action_type, trajectory_id = parts
        return self.base_image_dir / camera_type / data_source / action_type / trajectory_id / filename

    def verify_image_exists(self, sample_id: str, camera_type: str, filename: str) -> bool:
        """
        验证图像文件是否存在

        Args:
            sample_id: 样本ID
            camera_type: 相机类型
            filename: 文件名

        Returns:
            是否存在
        """
        if not self.verify_images:
            return True

        image_path = self.build_image_path(sample_id, camera_type, filename)
        return image_path.exists()

    def augment_sample(self, sample: Dict) -> Tuple[Dict, bool, str]:
        """
        对单个样本进行相机视角增强

        Args:
            sample: 原始样本

        Returns:
            (增强后的样本, 是否成功, 失败原因)
        """
        sample_id = sample['id']

        # 随机选择相机组合
        demo_camera, stage_camera = random.choice(self.CAMERA_COMBINATIONS)
        combo_key = f"{demo_camera}+{stage_camera}"

        # 创建新样本（深拷贝）
        new_sample = sample.copy()

        try:
            # 替换 visual_demo 中的所有图像
            if 'visual_demo' in sample:
                new_demo_images = []
                for img_filename in sample['visual_demo']:
                    new_filename = self.replace_camera_in_filename(img_filename, demo_camera)

                    # 验证图像是否存在
                    if not self.verify_image_exists(sample_id, demo_camera, new_filename):
                        return None, False, f"Demo image not found: {demo_camera}/{new_filename}"

                    new_demo_images.append(new_filename)

                new_sample['visual_demo'] = new_demo_images

            # 替换 stage_to_estimate 中的所有图像
            if 'stage_to_estimate' in sample:
                new_stage_images = []
                for img_filename in sample['stage_to_estimate']:
                    new_filename = self.replace_camera_in_filename(img_filename, stage_camera)

                    # 验证图像是否存在
                    if not self.verify_image_exists(sample_id, stage_camera, new_filename):
                        return None, False, f"Stage image not found: {stage_camera}/{new_filename}"

                    new_stage_images.append(new_filename)

                new_sample['stage_to_estimate'] = new_stage_images

            # 记录使用的相机组合
            new_sample['camera_combination'] = combo_key

            # 统计
            self.camera_combo_stats[combo_key] += 1

            return new_sample, True, ""

        except Exception as e:
            return None, False, f"Error: {str(e)}"

    def process_all(self) -> Dict:
        """
        处理所有样本

        Returns:
            统计报告
        """
        print(f"开始处理样本...")
        print(f"输入文件: {self.input_jsonl}")
        print(f"输出文件: {self.output_jsonl}")
        print(f"图像验证: {'启用' if self.verify_images else '禁用'}")
        print("-" * 80)

        # 读取所有样本
        samples = []
        with open(self.input_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))

        self.total_samples = len(samples)
        print(f"读取样本数: {self.total_samples}")

        # 处理每个样本
        augmented_samples = []
        for idx, sample in enumerate(samples, 1):
            new_sample, success, error_msg = self.augment_sample(sample)

            if success:
                augmented_samples.append(new_sample)
                self.success_samples += 1
            else:
                self.failed_samples += 1
                self.failed_records.append({
                    'sample_id': sample['id'],
                    'reason': error_msg
                })
                print(f"  ✗ 样本 {sample['id']} 失败: {error_msg}")

            # 进度显示
            if idx % 100 == 0:
                print(f"已处理: {idx}/{self.total_samples} ({idx*100//self.total_samples}%)")

        # 保存增强后的样本
        print(f"\n保存增强样本到: {self.output_jsonl}")
        self.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_jsonl, 'w', encoding='utf-8') as f:
            for sample in augmented_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        # 生成报告
        report = self.generate_report()
        return report

    def generate_report(self) -> Dict:
        """
        生成统计报告

        Returns:
            报告字典
        """
        report = {
            'summary': {
                'input_file': str(self.input_jsonl),
                'output_file': str(self.output_jsonl),
                'total_samples': self.total_samples,
                'success_samples': self.success_samples,
                'failed_samples': self.failed_samples,
                'success_rate': f"{self.success_samples * 100 / self.total_samples:.2f}%" if self.total_samples > 0 else "0%",
            },
            'camera_combination_stats': dict(self.camera_combo_stats),
            'failed_records': self.failed_records,
            'timestamp': datetime.now().isoformat()
        }

        return report

    def print_summary(self, report: Dict):
        """
        打印统计摘要

        Args:
            report: 统计报告
        """
        summary = report['summary']
        combo_stats = report['camera_combination_stats']

        print("\n" + "=" * 80)
        print("跨相机视角数据增强报告")
        print("=" * 80)

        print("\n【文件信息】")
        print(f"  输入文件: {summary['input_file']}")
        print(f"  输出文件: {summary['output_file']}")

        print("\n【处理统计】")
        print(f"  总样本数:     {summary['total_samples']}")
        print(f"  成功样本数:   {summary['success_samples']} ({summary['success_rate']})")
        print(f"  失败样本数:   {summary['failed_samples']}")

        print("\n【相机组合分布】")
        for combo, count in sorted(combo_stats.items()):
            demo_cam, stage_cam = combo.split('+')
            percentage = f"{count * 100 / summary['success_samples']:.2f}%" if summary['success_samples'] > 0 else "0%"
            print(f"  {combo:30s} {count:6d} ({percentage})")
            print(f"    ├─ visual_demo:        {demo_cam}")
            print(f"    └─ stage_to_estimate:  {stage_cam}")

        if report['failed_records']:
            print("\n【失败记录】")
            for record in report['failed_records'][:10]:  # 只显示前10条
                print(f"  ✗ {record['sample_id']}")
                print(f"    原因: {record['reason']}")
            if len(report['failed_records']) > 10:
                print(f"  ... 还有 {len(report['failed_records']) - 10} 条失败记录")

        print("=" * 80)

    def save_report(self, report: Dict, output_path: Path):
        """
        保存详细报告到JSON文件

        Args:
            report: 报告字典
            output_path: 输出文件路径
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n详细报告已保存至: {output_path}")


def main():
    """主函数"""
    # 配置路径
    INPUT_JSONL = "/home/vcj9002/jianshu/chengxuan/ProgressLM/data/eval/visual/visual_franka_complete.jsonl"
    OUTPUT_JSONL = "/home/vcj9002/jianshu/chengxuan/ProgressLM/data/eval/visual/visual_franka_multi_view_test.jsonl"
    BASE_IMAGE_DIR = "/home/vcj9002/jianshu/chengxuan/Data/robomind/data/3rgb"
    REPORT_DIR = Path("/home/vcj9002/jianshu/chengxuan/ProgressLM/data/utils_img/sft_manage")

    # 配置参数
    RANDOM_SEED = 42  # 随机种子，可重复实验
    VERIFY_IMAGES = True  # 是否验证图像文件存在

    print("=" * 80)
    print("跨相机视角数据增强工具")
    print("=" * 80)

    # 创建增强器
    augmenter = CrossCameraAugmenter(
        input_jsonl=INPUT_JSONL,
        output_jsonl=OUTPUT_JSONL,
        base_image_dir=BASE_IMAGE_DIR,
        random_seed=RANDOM_SEED,
        verify_images=VERIFY_IMAGES
    )

    # 执行增强
    report = augmenter.process_all()

    # 打印摘要
    augmenter.print_summary(report)

    # 保存详细报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"cross_camera_augmentation_report_{timestamp}.json"
    augmenter.save_report(report, report_path)

    print("\n✓ 增强完成！")


if __name__ == "__main__":
    main()
