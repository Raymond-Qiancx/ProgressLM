#!/usr/bin/env python3
"""
多线程验证JSONL文件中的图像是否在其他相机视角中存在对应帧
"""

import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple
from datetime import datetime
from collections import defaultdict
import threading


@dataclass
class ImageCheckResult:
    """图像检查结果"""
    sample_id: str
    image_filename: str
    current_camera: str
    frame_number: str
    missing_cameras: List[str]
    existing_cameras: List[str]


@dataclass
class SampleCheckResult:
    """样本级别检查结果"""
    sample_id: str
    total_images: int
    fully_matched_images: int
    is_sample_complete: bool  # 该样本的所有图像是否都完全匹配
    missing_images: List[Dict]  # 有缺失的图像详情


class MultiCameraVerifier:
    """多相机图像验证器"""

    # 所有相机类型
    ALL_CAMERAS = ['camera_left', 'camera_right', 'camera_top']

    def __init__(self, jsonl_path: str, base_image_dir: str, num_threads: int = 16):
        """
        初始化验证器

        Args:
            jsonl_path: JSONL文件路径
            base_image_dir: 图像基础目录路径
            num_threads: 线程数
        """
        self.jsonl_path = Path(jsonl_path)
        self.base_image_dir = Path(base_image_dir)
        self.num_threads = num_threads

        # 统计信息（线程安全）
        self.lock = threading.Lock()
        self.total_images = 0
        self.fully_matched = 0  # 另外两个视角都存在
        self.partially_matched = 0  # 只有一个视角存在
        self.fully_missing = 0  # 两个视角都不存在

        # 样本级别统计
        self.total_samples = 0
        self.fully_complete_samples = 0  # 所有图像都完全匹配的样本数

        # 详细结果
        self.detailed_results: List[ImageCheckResult] = []
        self.sample_results: List[SampleCheckResult] = []  # 样本级别结果

    def parse_image_filename(self, filename: str) -> Tuple[str, str]:
        """
        解析图像文件名，提取相机类型和帧号

        Args:
            filename: 图像文件名，如 "camera_top_0020.jpg"

        Returns:
            (camera_type, frame_number) 如 ("camera_top", "0020")
        """
        # 移除扩展名
        name_without_ext = filename.rsplit('.', 1)[0]

        # 分割获取相机类型和帧号
        parts = name_without_ext.rsplit('_', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid filename format: {filename}")

        camera_prefix, frame_number = parts
        return camera_prefix, frame_number

    def get_other_cameras(self, current_camera: str) -> List[str]:
        """
        获取除当前相机外的其他相机类型

        Args:
            current_camera: 当前相机类型

        Returns:
            其他相机类型列表
        """
        return [cam for cam in self.ALL_CAMERAS if cam != current_camera]

    def build_image_path(self, camera_type: str, data_source: str,
                        action_type: str, trajectory_id: str,
                        camera_name: str, frame_number: str) -> Path:
        """
        构建图像完整路径

        Args:
            camera_type: 相机类型目录 (camera_left/camera_right/camera_top)
            data_source: 数据源 (h5_franka_3rgb)
            action_type: 动作类型
            trajectory_id: 轨迹ID
            camera_name: 相机名称前缀
            frame_number: 帧号

        Returns:
            完整图像路径
        """
        filename = f"{camera_name}_{frame_number}.jpg"
        return self.base_image_dir / camera_type / data_source / action_type / trajectory_id / filename

    def check_single_image(self, sample_id: str, image_filename: str) -> ImageCheckResult:
        """
        检查单个图像在其他视角中是否存在

        Args:
            sample_id: 样本ID，格式为 "data_source/action_type/trajectory_id"
            image_filename: 图像文件名

        Returns:
            检查结果
        """
        # 解析sample_id
        parts = sample_id.split('/')
        if len(parts) != 3:
            raise ValueError(f"Invalid sample_id format: {sample_id}")
        data_source, action_type, trajectory_id = parts

        # 解析图像文件名
        current_camera, frame_number = self.parse_image_filename(image_filename)

        # 获取需要检查的其他相机
        other_cameras = self.get_other_cameras(current_camera)

        # 检查每个相机视角
        existing_cameras = []
        missing_cameras = []

        for camera_type in other_cameras:
            image_path = self.build_image_path(
                camera_type, data_source, action_type,
                trajectory_id, camera_type, frame_number
            )

            if image_path.exists():
                existing_cameras.append(camera_type)
            else:
                missing_cameras.append(camera_type)

        return ImageCheckResult(
            sample_id=sample_id,
            image_filename=image_filename,
            current_camera=current_camera,
            frame_number=frame_number,
            missing_cameras=missing_cameras,
            existing_cameras=existing_cameras
        )

    def process_sample(self, sample: Dict) -> List[ImageCheckResult]:
        """
        处理单个样本，检查所有图像

        Args:
            sample: JSONL中的一行样本

        Returns:
            所有图像的检查结果列表
        """
        sample_id = sample['id']
        results = []

        # 收集所有图像
        all_images = []
        if 'visual_demo' in sample:
            all_images.extend(sample['visual_demo'])
        if 'stage_to_estimate' in sample:
            all_images.extend(sample['stage_to_estimate'])

        # 检查每个图像
        for image_filename in all_images:
            try:
                result = self.check_single_image(sample_id, image_filename)
                results.append(result)
            except Exception as e:
                print(f"Error processing {sample_id}/{image_filename}: {e}")

        return results

    def update_statistics(self, results: List[ImageCheckResult], sample_id: str):
        """
        更新统计信息（线程安全）

        Args:
            results: 检查结果列表
            sample_id: 样本ID
        """
        # 计算样本级别的统计
        sample_total_images = len(results)
        sample_fully_matched = 0
        sample_missing_images = []

        with self.lock:
            self.total_samples += 1

            for result in results:
                self.total_images += 1

                num_existing = len(result.existing_cameras)
                num_missing = len(result.missing_cameras)

                if num_missing == 0:
                    # 另外两个视角都存在
                    self.fully_matched += 1
                    sample_fully_matched += 1
                elif num_existing == 0:
                    # 两个视角都不存在
                    self.fully_missing += 1
                else:
                    # 只有一个视角存在
                    self.partially_matched += 1

                # 只记录有缺失的结果
                if num_missing > 0:
                    self.detailed_results.append(result)
                    sample_missing_images.append({
                        'image': result.image_filename,
                        'current_camera': result.current_camera,
                        'frame': result.frame_number,
                        'missing_cameras': result.missing_cameras,
                        'existing_cameras': result.existing_cameras
                    })

            # 判断该样本是否完全匹配
            is_sample_complete = (sample_fully_matched == sample_total_images)
            if is_sample_complete:
                self.fully_complete_samples += 1

            # 记录样本级别结果
            sample_result = SampleCheckResult(
                sample_id=sample_id,
                total_images=sample_total_images,
                fully_matched_images=sample_fully_matched,
                is_sample_complete=is_sample_complete,
                missing_images=sample_missing_images
            )
            self.sample_results.append(sample_result)

    def verify_all(self) -> Dict:
        """
        验证所有样本

        Returns:
            验证报告字典
        """
        print(f"开始验证 JSONL 文件: {self.jsonl_path}")
        print(f"图像基础目录: {self.base_image_dir}")
        print(f"使用线程数: {self.num_threads}")
        print("-" * 80)

        # 读取所有样本
        samples = []
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))

        print(f"总样本数: {len(samples)}")

        # 多线程处理
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = {executor.submit(self.process_sample, sample): sample
                      for sample in samples}

            # 处理完成的任务
            completed = 0
            for future in as_completed(futures):
                sample = futures[future]
                results = future.result()
                self.update_statistics(results, sample['id'])

                completed += 1
                if completed % 100 == 0:
                    print(f"已处理: {completed}/{len(samples)} 样本 "
                          f"({completed*100//len(samples)}%)")

        print(f"\n验证完成！已处理 {completed}/{len(samples)} 样本")

        # 生成报告
        report = self.generate_report(len(samples))
        return report

    def generate_report(self, total_samples: int) -> Dict:
        """
        生成验证报告

        Args:
            total_samples: 总样本数

        Returns:
            报告字典
        """
        # 按样本ID分组缺失信息
        missing_by_sample = defaultdict(list)
        for result in self.detailed_results:
            missing_by_sample[result.sample_id].append({
                'image': result.image_filename,
                'current_camera': result.current_camera,
                'frame': result.frame_number,
                'missing_cameras': result.missing_cameras,
                'existing_cameras': result.existing_cameras
            })

        # 统计不完整的样本
        incomplete_samples = [sr for sr in self.sample_results if not sr.is_sample_complete]

        report = {
            'summary': {
                # 样本级别统计
                'total_samples': self.total_samples,
                'fully_complete_samples': self.fully_complete_samples,
                'incomplete_samples': self.total_samples - self.fully_complete_samples,
                'sample_complete_percentage': f"{self.fully_complete_samples * 100 / self.total_samples:.2f}%" if self.total_samples > 0 else "0%",

                # 图像级别统计
                'total_images': self.total_images,
                'fully_matched_images': self.fully_matched,
                'partially_matched_images': self.partially_matched,
                'fully_missing_images': self.fully_missing,
                'image_match_percentage': f"{self.fully_matched * 100 / self.total_images:.2f}%" if self.total_images > 0 else "0%",
            },
            'incomplete_samples': [
                {
                    'sample_id': sr.sample_id,
                    'total_images': sr.total_images,
                    'matched_images': sr.fully_matched_images,
                    'missing_count': len(sr.missing_images),
                    'missing_details': sr.missing_images
                }
                for sr in incomplete_samples
            ],
            'timestamp': datetime.now().isoformat()
        }

        return report

    def print_summary(self, report: Dict):
        """
        打印摘要信息

        Args:
            report: 报告字典
        """
        summary = report['summary']

        print("\n" + "=" * 80)
        print("验证报告摘要")
        print("=" * 80)

        print("\n【样本级别统计】")
        print(f"  总样本数:               {summary['total_samples']}")
        print(f"  完全匹配样本数:         {summary['fully_complete_samples']} ({summary['sample_complete_percentage']})")
        print(f"    (该样本的所有图像在另外两个视角都存在)")
        print(f"  不完整样本数:           {summary['incomplete_samples']}")
        print(f"    (该样本至少有一张图像缺少某个视角)")

        print("\n【图像级别统计】")
        print(f"  总图像数:               {summary['total_images']}")
        print(f"  完全匹配图像:           {summary['fully_matched_images']} ({summary['image_match_percentage']})")
        print(f"    (该图像在另外两个视角都存在)")
        print(f"  部分匹配图像:           {summary['partially_matched_images']}")
        print(f"    (该图像只在一个视角存在)")
        print(f"  完全缺失图像:           {summary['fully_missing_images']}")
        print(f"    (该图像在两个视角都不存在)")

        print("=" * 80)

    def save_report(self, report: Dict, output_path: Path):
        """
        保存详细报告到JSON文件

        Args:
            report: 报告字典
            output_path: 输出文件路径
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n详细报告已保存至: {output_path}")


def main():
    """主函数"""
    # 配置路径
    JSONL_PATH = "/home/vcj9002/jianshu/chengxuan/ProgressLM/data/train/visual_demo/visual_h5_franka_3rgb_sft_clean.jsonl"
    BASE_IMAGE_DIR = "/home/vcj9002/jianshu/chengxuan/Data/robomind/data/3rgb"
    OUTPUT_DIR = Path("/home/vcj9002/jianshu/chengxuan/ProgressLM/data/utils_img/sft_manage")

    # 配置参数
    NUM_THREADS = 512  # 可根据机器性能调整

    # 创建验证器
    verifier = MultiCameraVerifier(
        jsonl_path=JSONL_PATH,
        base_image_dir=BASE_IMAGE_DIR,
        num_threads=NUM_THREADS
    )

    # 执行验证
    report = verifier.verify_all()

    # 打印摘要
    verifier.print_summary(report)

    # 保存详细报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"verification_report_{timestamp}.json"
    verifier.save_report(report, output_path)


if __name__ == "__main__":
    main()
