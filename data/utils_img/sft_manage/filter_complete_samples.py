#!/usr/bin/env python3
"""
从JSONL文件中筛选出所有完备的样本（所有图像在其他两个相机视角都存在）
"""

import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Set, Tuple
import threading


class CompleteSampleFilter:
    """完备样本筛选器"""

    # 所有相机类型
    ALL_CAMERAS = ['camera_left', 'camera_right', 'camera_top']

    def __init__(self, jsonl_path: str, base_image_dir: str, num_threads: int = 16):
        """
        初始化筛选器

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
        self.total_samples = 0
        self.complete_samples = 0

        # 保存完备样本的ID
        self.complete_sample_ids: Set[str] = set()

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

    def check_single_image(self, sample_id: str, image_filename: str) -> bool:
        """
        检查单个图像在其他视角中是否完全存在

        Args:
            sample_id: 样本ID，格式为 "data_source/action_type/trajectory_id"
            image_filename: 图像文件名

        Returns:
            True表示在其他两个视角都存在，False表示至少缺少一个视角
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

        # 检查每个相机视角是否都存在
        for camera_type in other_cameras:
            image_path = self.build_image_path(
                camera_type, data_source, action_type,
                trajectory_id, camera_type, frame_number
            )

            if not image_path.exists():
                return False

        return True

    def is_sample_complete(self, sample: Dict) -> bool:
        """
        检查样本是否完备（所有图像在其他两个视角都存在）

        Args:
            sample: JSONL中的一行样本

        Returns:
            True表示完备，False表示不完备
        """
        sample_id = sample['id']

        # 收集所有图像
        all_images = []
        if 'visual_demo' in sample:
            all_images.extend(sample['visual_demo'])
        if 'stage_to_estimate' in sample:
            all_images.extend(sample['stage_to_estimate'])

        # 检查每个图像
        for image_filename in all_images:
            try:
                if not self.check_single_image(sample_id, image_filename):
                    return False
            except Exception as e:
                print(f"Error processing {sample_id}/{image_filename}: {e}")
                return False

        return True

    def process_sample(self, sample: Dict) -> Tuple[str, bool]:
        """
        处理单个样本，返回样本ID和是否完备

        Args:
            sample: JSONL中的一行样本

        Returns:
            (sample_id, is_complete)
        """
        sample_id = sample['id']
        is_complete = self.is_sample_complete(sample)

        with self.lock:
            self.total_samples += 1
            if is_complete:
                self.complete_samples += 1
                self.complete_sample_ids.add(sample_id)

        return sample_id, is_complete

    def filter_complete_samples(self, output_path: str) -> Dict:
        """
        筛选出所有完备样本并保存

        Args:
            output_path: 输出JSONL文件路径

        Returns:
            统计信息字典
        """
        print(f"开始筛选完备样本")
        print(f"输入文件: {self.jsonl_path}")
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
                sample_id, is_complete = future.result()

                completed += 1
                if completed % 100 == 0:
                    print(f"已处理: {completed}/{len(samples)} 样本 "
                          f"({completed*100//len(samples)}%)")

        print(f"\n筛选完成！已处理 {completed}/{len(samples)} 样本")
        print(f"完备样本数: {self.complete_samples}")
        print(f"不完备样本数: {self.total_samples - self.complete_samples}")

        # 保存完备样本
        output_path = Path(output_path)
        complete_count = 0

        with open(output_path, 'w', encoding='utf-8') as f_out:
            with open(self.jsonl_path, 'r', encoding='utf-8') as f_in:
                for line in f_in:
                    if line.strip():
                        sample = json.loads(line)
                        if sample['id'] in self.complete_sample_ids:
                            f_out.write(line)
                            complete_count += 1

        print(f"\n已保存 {complete_count} 个完备样本至: {output_path}")

        # 返回统计信息
        stats = {
            'total_samples': self.total_samples,
            'complete_samples': self.complete_samples,
            'incomplete_samples': self.total_samples - self.complete_samples,
            'complete_percentage': f"{self.complete_samples * 100 / self.total_samples:.2f}%" if self.total_samples > 0 else "0%",
            'output_file': str(output_path)
        }

        return stats


def main():
    """主函数"""
    # 配置路径
    INPUT_JSONL = "/home/vcj9002/jianshu/chengxuan/ProgressLM/data/eval/visual/visual_franka_candidate.jsonl"
    BASE_IMAGE_DIR = "/home/vcj9002/jianshu/chengxuan/Data/robomind/data/3rgb"
    OUTPUT_JSONL = "/home/vcj9002/jianshu/chengxuan/ProgressLM/data/eval/visual/visual_franka_complete.jsonl"

    # 配置参数
    NUM_THREADS = 512  # 可根据机器性能调整

    # 创建筛选器
    filter = CompleteSampleFilter(
        jsonl_path=INPUT_JSONL,
        base_image_dir=BASE_IMAGE_DIR,
        num_threads=NUM_THREADS
    )

    # 执行筛选
    stats = filter.filter_complete_samples(OUTPUT_JSONL)

    # 打印统计信息
    print("\n" + "=" * 80)
    print("筛选统计")
    print("=" * 80)
    print(f"总样本数:       {stats['total_samples']}")
    print(f"完备样本数:     {stats['complete_samples']} ({stats['complete_percentage']})")
    print(f"不完备样本数:   {stats['incomplete_samples']}")
    print(f"输出文件:       {stats['output_file']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
