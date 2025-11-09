#!/usr/bin/env python3
"""
将JSONL中引用的所有图像从3rgb目录复制到ProgressLM目录
使用多线程加速复制过程
"""

import json
import shutil
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import threading


class ImageCopyTask:
    """图像复制任务"""

    def __init__(self, jsonl_path: str, source_base_dir: str,
                 target_base_dir: str, num_threads: int = 512,
                 overwrite: bool = False):
        """
        初始化复制任务

        Args:
            jsonl_path: JSONL文件路径
            source_base_dir: 源图像基础目录 (3rgb)
            target_base_dir: 目标图像基础目录 (ProgressLM/images)
            num_threads: 线程数
            overwrite: 是否覆盖已存在的文件
        """
        self.jsonl_path = Path(jsonl_path)
        self.source_base_dir = Path(source_base_dir)
        self.target_base_dir = Path(target_base_dir)
        self.num_threads = num_threads
        self.overwrite = overwrite

        # 统计信息（线程安全）
        self.lock = threading.Lock()
        self.total_unique_images = 0
        self.success_count = 0
        self.skip_count = 0  # 已存在，跳过
        self.failed_count = 0
        self.total_bytes_copied = 0

        # 失败记录
        self.failed_records = []

    def parse_image_filename(self, filename: str) -> Tuple[str, str]:
        """
        解析图像文件名，提取相机类型和帧号

        Args:
            filename: 图像文件名，如 "camera_left_0000.jpg"

        Returns:
            (camera_type, frame_number) 如 ("camera_left", "0000")
        """
        name_without_ext = filename.rsplit('.', 1)[0]
        parts = name_without_ext.rsplit('_', 1)

        if len(parts) != 2:
            raise ValueError(f"Invalid filename format: {filename}")

        camera_type, frame_number = parts
        return camera_type, frame_number

    def build_source_path(self, sample_id: str, filename: str) -> Path:
        """
        构建源图像路径

        Args:
            sample_id: 样本ID "data_source/action_type/trajectory_id"
            filename: 图像文件名

        Returns:
            源图像完整路径
        """
        parts = sample_id.split('/')
        if len(parts) != 3:
            raise ValueError(f"Invalid sample_id format: {sample_id}")

        data_source, action_type, trajectory_id = parts
        camera_type, _ = self.parse_image_filename(filename)

        return self.source_base_dir / camera_type / data_source / action_type / trajectory_id / filename

    def build_target_path(self, sample_id: str, filename: str) -> Path:
        """
        构建目标图像路径（扁平化，去掉camera_type层）

        Args:
            sample_id: 样本ID "data_source/action_type/trajectory_id"
            filename: 图像文件名

        Returns:
            目标图像完整路径
        """
        parts = sample_id.split('/')
        if len(parts) != 3:
            raise ValueError(f"Invalid sample_id format: {sample_id}")

        data_source, action_type, trajectory_id = parts

        return self.target_base_dir / data_source / action_type / trajectory_id / filename

    def collect_unique_images(self) -> Set[Tuple[str, str]]:
        """
        收集JSONL中所有唯一的图像

        Returns:
            唯一图像集合 {(sample_id, filename), ...}
        """
        print(f"读取JSONL文件: {self.jsonl_path}")

        unique_images = set()
        sample_count = 0

        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue

                sample = json.loads(line)
                sample_id = sample['id']
                sample_count += 1

                # 收集visual_demo图像
                if 'visual_demo' in sample:
                    for img_filename in sample['visual_demo']:
                        unique_images.add((sample_id, img_filename))

                # 收集stage_to_estimate图像
                if 'stage_to_estimate' in sample:
                    for img_filename in sample['stage_to_estimate']:
                        unique_images.add((sample_id, img_filename))

        print(f"样本数: {sample_count}")
        print(f"唯一图像数: {len(unique_images)}")

        return unique_images

    def copy_single_image(self, sample_id: str, filename: str) -> Tuple[bool, str, int]:
        """
        复制单个图像文件

        Args:
            sample_id: 样本ID
            filename: 图像文件名

        Returns:
            (是否成功, 状态信息, 文件大小)
        """
        try:
            source_path = self.build_source_path(sample_id, filename)
            target_path = self.build_target_path(sample_id, filename)

            # 检查源文件是否存在
            if not source_path.exists():
                return False, f"Source not found: {source_path}", 0

            # 检查目标文件是否已存在
            if target_path.exists() and not self.overwrite:
                file_size = target_path.stat().st_size
                return True, "skipped", file_size

            # 创建目标目录
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # 复制文件
            shutil.copy2(source_path, target_path)
            file_size = target_path.stat().st_size

            return True, "success", file_size

        except Exception as e:
            return False, f"Error: {str(e)}", 0

    def update_statistics(self, success: bool, status: str, file_size: int,
                         sample_id: str, filename: str):
        """
        更新统计信息（线程安全）

        Args:
            success: 是否成功
            status: 状态信息
            file_size: 文件大小
            sample_id: 样本ID
            filename: 文件名
        """
        with self.lock:
            if success:
                if status == "skipped":
                    self.skip_count += 1
                else:
                    self.success_count += 1
                    self.total_bytes_copied += file_size
            else:
                self.failed_count += 1
                self.failed_records.append({
                    'sample_id': sample_id,
                    'filename': filename,
                    'reason': status
                })

    def copy_all_images(self, unique_images: Set[Tuple[str, str]]) -> Dict:
        """
        批量复制所有图像

        Args:
            unique_images: 唯一图像集合

        Returns:
            统计报告
        """
        self.total_unique_images = len(unique_images)

        print(f"\n开始复制图像...")
        print(f"源目录: {self.source_base_dir}")
        print(f"目标目录: {self.target_base_dir}")
        print(f"线程数: {self.num_threads}")
        print(f"覆盖模式: {'启用' if self.overwrite else '禁用'}")
        print("-" * 80)

        # 转换为列表以便处理
        image_list = list(unique_images)

        # 多线程复制
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # 提交所有任务
            futures = {
                executor.submit(self.copy_single_image, sample_id, filename): (sample_id, filename)
                for sample_id, filename in image_list
            }

            # 处理完成的任务
            completed = 0
            for future in as_completed(futures):
                sample_id, filename = futures[future]
                success, status, file_size = future.result()

                self.update_statistics(success, status, file_size, sample_id, filename)

                completed += 1

                # 实时显示进度
                if completed % 100 == 0 or completed == self.total_unique_images:
                    with self.lock:
                        progress = completed * 100 / self.total_unique_images
                        print(f"进度: {completed}/{self.total_unique_images} ({progress:.1f}%) | "
                              f"成功: {self.success_count} | 跳过: {self.skip_count} | "
                              f"失败: {self.failed_count}")

        print("\n复制完成！")

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
                'jsonl_file': str(self.jsonl_path),
                'source_dir': str(self.source_base_dir),
                'target_dir': str(self.target_base_dir),
                'total_unique_images': self.total_unique_images,
                'success_count': self.success_count,
                'skip_count': self.skip_count,
                'failed_count': self.failed_count,
                'success_rate': f"{(self.success_count + self.skip_count) * 100 / self.total_unique_images:.2f}%" if self.total_unique_images > 0 else "0%",
                'total_bytes_copied': self.total_bytes_copied,
                'total_size_mb': f"{self.total_bytes_copied / 1024 / 1024:.2f}",
                'total_size_gb': f"{self.total_bytes_copied / 1024 / 1024 / 1024:.2f}",
            },
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

        print("\n" + "=" * 80)
        print("图像复制报告")
        print("=" * 80)

        print("\n【文件信息】")
        print(f"  JSONL文件:    {summary['jsonl_file']}")
        print(f"  源目录:       {summary['source_dir']}")
        print(f"  目标目录:     {summary['target_dir']}")

        print("\n【复制统计】")
        print(f"  唯一图像总数:   {summary['total_unique_images']}")
        print(f"  成功复制:       {summary['success_count']}")
        print(f"  已存在跳过:     {summary['skip_count']}")
        print(f"  失败:           {summary['failed_count']}")
        print(f"  成功率:         {summary['success_rate']}")

        print("\n【磁盘占用】")
        print(f"  新复制数据:     {summary['total_size_mb']} MB ({summary['total_size_gb']} GB)")

        if report['failed_records']:
            print("\n【失败记录】")
            for i, record in enumerate(report['failed_records'][:10], 1):
                print(f"  {i}. {record['sample_id']}/{record['filename']}")
                print(f"     原因: {record['reason']}")
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

    def run(self) -> Dict:
        """
        执行完整的复制流程

        Returns:
            统计报告
        """
        # 收集唯一图像
        unique_images = self.collect_unique_images()

        # 复制所有图像
        report = self.copy_all_images(unique_images)

        return report


def main():
    """主函数"""
    # 配置路径
    JSONL_PATH = "/home/vcj9002/jianshu/chengxuan/ProgressLM/data/eval/visual/visual_franka_multi_view_3k.jsonl"
    SOURCE_BASE_DIR = "/home/vcj9002/jianshu/chengxuan/Data/robomind/data/3rgb"
    TARGET_BASE_DIR = "/home/vcj9002/jianshu/chengxuan/Data/ProgressLM-data/images"
    REPORT_DIR = Path("/home/vcj9002/jianshu/chengxuan/ProgressLM/data/utils_img/sft_manage")

    # 配置参数
    NUM_THREADS = 512  # 线程数
    OVERWRITE = False  # 是否覆盖已存在的文件

    print("=" * 80)
    print("图像复制工具 - ProgressLM")
    print("=" * 80)

    # 创建复制任务
    task = ImageCopyTask(
        jsonl_path=JSONL_PATH,
        source_base_dir=SOURCE_BASE_DIR,
        target_base_dir=TARGET_BASE_DIR,
        num_threads=NUM_THREADS,
        overwrite=OVERWRITE
    )

    # 执行复制
    report = task.run()

    # 打印摘要
    task.print_summary(report)

    # 保存详细报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_DIR / f"image_copy_report_{timestamp}.json"
    task.save_report(report, report_path)

    print("\n✓ 任务完成！")


if __name__ == "__main__":
    main()
