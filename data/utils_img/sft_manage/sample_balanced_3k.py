#!/usr/bin/env python3
"""
从JSONL文件中采样约3000个样本
要求：
1. 过滤 total_steps 在 [3, 7] 范围内
2. 同一ID的样本要么全部选中要么全部不选
3. 保证每个action_type都存在
4. 每个action_type的采样数量尽量均衡
"""

import json
import random
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Set


class BalancedSampler:
    """均衡采样器"""

    def __init__(self, input_path: str, output_path: str, seed: int = 42):
        """
        初始化采样器

        Args:
            input_path: 输入JSONL文件路径
            output_path: 输出JSONL文件路径
            seed: 随机种子
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        random.seed(seed)

        # 采样策略：每个action_type应选择的ID数量（策略3-混合策略）
        self.sampling_plan = {
            'blue_cub_on_pink': 18,
            'close_cap_lid': 19,
            'close_cap_trash_can': 16,
            'open_cap_lid': 19,
            'open_cap_trash_can': 14,
            'pick_plate_from_plate_rack': 19,
            'place_in_block_tennis_ball': 13,
            'place_in_trash': 20,
            'place_plate_in_plate_rack': 19,
            'slide_close_drawer_1_1': 19,
            'slide_open_drawer_1': 19,
            'stick_target_blue_on_the_pink_obejct': 19,
        }

    def extract_action_type(self, sample_id: str) -> str:
        """
        从ID中提取action_type

        Args:
            sample_id: 格式为 "source/action_type/trajectory_id"

        Returns:
            action_type
        """
        parts = sample_id.split('/')
        if len(parts) != 3:
            raise ValueError(f"Invalid sample_id format: {sample_id}")
        return parts[1]

    def filter_by_steps(self, samples: List[Dict]) -> List[Dict]:
        """
        过滤total_steps在[3,7]范围内的样本

        Args:
            samples: 样本列表

        Returns:
            过滤后的样本列表
        """
        filtered = []
        for sample in samples:
            total_steps = sample.get('total_steps', 0)
            # 转换为整数以防止类型错误
            try:
                total_steps = int(total_steps)
            except (ValueError, TypeError):
                total_steps = 0

            if 3 <= total_steps <= 7:
                filtered.append(sample)

        print(f"过滤前样本数: {len(samples)}")
        print(f"过滤后样本数: {len(filtered)}")
        print(f"过滤掉样本数: {len(samples) - len(filtered)}")

        return filtered

    def group_by_id_and_action(self, samples: List[Dict]) -> Dict[str, Dict[str, List[Dict]]]:
        """
        按action_type和ID分组样本

        Args:
            samples: 样本列表

        Returns:
            {action_type: {id: [samples]}}
        """
        grouped = defaultdict(lambda: defaultdict(list))

        for sample in samples:
            sample_id = sample['id']
            action_type = self.extract_action_type(sample_id)
            grouped[action_type][sample_id].append(sample)

        return grouped

    def select_ids_per_action(self, grouped: Dict[str, Dict[str, List[Dict]]]) -> Set[str]:
        """
        为每个action_type选择指定数量的ID

        Args:
            grouped: {action_type: {id: [samples]}}

        Returns:
            选中的ID集合
        """
        selected_ids = set()

        print("\n" + "=" * 80)
        print("采样策略")
        print("=" * 80)
        print(f"{'Action Type':<40} {'可用ID':<10} {'选择ID':<10} {'选择率':<10}")
        print("-" * 80)

        for action_type, id_groups in sorted(grouped.items()):
            available_ids = list(id_groups.keys())
            num_to_select = self.sampling_plan.get(action_type, 0)

            # 如果可用ID数少于计划数，全部选择
            num_to_select = min(num_to_select, len(available_ids))

            # 随机选择ID
            selected = random.sample(available_ids, num_to_select)
            selected_ids.update(selected)

            select_rate = num_to_select / len(available_ids) * 100 if available_ids else 0
            print(f"{action_type:<40} {len(available_ids):<10} {num_to_select:<10} {select_rate:>6.1f}%")

        print("=" * 80)
        print(f"总选择ID数: {len(selected_ids)}")
        print("=" * 80 + "\n")

        return selected_ids

    def extract_selected_samples(self, samples: List[Dict], selected_ids: Set[str]) -> List[Dict]:
        """
        提取选中ID对应的所有样本

        Args:
            samples: 样本列表
            selected_ids: 选中的ID集合

        Returns:
            选中的样本列表
        """
        selected_samples = []
        for sample in samples:
            if sample['id'] in selected_ids:
                selected_samples.append(sample)

        return selected_samples

    def generate_statistics(self, samples: List[Dict]) -> Dict:
        """
        生成统计信息

        Args:
            samples: 样本列表

        Returns:
            统计字典
        """
        stats = {
            'total_samples': len(samples),
            'unique_ids': len(set(s['id'] for s in samples)),
            'action_types': defaultdict(lambda: {'ids': set(), 'samples': 0})
        }

        for sample in samples:
            action_type = self.extract_action_type(sample['id'])
            stats['action_types'][action_type]['ids'].add(sample['id'])
            stats['action_types'][action_type]['samples'] += 1

        # 转换set为数量
        for action_type in stats['action_types']:
            stats['action_types'][action_type]['num_ids'] = len(stats['action_types'][action_type]['ids'])
            del stats['action_types'][action_type]['ids']

        return stats

    def print_statistics(self, title: str, stats: Dict):
        """
        打印统计信息

        Args:
            title: 标题
            stats: 统计字典
        """
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)
        print(f"总样本数: {stats['total_samples']}")
        print(f"不同ID数: {stats['unique_ids']}")
        print("\n按Action Type统计:")
        print(f"{'Action Type':<40} {'ID数':<10} {'样本数':<10} {'平均样本/ID':<15}")
        print("-" * 80)

        for action_type, data in sorted(stats['action_types'].items()):
            avg = data['samples'] / data['num_ids'] if data['num_ids'] > 0 else 0
            print(f"{action_type:<40} {data['num_ids']:<10} {data['samples']:<10} {avg:>10.2f}")

        print("=" * 80)

    def run(self):
        """
        执行采样流程
        """
        print(f"开始采样任务")
        print(f"输入文件: {self.input_path}")
        print(f"输出文件: {self.output_path}")
        print("-" * 80)

        # 1. 读取所有样本
        print("\n[步骤1] 读取输入文件...")
        samples = []
        with open(self.input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        print(f"读取完成，总样本数: {len(samples)}")

        # 2. 过滤total_steps
        print("\n[步骤2] 过滤total_steps在[3,7]范围...")
        filtered_samples = self.filter_by_steps(samples)

        # 统计过滤后的数据
        filtered_stats = self.generate_statistics(filtered_samples)
        self.print_statistics("过滤后数据统计", filtered_stats)

        # 3. 按action_type和ID分组
        print("\n[步骤3] 按action_type和ID分组...")
        grouped = self.group_by_id_and_action(filtered_samples)
        print(f"分组完成，共{len(grouped)}个action_type")

        # 4. 选择ID
        print("\n[步骤4] 为每个action_type选择ID...")
        selected_ids = self.select_ids_per_action(grouped)

        # 5. 提取选中样本
        print("[步骤5] 提取选中ID对应的所有样本...")
        selected_samples = self.extract_selected_samples(filtered_samples, selected_ids)
        print(f"提取完成，共{len(selected_samples)}个样本")

        # 6. 保存到输出文件
        print(f"\n[步骤6] 保存到输出文件: {self.output_path}")
        with open(self.output_path, 'w', encoding='utf-8') as f:
            for sample in selected_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        print("保存完成！")

        # 7. 生成并打印最终统计
        final_stats = self.generate_statistics(selected_samples)
        self.print_statistics("最终采样结果统计", final_stats)


def main():
    """主函数"""
    INPUT_PATH = "/home/vcj9002/jianshu/chengxuan/ProgressLM/data/eval/visual/visual_franka_multi_view_test.jsonl"
    OUTPUT_PATH = "/home/vcj9002/jianshu/chengxuan/ProgressLM/data/eval/visual/visual_franka_multi_view_3k.jsonl"

    sampler = BalancedSampler(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        seed=42  # 固定随机种子以保证可重复性
    )

    sampler.run()


if __name__ == "__main__":
    main()
