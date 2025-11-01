import json
import random
from collections import defaultdict
from typing import List, Dict, Tuple

def parse_id(id_str: str) -> Tuple[str, str, str]:
    """解析id字段，返回(source, action_category, trajectory_id)"""
    parts = id_str.split('/')
    if len(parts) >= 3:
        source = parts[0]
        action_category = parts[1]
        trajectory_id = parts[2].split('.')[0]  # 去掉可能的扩展名部分
        return source, action_category, trajectory_id
    return None, None, None

def load_jsonl(filepath: str) -> List[Dict]:
    """加载JSONL文件"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_jsonl(data: List[Dict], filepath: str):
    """保存为JSONL文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def filter_and_sample(input_file: str, output_file: str, target_count: int = 3000):
    """
    筛选和采样数据
    
    Args:
        input_file: 输入的JSONL文件路径
        output_file: 输出的JSONL文件路径
        target_count: 目标采样数量（约3000）
    """
    print(f"开始读取数据...")
    all_data = load_jsonl(input_file)
    print(f"总共读取 {len(all_data)} 条数据")
    
    # 步骤1: 筛选total_steps在3-7之间的数据
    print("\n步骤1: 筛选total_steps在3-7之间的数据...")
    filtered_data = []
    for item in all_data:
        total_steps = item.get('total_steps')
        # 处理可能是字符串的情况
        if isinstance(total_steps, str):
            total_steps = int(total_steps)
        if 3 <= total_steps <= 7:
            filtered_data.append(item)
    
    print(f"筛选后剩余 {len(filtered_data)} 条数据")
    
    if len(filtered_data) == 0:
        print("错误：没有符合条件的数据！")
        return
    
    # 步骤2: 按trajectory_id分组
    print("\n步骤2: 按trajectory_id分组...")
    trajectory_groups = defaultdict(list)
    source_stats = defaultdict(int)
    action_categories = set()
    
    for item in filtered_data:
        source, action_category, trajectory_id = parse_id(item['id'])
        if source and action_category and trajectory_id:
            full_trajectory_id = f"{source}/{action_category}/{trajectory_id}"
            trajectory_groups[full_trajectory_id].append(item)
            source_stats[source] += 1
            action_categories.add(f"{source}/{action_category}")
    
    print(f"共有 {len(trajectory_groups)} 个不同的trajectory")
    print(f"共有 {len(action_categories)} 个不同的action类别")
    
    # 统计每个source的比例
    total_filtered = len(filtered_data)
    source_ratios = {source: count / total_filtered for source, count in source_stats.items()}
    
    print("\n各data_source的数据分布:")
    for source, ratio in sorted(source_ratios.items()):
        print(f"  {source}: {source_stats[source]} 条 ({ratio*100:.2f}%)")
    
    print("\n各action类别:")
    for action in sorted(action_categories):
        action_count = sum(1 for tid, items in trajectory_groups.items() if tid.startswith(action))
        print(f"  {action}: {action_count} trajectories")
    
    # 步骤3: 分层采样
    print(f"\n步骤3: 进行分层采样 (目标约{target_count}条)...")
    
    # 为每个action类别至少保留一个trajectory
    action_to_trajectories = defaultdict(list)
    for trajectory_id, items in trajectory_groups.items():
        parts = trajectory_id.split('/')
        if len(parts) >= 2:
            action_key = f"{parts[0]}/{parts[1]}"
            action_to_trajectories[action_key].append(trajectory_id)
    
    # 确保每个action类别至少有一个trajectory
    must_include_trajectories = set()
    for action_key, trajectories in action_to_trajectories.items():
        # 随机选择一个trajectory作为该类别的代表
        must_include_trajectories.add(random.choice(trajectories))
    
    print(f"为保证覆盖所有action类别，至少包含 {len(must_include_trajectories)} 个trajectory")
    
    # 按source分组trajectories
    source_to_trajectories = defaultdict(list)
    for trajectory_id in trajectory_groups.keys():
        source = trajectory_id.split('/')[0]
        source_to_trajectories[source].append(trajectory_id)
    
    # 计算每个source应该采样多少个trajectory
    selected_trajectories = set(must_include_trajectories)
    
    # 估算：如果平均每个trajectory有n条数据
    avg_items_per_trajectory = len(filtered_data) / len(trajectory_groups)
    print(f"平均每个trajectory有 {avg_items_per_trajectory:.1f} 条数据")
    
    # 计算还需要多少条数据
    current_count = sum(len(trajectory_groups[tid]) for tid in selected_trajectories)
    remaining_target = target_count - current_count
    
    print(f"已选中 {len(selected_trajectories)} 个trajectory，共 {current_count} 条数据")
    print(f"还需要约 {remaining_target} 条数据")
    
    # 按比例从各个source采样剩余的trajectories
    for source, trajectories in source_to_trajectories.items():
        # 该source还有哪些trajectory未被选中
        available = [t for t in trajectories if t not in selected_trajectories]
        if not available:
            continue
        
        # 计算该source应该采样的数量（按比例）
        source_target = int(remaining_target * source_ratios[source])
        # 转换为trajectory数量
        needed_trajectories = max(1, int(source_target / avg_items_per_trajectory))
        needed_trajectories = min(needed_trajectories, len(available))
        
        # 随机采样
        sampled = random.sample(available, needed_trajectories)
        selected_trajectories.update(sampled)
    
    # 收集所有选中trajectory的数据
    sampled_data = []
    for trajectory_id in selected_trajectories:
        sampled_data.extend(trajectory_groups[trajectory_id])
    
    # 如果数据量还不够，继续补充
    if len(sampled_data) < target_count * 0.9:  # 如果少于目标的90%
        print(f"\n数据量不足，继续补充...")
        remaining_trajectories = [t for t in trajectory_groups.keys() if t not in selected_trajectories]
        random.shuffle(remaining_trajectories)
        
        for trajectory_id in remaining_trajectories:
            if len(sampled_data) >= target_count:
                break
            sampled_data.extend(trajectory_groups[trajectory_id])
            selected_trajectories.add(trajectory_id)
    
    print(f"\n最终选中 {len(selected_trajectories)} 个trajectory，共 {len(sampled_data)} 条数据")
    
    # 验证各source的比例
    print("\n最终各data_source的数据分布:")
    final_source_stats = defaultdict(int)
    for item in sampled_data:
        source = item.get('data_source')
        final_source_stats[source] += 1
    
    for source in sorted(final_source_stats.keys()):
        count = final_source_stats[source]
        ratio = count / len(sampled_data)
        original_ratio = source_ratios.get(source, 0)
        print(f"  {source}: {count} 条 ({ratio*100:.2f}%, 原始比例: {original_ratio*100:.2f}%)")
    
    # 验证action类别覆盖
    print("\n最终action类别覆盖:")
    final_actions = set()
    for item in sampled_data:
        source, action_category, _ = parse_id(item['id'])
        if source and action_category:
            final_actions.add(f"{source}/{action_category}")
    
    print(f"覆盖了 {len(final_actions)} / {len(action_categories)} 个action类别")
    missing_actions = action_categories - final_actions
    if missing_actions:
        print(f"缺失的类别: {missing_actions}")
    else:
        print("✓ 所有action类别都已覆盖")
    
    # 保存结果
    print(f"\n保存结果到 {output_file}...")
    save_jsonl(sampled_data, output_file)
    print("完成！")
    
    # 生成统计报告
    report_file = output_file.replace('.jsonl', '_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"数据采样报告\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"原始数据总量: {len(all_data)}\n")
        f.write(f"筛选后数据量: {len(filtered_data)}\n")
        f.write(f"采样后数据量: {len(sampled_data)}\n")
        f.write(f"选中trajectory数: {len(selected_trajectories)}\n\n")
        f.write(f"各data_source分布:\n")
        for source in sorted(final_source_stats.keys()):
            count = final_source_stats[source]
            ratio = count / len(sampled_data)
            original_ratio = source_ratios.get(source, 0)
            f.write(f"  {source}: {count} 条 ({ratio*100:.2f}%, 原始: {original_ratio*100:.2f}%)\n")
        f.write(f"\nAction类别覆盖: {len(final_actions)} / {len(action_categories)}\n")
    
    print(f"统计报告已保存到 {report_file}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python filter_jsonl.py <input_file> [output_file] [target_count]")
        print("示例: python filter_jsonl.py data.jsonl filtered_data.jsonl 3000")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "filtered_output.jsonl"
    target_count = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    
    random.seed(42)  # 设置随机种子以便结果可复现
    filter_and_sample(input_file, output_file, target_count)