import json
import re
import numpy as np
from scipy.stats import spearmanr
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


def extract_ref_from_text(response: str) -> Optional[int]:
    """从文本中提取ref值，从头扫描第一个数字"""
    match = re.search(r'\d+', response)
    if match:
        return int(match.group())
    return None


def extract_score_from_text(response: str) -> Optional[float]:
    """从文本中提取score值，从尾扫描第一个百分号及其前面的数字"""
    # 从尾部开始查找 %
    matches = list(re.finditer(r'(\d+(?:\.\d+)?)\s*%', response))
    if matches:
        # 取最后一个匹配
        return float(matches[-1].group(1))
    return None


def parse_percentage(score_str: str) -> Optional[float]:
    """解析百分比字符串"""
    if isinstance(score_str, (int, float)):
        return float(score_str)
    if isinstance(score_str, str):
        match = re.search(r'(\d+(?:\.\d+)?)', score_str.strip().replace('%', ''))
        if match:
            return float(match.group(1))
    return None


def check_format(data: Dict) -> bool:
    """检查输出格式是否符合预期"""
    try:
        # 检查是否有ref和score字段
        if 'ref' not in data or 'score' not in data:
            return False
        
        # 检查ref是否为数字或可转换为数字
        ref = data['ref']
        if isinstance(ref, str):
            if not ref.isdigit():
                return False
        elif not isinstance(ref, int):
            return False
        
        # 检查score是否为百分比格式
        score = data['score']
        if isinstance(score, str):
            if '%' not in score or not re.search(r'\d+', score):
                return False
        elif not isinstance(score, (int, float)):
            return False
        
        return True
    except:
        return False


def extract_values(data: Dict) -> Tuple[Optional[int], Optional[float], int]:
    """
    提取ref和score值
    返回: (ref, score, format_score)
    """
    format_score = 1 if check_format(data) else 0
    
    # 提取ref
    ref = None
    if 'ref' in data:
        if isinstance(data['ref'], int):
            ref = data['ref']
        elif isinstance(data['ref'], str) and data['ref'].isdigit():
            ref = int(data['ref'])
    
    # 如果ref提取失败，从response中提取
    if ref is None and 'response' in data:
        ref = extract_ref_from_text(data['response'])
    
    # 提取score
    score = None
    if 'score' in data:
        score = parse_percentage(data['score'])
    
    # 如果score提取失败，从response中提取
    if score is None and 'response' in data:
        score = extract_score_from_text(data['response'])
    
    return ref, score, format_score


def compute_relative_error(predicted, ground_truth):
    """计算相对误差: |ground_truth - predicted| / ground_truth"""
    if ground_truth == 0:
        return None  # 避免除以0
    return abs(ground_truth - predicted) / abs(ground_truth)


def compute_negative_rate(critic_list):
    """计算负值（倒退步骤）的比例"""
    if len(critic_list) == 0:
        return 0.0
    negative_critic = [one for one in critic_list if one < 0]
    return len(negative_critic) / len(critic_list)


def evaluate_jsonl(file_path: str):
    """主评估函数"""
    
    # 读取所有数据
    all_data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                all_data.append(json.loads(line))
    
    print(f"Total samples: {len(all_data)}")
    print("=" * 80)
    
    # 统计各项指标
    ref_errors = []
    score_errors = []
    format_scores = []
    
    # 按trajectory ID分组
    trajectories = defaultdict(list)
    
    for data in all_data:
        # 提取值
        ref, score, format_score = extract_values(data)
        format_scores.append(format_score)
        
        # 提取ground truth
        closest_idx = data.get('closest_idx')
        ground_truth_score = parse_percentage(data.get('ground_truth_score'))
        
        # 转换closest_idx
        if isinstance(closest_idx, str):
            closest_idx = int(closest_idx) if closest_idx.isdigit() else None
        
        # 计算相对误差
        if ref is not None and closest_idx is not None:
            ref_error = compute_relative_error(ref, closest_idx)
            if ref_error is not None:
                ref_errors.append(ref_error)
        
        if score is not None and ground_truth_score is not None:
            score_error = compute_relative_error(score, ground_truth_score)
            if score_error is not None:
                score_errors.append(score_error)
        
        # 按trajectory分组
        if 'meta_data' in data and 'id' in data['meta_data']:
            traj_id = data['meta_data']['id']
            trajectories[traj_id].append({
                'ref': ref,
                'score': score,
                'closest_idx': closest_idx,
                'ground_truth_score': ground_truth_score,
                'format_score': format_score
            })
    
    # 打印基础统计
    print("\n### Basic Statistics ###")
    print(f"Format Score (符合格式比例): {np.mean(format_scores):.2%}")
    print(f"  - Correct format: {sum(format_scores)}/{len(format_scores)}")
    
    if ref_errors:
        print(f"\nRef vs Closest_idx Relative Error:")
        print(f"  - Mean: {np.mean(ref_errors):.4f}")
        print(f"  - Median: {np.median(ref_errors):.4f}")
        print(f"  - Std: {np.std(ref_errors):.4f}")
        print(f"  - Valid samples: {len(ref_errors)}/{len(all_data)}")
    
    if score_errors:
        print(f"\nScore vs Ground_truth_score Relative Error:")
        print(f"  - Mean: {np.mean(score_errors):.4f}")
        print(f"  - Median: {np.median(score_errors):.4f}")
        print(f"  - Std: {np.std(score_errors):.4f}")
        print(f"  - Valid samples: {len(score_errors)}/{len(all_data)}")
    
    # 计算Spearman correlation和Negative Rate
    print("\n" + "=" * 80)
    print("### Per-Trajectory Analysis ###")
    print(f"Total trajectories: {len(trajectories)}")
    
    spearman_correlations = []
    negative_rates = []
    
    for traj_id, samples in trajectories.items():
        # 按ground_truth_score排序
        sorted_samples = sorted(samples, key=lambda x: x['ground_truth_score'] if x['ground_truth_score'] is not None else -1)
        
        # 提取GT数组（closest_idx）
        gt_array = [s['closest_idx'] for s in sorted_samples if s['closest_idx'] is not None]
        
        # 提取预测数组（ref）
        pred_array = [s['ref'] for s in sorted_samples if s['ref'] is not None]
        
        # 提取score数组用于计算negative rate
        score_array = [s['score'] for s in sorted_samples if s['score'] is not None]
        
        # 计算Spearman correlation
        if len(gt_array) >= 2 and len(pred_array) >= 2 and len(gt_array) == len(pred_array):
            corr, p_value = spearmanr(gt_array, pred_array)
            if not np.isnan(corr):
                spearman_correlations.append(corr)
        
        # 计算Negative Rate（score的变化）
        if len(score_array) >= 2:
            score_diffs = [score_array[i+1] - score_array[i] for i in range(len(score_array)-1)]
            neg_rate = compute_negative_rate(score_diffs)
            negative_rates.append(neg_rate)
    
    if spearman_correlations:
        print(f"\nSpearman Correlation (ref vs closest_idx ordering):")
        print(f"  - Mean: {np.mean(spearman_correlations):.4f}")
        print(f"  - Median: {np.median(spearman_correlations):.4f}")
        print(f"  - Std: {np.std(spearman_correlations):.4f}")
        print(f"  - Valid trajectories: {len(spearman_correlations)}/{len(trajectories)}")
    
    if negative_rates:
        print(f"\nNegative Rate (score decreasing ratio):")
        print(f"  - Mean: {np.mean(negative_rates):.4f}")
        print(f"  - Median: {np.median(negative_rates):.4f}")
        print(f"  - Std: {np.std(negative_rates):.4f}")
        print(f"  - Valid trajectories: {len(negative_rates)}/{len(trajectories)}")
    
    # 详细的轨迹分析（可选，显示前5个）
    print("\n" + "=" * 80)
    print("### Sample Trajectory Details (first 5) ###")
    for idx, (traj_id, samples) in enumerate(list(trajectories.items())[:5]):
        print(f"\nTrajectory {idx+1}: {traj_id}")
        sorted_samples = sorted(samples, key=lambda x: x['ground_truth_score'] if x['ground_truth_score'] is not None else -1)
        
        gt_array = [s['closest_idx'] for s in sorted_samples if s['closest_idx'] is not None]
        pred_array = [s['ref'] for s in sorted_samples if s['ref'] is not None]
        
        print(f"  GT array (closest_idx):  {gt_array}")
        print(f"  Pred array (ref):        {pred_array}")
        
        if len(gt_array) == len(pred_array) and len(gt_array) >= 2:
            corr, _ = spearmanr(gt_array, pred_array)
            print(f"  Spearman correlation: {corr:.4f}")
    
    # 汇总结果
    print("\n" + "=" * 80)
    print("### Summary ###")
    results = {
        'format_score': np.mean(format_scores),
        'ref_relative_error_mean': np.mean(ref_errors) if ref_errors else None,
        'score_relative_error_mean': np.mean(score_errors) if score_errors else None,
        'spearman_correlation_mean': np.mean(spearman_correlations) if spearman_correlations else None,
        'negative_rate_mean': np.mean(negative_rates) if negative_rates else None,
    }
    
    for key, value in results.items():
        if value is not None:
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: N/A")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluate_jsonl.py <jsonl_file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    evaluate_jsonl(file_path)

# python /projects/b1222/userdata/jianshu/chengxuan/ProgressLM/eval/eval_results/eval_results.py /projects/b1222/userdata/jianshu/chengxuan/saved/eval_results/sft_3b_visual/tiny/raw/eval_sft3b_visual_20251023_182205.jsonl