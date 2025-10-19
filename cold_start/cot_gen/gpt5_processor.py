#!/usr/bin/env python3
"""
GPT-5-mini Real-time API Processor for Visual Task Progress Evaluation

功能特性：
1. LIMIT: 限制处理的样本数量
2. RESUME: 断点续传功能，可以从上次中断的地方继续处理
3. 唯一标识: 使用 "id"_"progress_score" 作为样本的唯一标识

成功输出格式：
{
    "ref": "2",                              # 从<ref>标签提取的参考帧编号
    "score": "8%",                           # 从<score>标签提取的进度分数
    "closest_idx": "1",                      # 输入数据中的最近索引
    "ground_truth_score": "8%",              # 输入数据中的真实分数
    "response": "完整的GPT-5响应...",        # 包含所有标签的完整响应
    "meta_data": {                           # 元数据信息
        "id": "样本ID",
        "task_goal": "任务描述",
        "tokens_used": 2500,
        "model": "gpt-5-mini",
        "timestamp": "2025-01-17T10:30:45",
        "status": "success"
    }
}

错误输出格式：
{
    "ref": null,
    "score": null,
    "closest_idx": "1",
    "ground_truth_score": "8%",
    "response": null,
    "meta_data": {
        "id": "样本ID",
        "task_goal": "任务描述",
        "error": "错误信息",
        "traceback": "完整堆栈追踪",
        "timestamp": "2025-01-17T10:30:45",
        "status": "error"
    }
}
"""

import json
import os
import sys
import time
import base64
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import argparse
from tqdm import tqdm
import traceback
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 全局锁用于文件写入
write_lock = threading.Lock()

class VisualProgressProcessor:
    """视觉进度评估处理器"""
    
    def __init__(self, api_key: str, image_dir: str, model: str = "gpt-5-mini"):
        """
        初始化处理器
        
        Args:
            api_key: OpenAI API密钥
            image_dir: 图像基础目录
            model: 使用的模型
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.image_dir = Path(image_dir)
        
        if not self.image_dir.exists():
            raise ValueError(f"图像目录不存在: {image_dir}")
    
    def encode_image(self, image_path: Path) -> str:
        """
        将图像编码为base64
        
        Args:
            image_path: 图像路径
        
        Returns:
            base64编码的图像字符串
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            raise ValueError(f"无法读取图像 {image_path}: {str(e)}")
    
    def build_image_content(self, image_path: Path) -> Dict:
        """
        构建图像消息内容
        
        Args:
            image_path: 图像路径
        
        Returns:
            OpenAI API格式的图像内容
        """
        base64_image = self.encode_image(image_path)
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": "high"  # 使用高质量图像分析
            }
        }
    
    def calculate_progress_scores(self, total_steps: int) -> List[str]:
        """
        根据总步数计算进度分数
        
        Args:
            total_steps: 总步数
        
        Returns:
            进度百分比列表
        """
        scores = ["0%"]
        if total_steps > 0:
            step_size = 100 / total_steps
            for i in range(1, total_steps + 1):
                scores.append(f"{int(i * step_size)}%")
        return scores
    
    def build_message_content(self, sample: Dict) -> List[Dict]:
        """
        构建完整的消息内容
        
        Args:
            sample: JSONL中的一个样本
        
        Returns:
            消息内容列表
        """
        content = []
        
        # 1. 系统提示词的第一部分
        system_prompt = (
            "You are an expert AI analyst specializing in visual task-progress evaluations "
            "Your objective is not to estimate from scratch. "
            "Instead, your task is to construct a perfect, human-like chain of thought that "
            "logically explains and justifies a known, ground-truth progress score. "
            "Your entire response must read as if you are deducing the conclusion independently "
            "from visual analysis alone. This is the system prompt for normal inference. "
            "You are a progress estimator specializing in evaluating the progress of an ongoing "
            "task based on visual evidence. The demonstration consists of a sequence of video "
            "frames (images) showing how the task evolves from 0% (start) to 100% (completion). "
            "Your goal is to produce a human-like reasoning chain that logically supports the "
            "given progress score. Here is the demonstration:"
        )
        content.append({"type": "text", "text": system_prompt})
        
        # 2. 添加visual_demo中的所有图片
        sample_id = sample['id']
        for demo_image in sample['visual_demo']:
            image_path = self.image_dir / sample_id / demo_image
            if not image_path.exists():
                raise FileNotFoundError(f"演示图像不存在: {image_path}")
            content.append(self.build_image_content(image_path))
        
        # 3. 构建进度转换文本
        total_steps = int(sample['total_steps'])
        progress_scores = self.calculate_progress_scores(total_steps)
        
        progress_text = "The progress shifts across all given visual demos is: "
        for i, score in enumerate(progress_scores):
            if i > 0:
                progress_text += " "
            progress_text += f"<image> {score}"
        
        content.append({"type": "text", "text": progress_text})
        
        # 4. 添加当前状态提示
        content.append({
            "type": "text", 
            "text": "Here is the current state that you need to estimate:"
        })
        
        # 5. 添加stage_to_estimate图片
        stage_image = sample['stage_to_estimate'][0]  # 假设只有一张图片
        stage_path = self.image_dir / sample_id / stage_image
        if not stage_path.exists():
            raise FileNotFoundError(f"评估图像不存在: {stage_path}")
        content.append(self.build_image_content(stage_path))
        
        # 6. 添加关键规则和ground truth
        critical_rule = (
            f"**Critical Rule** The correct final progress score will be provided to you. "
            f"However, you must **never** reveal or imply that you already know the answer. "
            f"Your reasoning must appear as a fully original, independent visual analysis "
            f"derived from the images.\n\n"
            f"**Ground-Truth Progress Result**\n"
            f"Closest Reference Frame: The No. {sample['closest_idx']} demo image is the most relevant frame\n"
            f"Final Progress Score to Justify: {sample['progress_score']}"
        )
        content.append({"type": "text", "text": critical_rule})
        
        # 7. 添加任务说明和输出格式
        task_instructions = (
            "\nYour task:\n"
            "1. Analyze the demonstration images to understand how the task visually progresses from start to completion.\n"
            "2. Identify which frame in the provided visual demos is visually most similar to the current state image.\n"
            "3. Compare the current state to that reference frame and determine whether it shows more or less progress.\n"
            "4. Finally, provide a numeric progress estimation between 0% and 100%.\n\n"
            "**Output Format**\n"
            "Your response must strictly follow this format:\n"
            "<ref_think>Your reasoning for choosing the closest demonstration frame as the reference</ref_think>\n"
            "<ref>identify which image is most visually similar to the current state, and output only the number of that image</ref>\n"
            "<score_think>Your reasoning for comparing the current state image with the reference frame(s)</score_think>\n"
            "<score>Your final estimated progress score here</score>"
        )
        content.append({"type": "text", "text": task_instructions})
        
        return content
        
        # 4. 添加当前状态提示
        content.append({
            "type": "text", 
            "text": "Here is the current state that you need to estimate:"
        })
        
        # 5. 添加stage_to_estimate图片
        stage_image = sample['stage_to_estimate'][0]  # 假设只有一张图片
        stage_path = self.image_dir / sample_id / stage_image
        if not stage_path.exists():
            raise FileNotFoundError(f"评估图像不存在: {stage_path}")
        content.append(self.build_image_content(stage_path))
        
        # 6. 添加关键规则和ground truth
        critical_rule = (
            f"**Critical Rule** The correct final progress score will be provided to you. "
            f"However, you must **never** reveal or imply that you already know the answer. "
            f"Your reasoning must appear as a fully original, independent visual analysis "
            f"derived from the images.\n\n"
            f"**Ground-Truth Progress Result**\n"
            f"Closest Reference Frame: The No. {sample['closest_idx']} demo image is the most relevant frame\n"
            f"Final Progress Score to Justify: {sample['progress_score']}"
        )
        content.append({"type": "text", "text": critical_rule})
        
        # 7. 添加任务说明和输出格式
        task_instructions = (
            "\nYour task:\n"
            "1. Analyze the demonstration images to understand how the task visually progresses from start to completion.\n"
            "2. Identify which frame in the provided visual demos is visually most similar to the current state image.\n"
            "3. Compare the current state to that reference frame and determine whether it shows more or less progress.\n"
            "4. Finally, provide a numeric progress estimation between 0% and 100%.\n\n"
            "**Output Format**\n"
            "Your response must strictly follow this format:\n"
            "<ref_think>Your reasoning for choosing the closest demonstration frame as the reference</ref_think>\n"
            "<ref>identify which image is most visually similar to the current state, and output only the number of that image</ref>\n"
            "<score_think>Your reasoning for comparing the current state image with the reference frame(s)</score_think>\n"
            "<score>Your final estimated progress score here</score>"
        )
        content.append({"type": "text", "text": task_instructions})
        
        return content
    
    def get_sample_unique_id(self, sample: Dict) -> str:
        """
        生成样本的唯一标识
        
        Args:
            sample: JSONL中的一个样本
        
        Returns:
            唯一标识字符串: id_progress_score
        """
        sample_id = sample.get('id', 'unknown')
        progress_score = sample.get('progress_score', 'unknown')
        return f"{sample_id}_{progress_score}"
    
    def load_processed_ids(self, output_file: Path) -> set:
        """
        从输出文件加载已处理的样本ID
        
        Args:
            output_file: 输出文件路径
        
        Returns:
            已处理的唯一ID集合
        """
        processed_ids = set()
        
        if not output_file.exists():
            return processed_ids
        
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        result = json.loads(line.strip())
                        # 从meta_data中重建unique_id
                        if 'meta_data' in result:
                            sample_id = result['meta_data'].get('id', 'unknown')
                            progress_score = result.get('ground_truth_score', 'unknown')
                            unique_id = f"{sample_id}_{progress_score}"
                            processed_ids.add(unique_id)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"⚠️  读取已处理文件时出错: {str(e)}")
        
        return processed_ids
    
    def extract_tags(self, response: str) -> Dict[str, str]:
        """
        从响应中提取特定标签的内容
        
        Args:
            response: GPT-5的响应文本
        
        Returns:
            包含提取内容的字典
        """
        import re
        
        extracted = {}
        
        # 提取<ref>标签内容
        ref_match = re.search(r'<ref>(.*?)</ref>', response, re.DOTALL)
        extracted['ref'] = ref_match.group(1).strip() if ref_match else None
        
        # 提取<score>标签内容
        score_match = re.search(r'<score>(.*?)</score>', response, re.DOTALL)
        extracted['score'] = score_match.group(1).strip() if score_match else None
        
        return extracted
    
    def process_single_sample(self, sample: Dict) -> Dict:
        """
        处理单个样本
        
        Args:
            sample: JSONL中的一个样本
        
        Returns:
            处理结果
        """
        try:
            # 构建消息内容
            message_content = self.build_message_content(sample)
            
            # 调用GPT-5 API
            # 注意：GPT-5使用max_completion_tokens而不是max_tokens
            api_params = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": message_content
                    }
                ],
                "temperature": 1,
                "max_completion_tokens": 3000  # GPT-5使用max_completion_tokens
            }
            
            # 添加GPT-5特有参数（如果支持）
            # 注意：如果这些参数导致错误，可以注释掉
            if self.model.startswith("gpt-5"):
                # api_params["verbosity"] = "medium"  # 如果不支持，注释此行
                # api_params["reasoning_effort"] = "medium"  # 如果不支持，注释此行
                pass  # 暂时不添加特殊参数，避免兼容性问题
            
            response = self.client.chat.completions.create(**api_params)
            
            # 提取响应
            assistant_response = response.choices[0].message.content
            
            # 提取标签内容
            extracted = self.extract_tags(assistant_response)
            
            # 构建输出结果 - 新格式
            result = {
                "ref": extracted.get('ref'),
                "score": extracted.get('score'),
                "closest_idx": sample["closest_idx"],
                "ground_truth_score": sample["progress_score"],
                "response": assistant_response,
                "meta_data": {
                    "id": sample["id"],
                    "task_goal": sample["task_goal"],
                    "tokens_used": response.usage.total_tokens,
                    "model": self.model,
                    "timestamp": datetime.now().isoformat(),
                    "status": "success"
                }
            }
            
            return result
            
        except Exception as e:
            # 错误处理
            error_msg = f"处理失败: {str(e)}"
            if hasattr(e, 'response'):
                error_msg += f"\nAPI响应: {e.response}"
            
            return {
                "ref": None,
                "score": None,
                "closest_idx": sample.get("closest_idx", ""),
                "ground_truth_score": sample.get("progress_score", ""),
                "response": None,
                "meta_data": {
                    "id": sample["id"],
                    "task_goal": sample.get("task_goal", ""),
                    "error": error_msg,
                    "traceback": traceback.format_exc(),
                    "timestamp": datetime.now().isoformat(),
                    "status": "error"
                }
            }
    
    def save_result(self, result: Dict, output_file: Path):
        """
        保存单个结果到JSONL文件（线程安全）
        
        Args:
            result: 处理结果
            output_file: 输出文件路径
        """
        with write_lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    def process_batch(self, 
                     input_file: str, 
                     output_file: str,
                     max_workers: int = 5,
                     retry_failed: bool = True,
                     limit: int = None,
                     resume: bool = False):
        """
        批量处理JSONL文件
        
        Args:
            input_file: 输入JSONL文件路径
            output_file: 输出JSONL文件路径
            max_workers: 最大并发数
            retry_failed: 是否重试失败的样本
            limit: 限制处理的样本数量
            resume: 是否启用断点续传
        """
        # 加载输入数据
        all_samples = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                all_samples.append(json.loads(line.strip()))
        
        print(f"📊 总共加载了 {len(all_samples)} 个样本")
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 断点续传：加载已处理的样本
        processed_ids = set()
        if resume:
            processed_ids = self.load_processed_ids(output_path)
            if processed_ids:
                print(f"🔄 断点续传模式：发现 {len(processed_ids)} 个已处理样本")
        else:
            # 非续传模式，清空输出文件
            if output_path.exists():
                output_path.unlink()
                print(f"🗑️  已清空现有输出文件")
        
        # 过滤出需要处理的样本
        samples_to_process = []
        skipped_count = 0
        
        for sample in all_samples:
            unique_id = self.get_sample_unique_id(sample)
            if unique_id in processed_ids:
                skipped_count += 1
                continue
            samples_to_process.append(sample)
            
            # 如果设置了limit，检查是否达到限制
            if limit and len(samples_to_process) >= limit:
                break
        
        if skipped_count > 0:
            print(f"⏭️  跳过 {skipped_count} 个已处理样本")
        
        if limit:
            print(f"🎯 限制处理数量: {limit}")
            samples_to_process = samples_to_process[:limit]
        
        samples = samples_to_process
        
        if not samples:
            print(f"✅ 没有需要处理的新样本")
            return 0, 0
        
        print(f"🚀 开始处理 {len(samples)} 个样本 (并发数: {max_workers})")
        
        # 统计
        success_count = 0
        error_count = 0
        total_tokens = 0
        failed_samples = []
        
        # 使用线程池并发处理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_sample = {
                executor.submit(self.process_single_sample, sample): sample 
                for sample in samples
            }
            
            # 使用tqdm显示进度
            desc = "续传进度" if resume else "处理进度"
            with tqdm(total=len(samples), desc=desc) as pbar:
                for future in as_completed(future_to_sample):
                    sample = future_to_sample[future]
                    
                    try:
                        result = future.result(timeout=60)  # 60秒超时
                        
                        # 保存结果
                        self.save_result(result, output_path)
                        
                        # 更新统计
                        if result['meta_data']['status'] == 'success':
                            success_count += 1
                            total_tokens += result['meta_data'].get('tokens_used', 0)
                            pbar.set_postfix({
                                '✅': success_count,
                                '❌': error_count,
                                'tokens': total_tokens
                            })
                        else:
                            error_count += 1
                            failed_samples.append(sample)
                            pbar.set_postfix({
                                '✅': success_count,
                                '❌': error_count,
                                'tokens': total_tokens,
                                'last_error': result['meta_data'].get('error', '')[:50]
                            })
                        
                    except Exception as e:
                        error_count += 1
                        failed_samples.append(sample)
                        error_result = {
                            "ref": None,
                            "score": None,
                            "closest_idx": sample.get("closest_idx", ""),
                            "ground_truth_score": sample.get("progress_score", ""),
                            "response": None,
                            "meta_data": {
                                "id": sample.get("id", "unknown"),
                                "error": f"执行超时或异常: {str(e)}",
                                "timestamp": datetime.now().isoformat(),
                                "status": "error"
                            }
                        }
                        self.save_result(error_result, output_path)
                        pbar.set_postfix({
                            '✅': success_count,
                            '❌': error_count,
                            'timeout': True
                        })
                    
                    pbar.update(1)
        
        # 重试失败的样本（如果需要）
        if retry_failed and failed_samples:
            print(f"\n🔄 重试 {len(failed_samples)} 个失败的样本...")
            retry_success = 0
            
            with tqdm(total=len(failed_samples), desc="重试进度") as pbar:
                for sample in failed_samples:
                    time.sleep(1)  # 避免速率限制
                    result = self.process_single_sample(sample)
                    self.save_result(result, output_path)
                    
                    if result['meta_data']['status'] == 'success':
                        retry_success += 1
                        success_count += 1
                        error_count -= 1
                        total_tokens += result['meta_data'].get('tokens_used', 0)
                    
                    pbar.update(1)
                    pbar.set_postfix({'重试成功': retry_success})
        
        # 打印最终统计
        total_processed = success_count + error_count
        if resume and processed_ids:
            print(f"\n📊 本次处理统计:")
            print(f"  🔄 之前已处理: {len(processed_ids)}")
            print(f"  ✨ 本次处理: {total_processed}")
            print(f"    - ✅ 成功: {success_count}")
            print(f"    - ❌ 失败: {error_count}")
            print(f"  📈 累计处理: {len(processed_ids) + total_processed}")
        else:
            print(f"\n📊 处理完成统计:")
            print(f"  ✅ 成功: {success_count}/{total_processed}")
            print(f"  ❌ 失败: {error_count}/{total_processed}")
        
        print(f"  💰 本次Token使用: {total_tokens:,}")
        print(f"  📄 结果保存至: {output_path}")
        
        # 计算估算成本（基于GPT-5-mini价格）
        # 注意：这里简化计算，实际上输入和输出token应该分开计算
        input_cost = total_tokens * 0.25 / 1_000_000  # $0.25 per 1M input tokens
        output_cost = total_tokens * 2.0 / 1_000_000  # $2.00 per 1M output tokens
        estimated_cost = input_cost + output_cost
        print(f"  💵 本次估算成本: ${estimated_cost:.4f} (简化计算)")
        
        return success_count, error_count


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="GPT-5-mini Visual Progress Evaluation Processor"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="OpenAI API密钥"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入JSONL文件路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出JSONL文件路径"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        required=True,
        help="图像基础目录路径"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        choices=["gpt-5", "gpt-5-mini", "gpt-5-nano"],
        help="使用的GPT-5模型版本"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="最大并发处理数（默认: 5）"
    )
    parser.add_argument(
        "--no-retry",
        action="store_true",
        help="不重试失败的样本"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="限制处理的样本数量"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="启用断点续传（从上次中断处继续）"
    )
    
    args = parser.parse_args()
    
    # 验证输入文件
    if not Path(args.input).exists():
        print(f"❌ 输入文件不存在: {args.input}")
        sys.exit(1)
    
    # 验证图像目录
    if not Path(args.image_dir).exists():
        print(f"❌ 图像目录不存在: {args.image_dir}")
        sys.exit(1)
    
    # 创建处理器
    try:
        processor = VisualProgressProcessor(
            api_key=args.api_key,
            image_dir=args.image_dir,
            model=args.model
        )
    except Exception as e:
        print(f"❌ 初始化失败: {str(e)}")
        sys.exit(1)
    
    # 开始处理
    print(f"\n{'='*60}")
    print(f"GPT-5 Visual Progress Evaluation")
    print(f"{'='*60}")
    print(f"📁 输入文件: {args.input}")
    print(f"🖼️  图像目录: {args.image_dir}")
    print(f"🤖 模型: {args.model}")
    print(f"🔄 断点续传: {'是' if args.resume else '否'}")
    if args.limit:
        print(f"🎯 处理限制: {args.limit} 个样本")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    try:
        success_count, error_count = processor.process_batch(
            input_file=args.input,
            output_file=args.output,
            max_workers=args.max_workers,
            retry_failed=not args.no_retry,
            limit=args.limit,
            resume=args.resume
        )
        
        elapsed_time = time.time() - start_time
        print(f"\n⏱️  总耗时: {elapsed_time:.2f} 秒")
        
        # 返回适当的退出码
        if error_count == 0:
            sys.exit(0)
        elif success_count > 0:
            sys.exit(1)  # 部分成功
        else:
            sys.exit(2)  # 全部失败
            
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断处理")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ 处理过程中发生错误: {str(e)}")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()