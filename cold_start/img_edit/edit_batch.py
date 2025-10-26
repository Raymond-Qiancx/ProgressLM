#!/usr/bin/env python
"""
Qwen-Image-Edit Multi-GPU Processing with Global Progress Tracking
"""

import os
import json
import torch
import logging
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from accelerate import PartialState
from diffusers import QwenImageEditPipeline
import time
from tqdm import tqdm
import multiprocessing

# Set up multiprocessing manager for shared progress (only used for total count)
manager = multiprocessing.Manager()
shared_progress = manager.dict()
shared_progress['total'] = 0

@dataclass
class EditTask:
    prompt: str
    image_path: str
    task_id: str
    meta_data: Dict
    output_path: Optional[str] = None

class PerGPUProgressBar:
    """Per-GPU progress bar - each GPU shows its own progress on a fixed line"""

    def __init__(self, total: int, gpu_id: int, num_processes: int, initial: int = 0):
        self.total = total
        self.gpu_id = gpu_id
        self.num_processes = num_processes
        self.initial = initial
        self.pbar = None
        self.completed = initial  # Start from already completed count
        self.failed = 0

    def __enter__(self):
        # Each GPU creates its own progress bar at a fixed position
        self.pbar = tqdm(
            total=self.total,
            initial=self.initial,  # Start progress bar from already completed tasks
            position=self.gpu_id,
            desc=f"GPU {self.gpu_id}",
            unit="img",
            leave=True,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        self.pbar.set_postfix({'errors': 0})
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pbar:
            self.pbar.close()

    def update(self, success: bool = True):
        """Update this GPU's progress bar"""
        if self.pbar:
            if success:
                self.completed += 1
            else:
                self.failed += 1

            self.pbar.update(1)
            self.pbar.set_postfix({'errors': self.failed})
            self.pbar.refresh()

class QwenImageEditProcessor:
    def __init__(
        self,
        model_path: str,
        image_dir: str,
        save_dir: str,
        checkpoint_dir: str = "./checkpoints",
        enable_checkpoint: bool = True,
        max_retries: int = 2
    ):
        self.model_path = model_path
        self.image_dir = Path(image_dir)
        self.save_dir = Path(save_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.enable_checkpoint = enable_checkpoint
        self.max_retries = max_retries
        
        # Initialize distributed state
        self.distributed_state = PartialState()
        self.device = self.distributed_state.device
        self.gpu_id = self.distributed_state.process_index
        self.num_processes = self.distributed_state.num_processes
        
        self.logger = self._setup_logger()
        
        # Create directories on main process
        if self.gpu_id == 0:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
        self.distributed_state.wait_for_everyone()
        
        # Load checkpoint if enabled
        self.completed_tasks = set()
        if self.enable_checkpoint:
            self._load_checkpoint()
            
        # Load model
        self.pipeline = self._load_pipeline()
        
    def _setup_logger(self):
        logger = logging.getLogger(f"GPU_{self.gpu_id}")
        logger.setLevel(logging.INFO)

        # Remove existing handlers to avoid duplication
        logger.handlers.clear()

        # Create handler with custom format
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            f'[%(asctime)s][GPU:{self.gpu_id}] %(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger
    
    def _load_checkpoint(self):
        checkpoint_file = self.checkpoint_dir / f"checkpoint_gpu_{self.gpu_id}.json"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
                self.completed_tasks = set(data.get('completed_tasks', []))
                self.logger.info(f"Loaded {len(self.completed_tasks)} completed tasks from checkpoint")
    
    def _save_checkpoint(self, task_id: str):
        if not self.enable_checkpoint:
            return
        self.completed_tasks.add(task_id)
        checkpoint_file = self.checkpoint_dir / f"checkpoint_gpu_{self.gpu_id}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump({'completed_tasks': list(self.completed_tasks)}, f)
    
    def _load_pipeline(self):
        self.logger.info(f"Loading pipeline on {self.device}")
        
        pipeline = QwenImageEditPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16
        )
        
        pipeline.to(self.device)
        pipeline.set_progress_bar_config(disable=True)
        
        # Enable memory optimizations
        if hasattr(pipeline, 'enable_attention_slicing'):
            pipeline.enable_attention_slicing()
        if hasattr(pipeline, 'enable_vae_slicing'):
            pipeline.enable_vae_slicing()
            
        self.logger.info("Pipeline loaded successfully")
        return pipeline
    
    def load_tasks(self, jsonl_path: str) -> List[EditTask]:
        tasks = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    image_path = self.image_dir / data['meta_data']['id'] / data['meta_data']['image']
                    
                    task = EditTask(
                        prompt=data['prompt'],
                        image_path=str(image_path),
                        task_id=f"{data['meta_data']['id']}_{line_num}",
                        meta_data=data['meta_data']
                    )
                    
                    # Skip if already completed
                    if task.task_id in self.completed_tasks:
                        self.logger.info(f"Skipping completed task: {task.task_id}")
                        continue
                        
                    tasks.append(task)
                    
                except Exception as e:
                    self.logger.error(f"Error loading line {line_num}: {e}")
        
        return tasks
    
    def process_single_task(self, task: EditTask, progress_bar: PerGPUProgressBar) -> Dict:
        """Process single task with retries and immediate save"""
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                # Check if image exists
                if not Path(task.image_path).exists():
                    raise FileNotFoundError(f"Image not found: {task.image_path}")
                
                # Load and process image
                image = Image.open(task.image_path).convert("RGB")
                
                inputs = {
                    "image": image,
                    "prompt": task.prompt,
                    "generator": torch.Generator(device=self.device).manual_seed(42),
                    "true_cfg_scale": 4.0,
                    "negative_prompt": " ",
                    "num_inference_steps": 50,
                }
                
                # Run inference
                with torch.inference_mode():
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        output = self.pipeline(**inputs)
                        output_image = output.images[0]
                
                # Save with original directory structure
                relative_path = Path(task.meta_data['id'])
                original_name = Path(task.meta_data['image']).stem
                original_ext = Path(task.meta_data['image']).suffix
                
                # Create output directory
                output_subdir = self.save_dir / relative_path
                output_subdir.mkdir(parents=True, exist_ok=True)
                
                # Save edited image immediately
                output_filename = f"{original_name}_edit{original_ext}"
                output_path = output_subdir / output_filename
                output_image.save(output_path)
                
                # Save checkpoint
                self._save_checkpoint(task.task_id)
                
                # Update progress
                progress_bar.update(success=True)
                
                return {
                    'task_id': task.task_id,
                    'status': 'success',
                    'output_path': str(output_path),
                    'processing_time': time.time() - start_time,
                    'gpu_id': self.gpu_id
                }
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    self.logger.error(f"Task {task.task_id} failed after {self.max_retries} attempts: {e}")
                    progress_bar.update(success=False)
                    return {
                        'task_id': task.task_id,
                        'status': 'failed',
                        'error': str(e),
                        'gpu_id': self.gpu_id
                    }
                else:
                    self.logger.warning(f"Attempt {attempt+1} failed for {task.task_id}: {e}")
                    torch.cuda.empty_cache()
                    time.sleep(1)
    
    def _count_completed_from_checkpoints(self) -> int:
        """Count total completed tasks from all GPU checkpoint files"""
        total_completed = 0
        for i in range(self.num_processes):
            checkpoint_file = self.checkpoint_dir / f"checkpoint_gpu_{i}.json"
            if checkpoint_file.exists():
                try:
                    with open(checkpoint_file, 'r') as f:
                        data = json.load(f)
                        completed_count = len(data.get('completed_tasks', []))
                        total_completed += completed_count
                        if self.gpu_id == 0:
                            self.logger.info(f"GPU {i}: {completed_count} tasks completed")
                except Exception as e:
                    if self.gpu_id == 0:
                        self.logger.warning(f"Error reading checkpoint for GPU {i}: {e}")
        return total_completed

    def process_batch(self, jsonl_path: str):
        """Process batch with global progress tracking"""
        # Load all tasks (without checkpoint filtering first)
        all_tasks_raw = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    image_path = self.image_dir / data['meta_data']['id'] / data['meta_data']['image']

                    task = EditTask(
                        prompt=data['prompt'],
                        image_path=str(image_path),
                        task_id=f"{data['meta_data']['id']}_{line_num}",
                        meta_data=data['meta_data']
                    )
                    all_tasks_raw.append(task)
                except Exception as e:
                    self.logger.error(f"Error loading line {line_num}: {e}")

        # Only GPU 0 reads all checkpoint files and initializes shared progress
        if self.gpu_id == 0:
            total_already_completed = self._count_completed_from_checkpoints()
            shared_progress['total'] = len(all_tasks_raw)
            if total_already_completed > 0:
                self.logger.info(f"Resuming from checkpoint: {total_already_completed}/{len(all_tasks_raw)} tasks already completed")

        # Wait for GPU 0 to initialize shared progress
        self.distributed_state.wait_for_everyone()

        # Get total from shared memory
        total_tasks = shared_progress['total']

        # Split ALL tasks between GPUs first (before filtering)
        with self.distributed_state.split_between_processes(all_tasks_raw) as tasks_for_this_gpu:
            # Convert to list to work with
            tasks_assigned = list(tasks_for_this_gpu)
            total_assigned = len(tasks_assigned)

            # Now filter out completed tasks for this GPU
            tasks_to_process = []
            completed_count = 0
            for task in tasks_assigned:
                if task.task_id in self.completed_tasks:
                    self.logger.info(f"Skipping completed task: {task.task_id}")
                    completed_count += 1
                else:
                    tasks_to_process.append(task)

            self.logger.info(f"Assigned {total_assigned} tasks, {completed_count} already completed, {len(tasks_to_process)} to process")

            results = []

            # Create per-GPU progress bar - total is all assigned tasks, initial is already completed
            with PerGPUProgressBar(total_assigned, self.gpu_id, self.num_processes, initial=completed_count) as progress_bar:
                # Process each task
                for task in tasks_to_process:
                    result = self.process_single_task(task, progress_bar)
                    results.append(result)
                    
                    # Clear cache periodically
                    if len(results) % 10 == 0:
                        torch.cuda.empty_cache()
            
            # Save results
            output_file = self.save_dir / f"results_gpu_{self.gpu_id}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved {len(results)} results to {output_file}")
            
            # Wait for all GPUs
            self.distributed_state.wait_for_everyone()
            
            # Aggregate results on GPU 0
            if self.gpu_id == 0:
                self._aggregate_results()
    
    def _aggregate_results(self):
        """Aggregate results from all GPUs and print final statistics"""
        all_results = []
        for i in range(self.num_processes):
            result_file = self.save_dir / f"results_gpu_{i}.json"
            if result_file.exists():
                with open(result_file, 'r') as f:
                    all_results.extend(json.load(f))
        
        # Save aggregated results
        with open(self.save_dir / "all_results.json", 'w') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        # Calculate statistics
        success = sum(1 for r in all_results if r['status'] == 'success')
        failed = sum(1 for r in all_results if r['status'] == 'failed')
        total_time = sum(r.get('processing_time', 0) for r in all_results)
        
        print(f"\n{'='*60}")
        print(f"{'PROCESSING COMPLETE':^60}")
        print(f"{'='*60}")
        print(f"Total tasks:       {len(all_results):>10}")
        print(f"Successful:        {success:>10} ({success/len(all_results)*100:.1f}%)")
        print(f"Failed:            {failed:>10} ({failed/len(all_results)*100:.1f}%)")
        print(f"Total time:        {total_time:>10.1f}s")
        print(f"Avg time/task:     {total_time/len(all_results):>10.2f}s")
        print(f"Throughput:        {len(all_results)/total_time:>10.2f} img/s")
        print(f"{'='*60}\n")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Qwen-Image-Edit Multi-GPU Processing")
    parser.add_argument("--jsonl", type=str, default="/projects/b1222/userdata/jianshu/chengxuan/ProgressLM/negative_samples.jsonl", help="Input JSONL file")
    parser.add_argument("--model-path", type=str, default="/projects/p32958/chengxuan/models/Qwen-Image-Edit", help="Model path")
    parser.add_argument("--image-dir", type=str, default="/projects/b1222/userdata/jianshu/chengxuan/ProgressLM/data/images", help="Image directory")
    parser.add_argument("--save-dir", type=str, default="/projects/p32958/chengxuan/results/progresslm/negative/image", help="Save directory for edited images")
    parser.add_argument("--checkpoint-dir", type=str, default="/projects/p32958/chengxuan/results/progresslm/negative/ckpt", help="Checkpoint directory")
    parser.add_argument("--no-checkpoint", action="store_true", help="Disable checkpoint")
    parser.add_argument("--max-retries", type=int, default=2, help="Max retries for failed tasks")
    
    args = parser.parse_args()
    
    processor = QwenImageEditProcessor(
        model_path=args.model_path,
        image_dir=args.image_dir,
        save_dir=args.save_dir,
        checkpoint_dir=args.checkpoint_dir,
        enable_checkpoint=not args.no_checkpoint,
        max_retries=args.max_retries
    )
    
    processor.process_batch(args.jsonl)

if __name__ == "__main__":
    main()