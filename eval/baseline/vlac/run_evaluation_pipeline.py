import argparse
import json
import os
import glob
import tempfile
import shutil
import time
import threading
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from PIL import Image

# --- Utility functions (some are moved from run_eval.py) ---

def load_images_from_dir(image_paths: list[str]):
    """Loads a list of images from their absolute paths."""
    images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert('RGB')
            images.append(img)
        except Exception as e:
            # This warning will be printed from the worker process
            print(f"Warning: Failed to load {path}: {e}")
            continue
    return images

def load_trajectories_from_file(jsonl_path, image_root_dir):
    trajectories = {}
    with open(jsonl_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed JSON line in {jsonl_path}: {line.strip()}")
                continue

            parts = data['id'].split('/')
            timestamp_id = parts[-1]

            def resolve_paths(img_names):
                if isinstance(img_names, str):
                    img_names = [img_names]
                base_path = os.path.join(image_root_dir, data['id'])
                return [os.path.join(base_path, name) for name in img_names]

            if timestamp_id not in trajectories:
                trajectories[timestamp_id] = {
                    "task_goal": data["task_goal"],
                    "total_steps": int(data["total_steps"]),
                    "visual_demo_paths": resolve_paths(data["visual_demo"]),
                    "stages": []
                }
            
            progress_score_str = data["progress_score"].replace('%', '')
            trajectories[timestamp_id]["stages"].append({
                "image_path": resolve_paths(data["stage_to_estimate"])[0],
                "progress": float(progress_score_str)
            })
    return trajectories

def find_self_reference(trajectory_data):
    return trajectory_data["visual_demo_paths"]

def find_cross_reference(current_timestamp_id, search_pool):
    current_trajectory = search_pool[current_timestamp_id]
    for timestamp, trajectory_data in search_pool.items():
        if timestamp == current_timestamp_id:
            continue
        if trajectory_data["task_goal"] == current_trajectory["task_goal"]:
            return trajectory_data["visual_demo_paths"]
    for timestamp, trajectory_data in search_pool.items():
        if timestamp == current_timestamp_id:
            continue
        if trajectory_data["total_steps"] == current_trajectory["total_steps"]:
            return trajectory_data["visual_demo_paths"]
    return current_trajectory["visual_demo_paths"]

# --- Worker Process Logic ---

def pipeline_worker(
    gpu_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    model_path: str,
    model_type: str,
    passthrough_args: dict
):
    """
    A long-running worker process that initializes a model on one GPU and processes jobs.
    """
    device = f"cuda:{gpu_id}"
    # Import must be inside the worker
    from utils.model_utils import GAC_model

    # 1. Initialize model once
    try:
        critic = GAC_model(tag='critic')
        critic.init_model(
            model_path=model_path,
            model_type=model_type,
            device_map=device
        )
        # Apply relevant passthrough args to the model instance
        critic.temperature = passthrough_args.get('temperature', 0.5)
        critic.top_k = passthrough_args.get('top_k', 1)
        critic.set_config()
        critic.set_system_prompt()
    except Exception as e:
        # If model init fails, this worker is dead. Report and exit.
        result_queue.put({'status': 'ERROR', 'error': f"[GPU:{gpu_id}] Model initialization failed: {e}"})
        return

    # 2. Process jobs from the queue
    while True:
        job = task_queue.get()
        if job is None: # Sentinel value to stop
            break

        try:
            # Unpack job
            main_img_paths = job['main_img_paths']
            ref_img_paths = job['ref_img_paths']
            task = job['task']
            
            # Load images
            test_images = load_images_from_dir(main_img_paths)
            ref_images = load_images_from_dir(ref_img_paths)

            if not test_images:
                raise ValueError("Failed to load any test images.")

            # Get critic and value lists from the model
            critic_list, value_list = critic.get_trajectory_critic(
                task=task,
                image_list=test_images,
                ref_image_list=ref_images,
                batch_num=int(passthrough_args.get('batch_num', 5)),
                ref_num=int(passthrough_args.get('ref_num', 6)) if ref_images else 0,
                skip=int(passthrough_args.get('skip', 1)),
                rich=passthrough_args.get('rich', False),
                think=passthrough_args.get('think', False),
                reverse_eval=passthrough_args.get('reverse_eval', False),
                frame_skip=True,
                show_progress=False, # Disable worker tqdm bars
                verbose=False # Disable worker print statements
            )

            # Compute metrics
            voc = critic.compute_voc(value_list)
            negative_rate = critic.compute_negative_rate(critic_list)

            # Create result structure
            result = {
                'status': 'SUCCESS',
                'output_path': job['output_path'],
                'payload': {
                    'config': job['config'],
                    'metrics': {
                        'voc': float(voc),
                        'negative_rate': float(negative_rate),
                        'num_frames': len(test_images),
                        'num_steps': len(critic_list),
                    },
                    'results': {
                        'value_list': [float(v) for v in value_list],
                        'critic_list': [str(c) for c in critic_list]
                    }
                }
            }
            result_queue.put(result)

        except Exception as e:
            result_queue.put({'status': 'ERROR', 'error': f"[GPU:{gpu_id}] Failed on job {job['output_path']}: {e}"})


# --- Main Orchestrator Logic ---

def feed_the_workers(task_queue: mp.Queue, all_jobs: list, num_workers: int):
    """
    Puts all jobs and sentinel values onto the task queue in the background.
    """
    for job in all_jobs:
        task_queue.put(job)
    # Add sentinel values to stop workers
    for _ in range(num_workers):
        task_queue.put(None)

def main():
    parser = argparse.ArgumentParser(description="Main pipeline to evaluate VLAC with a persistent process pool.")
    parser.add_argument('--processed_data_dir', type=str, required=True, help="Path to the directory with split .jsonl files.")
    parser.add_argument('--image_root_dir', type=str, required=True, help="Root directory where all trajectory image folders are stored.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the VLAC model directory.")
    parser.add_argument('--output_dir', type=str, default='./results_pipeline', help="Directory to save evaluation results.")
    parser.add_argument('--cross_trajectory_ref', action='store_true', help="Enable cross-trajectory reference finding.")
    parser.add_argument('--gpu_ids', type=str, default="0", help="Comma-separated list of GPU IDs to use.")
    
    args, passthrough_list = parser.parse_known_args()
    # Convert passthrough args to a dict for easier lookup in workers
    passthrough_args = {arg.lstrip('--').replace('-', '_'): val for arg, val in zip(passthrough_list[::2], passthrough_list[1::2])}
    # Handle action_store_true args
    for arg in passthrough_list:
        if arg.startswith('--'):
            if arg not in [a+v for a,v in zip(passthrough_list[::2], passthrough_list[1::2])]:
                 passthrough_args[arg.lstrip('--').replace('-', '_')] = True


    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Gather all jobs first
    print("Gathering and preparing all trajectory jobs...")
    jsonl_files = glob.glob(os.path.join(args.processed_data_dir, '*.jsonl'))
    if not jsonl_files:
        print(f"Error: No .jsonl files found in {args.processed_data_dir}")
        return

    all_jobs = []
    for jsonl_file in jsonl_files:
        trajectories_in_task = load_trajectories_from_file(jsonl_file, args.image_root_dir)
        task_type_sanitized = os.path.basename(jsonl_file).replace('.jsonl', '')

        for timestamp_id, trajectory_data in trajectories_in_task.items():
            stages = sorted(trajectory_data["stages"], key=lambda x: x["progress"])
            main_trajectory_paths = [s["image_path"] for s in stages]

            if not main_trajectory_paths:
                continue

            if args.cross_trajectory_ref:
                ref_trajectory_paths = find_cross_reference(timestamp_id, trajectories_in_task)
            else:
                ref_trajectory_paths = find_self_reference(trajectory_data)

            if not ref_trajectory_paths:
                continue

            output_filename = f"{task_type_sanitized}_{timestamp_id}.json"
            output_path = os.path.join(args.output_dir, output_filename)

            job = {
                'main_img_paths': main_trajectory_paths,
                'ref_img_paths': ref_trajectory_paths,
                'task': trajectory_data['task_goal'],
                'output_path': output_path,
                'config': { # Config for the final JSON file
                    'model_path': args.model_path,
                    'task': trajectory_data['task_goal'],
                }
            }
            all_jobs.append(job)

    print(f"Found {len(all_jobs)} total trajectories to evaluate.")

    # 2. Set up multiprocessing context, queues, and workers
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass # Start method can only be set once

    task_queue = mp.Queue()
    result_queue = mp.Queue()
    gpu_ids = [int(gid) for gid in args.gpu_ids.split(',')]

    workers = []
    for gpu_id in gpu_ids:
        worker = mp.Process(
            target=pipeline_worker,
            args=(gpu_id, task_queue, result_queue, args.model_path, 'internvl2', passthrough_args)
        )
        worker.start()
        workers.append(worker)

    # 3. Start a background thread to feed the task queue
    # This prevents the main process from blocking if the queue gets full.
    feeder_thread = threading.Thread(
        target=feed_the_workers,
        args=(task_queue, all_jobs, len(gpu_ids))
    )
    feeder_thread.start()

    # 4. Collect results and update progress bar
    error_count = 0
    voc_scores = []
    neg_rates = []
    all_results_agg = {} # Dictionary to aggregate all results

    with tqdm(total=len(all_jobs), desc="Overall Progress", unit="traj") as pbar:
        for _ in range(len(all_jobs)):
            result = result_queue.get()

            if result['status'] == 'SUCCESS':
                # Get a unique key for the trajectory
                trajectory_key = os.path.basename(result['output_path']).replace('.json', '')
                all_results_agg[trajectory_key] = result['payload']
                
                # Update metrics
                voc = result['payload']['metrics']['voc']
                neg_rate = result['payload']['metrics']['negative_rate']
                voc_scores.append(voc)
                neg_rates.append(neg_rate)
            else:
                error_count += 1
                pbar.write(result['error'])

            # Update progress bar postfix
            avg_voc = sum(voc_scores) / len(voc_scores) if voc_scores else 0
            avg_neg_rate = sum(neg_rates) / len(neg_rates) if neg_rates else 0
            pbar.set_postfix(avg_VOC=f'{avg_voc:.3f}', avg_NegRate=f'{avg_neg_rate:.3f}', errors=error_count, refresh=True)
            pbar.update(1)

    # 5. Cleanup and Final Write
    feeder_thread.join() # Wait for the feeder thread to finish
    for worker in workers:
        worker.join()

    # Write aggregated results to a single JSON file
    final_output_path = os.path.join(args.output_dir, 'evaluation_results_all.json')
    print(f"\nAggregating results into a single file...")
    with open(final_output_path, 'w') as f:
        json.dump(all_results_agg, f, indent=2)
    print(f"All results saved to: {final_output_path}")

    # Final summary
    print("\n" + "="*80)
    print("Evaluation Summary")
    print("="*80)
    final_avg_voc = sum(voc_scores) / len(voc_scores) if voc_scores else 0
    final_avg_neg_rate = sum(neg_rates) / len(neg_rates) if neg_rates else 0
    print(f"Total Trajectories Evaluated: {len(all_jobs)}")
    print(f"Successful Evaluations: {len(voc_scores)}")
    print(f"Failed Evaluations (Errors): {error_count}")
    print(f"\nFinal Average VOC: {final_avg_voc:.4f}")
    print(f"Final Average Negative Rate: {final_avg_neg_rate:.4f}")
    print("="*80)

if __name__ == "__main__":
    main()