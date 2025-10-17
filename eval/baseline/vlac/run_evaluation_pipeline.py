import argparse
import json
import os
import glob
import subprocess
import tempfile
import shutil
import collections
from pathlib import Path

def load_trajectories_from_file(jsonl_path, image_root_dir):
    """
    Loads all trajectories from a single JSONL file.
    Each file is assumed to contain trajectories for a single task_type.
    """
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
    """Returns the trajectory's own visual demo."""
    return trajectory_data["visual_demo_paths"]

def find_cross_reference(current_timestamp_id, search_pool):
    """
    Finds a reference trajectory from a different instance within the same task type.
    """
    current_trajectory = search_pool[current_timestamp_id]

    # Priority 1: Find another trajectory with the exact same task_goal (implicitly same task_type)
    for timestamp, trajectory_data in search_pool.items():
        if timestamp == current_timestamp_id:
            continue
        # Since all trajectories in the search_pool are from the same task_type file,
        # we can be more specific if needed, but for now, any other trajectory is a valid cross-ref.
        # This implementation prioritizes by task_goal string, which should be the same for all.
        if trajectory_data["task_goal"] == current_trajectory["task_goal"]:
            print(f"  Found cross-ref for {current_timestamp_id} (same task_goal): {timestamp}")
            return trajectory_data["visual_demo_paths"]

    # Priority 2: Find another trajectory with the same number of total_steps
    for timestamp, trajectory_data in search_pool.items():
        if timestamp == current_timestamp_id:
            continue
        if trajectory_data["total_steps"] == current_trajectory["total_steps"]:
            print(f"  Found cross-ref for {current_timestamp_id} (same total_steps): {timestamp}")
            return trajectory_data["visual_demo_paths"]

    # Fallback: If no suitable cross-reference is found, use its own demo
    print(f"  Warning: No suitable cross-ref found for {current_timestamp_id}. Falling back to self-reference.")
    return current_trajectory["visual_demo_paths"]

def prepare_temp_dir(image_paths, temp_dir_root):
    """Creates a temporary directory and copies image files into it."""
    temp_dir = tempfile.mkdtemp(dir=temp_dir_root)
    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"Warning: Image file not found, skipping: {img_path}")
            continue
        shutil.copy(img_path, temp_dir)
    return temp_dir

def main():
    parser = argparse.ArgumentParser(description="Main pipeline to evaluate VLAC based on a pre-processed dataset.")
    parser.add_argument('--processed_data_dir', type=str, required=True, help="Path to the directory with split .jsonl files.")
    parser.add_argument('--image_root_dir', type=str, required=True, help="Root directory where all trajectory image folders are stored.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the VLAC model directory.")
    parser.add_argument('--output_dir', type=str, default='./results_pipeline', help="Directory to save evaluation results.")
    parser.add_argument('--cross_trajectory_ref', action='store_true', help="Enable cross-trajectory reference finding.")
    parser.add_argument('--gpu_ids', type=str, default="0", help="Comma-separated list of GPU IDs to use (e.g., '0,1,4').")

    args, passthrough_args = parser.parse_known_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    run_temp_root = tempfile.mkdtemp(prefix="vlac_eval_")
    print(f"Using temporary root for image data: {run_temp_root}")

    try:
        jsonl_files = glob.glob(os.path.join(args.processed_data_dir, '*.jsonl'))
        if not jsonl_files:
            print(f"Error: No .jsonl files found in {args.processed_data_dir}")
            return

        script_dir = Path(__file__).parent
        vlac_eval_script = script_dir / 'run_eval.py'

        total_files = len(jsonl_files)
        print(f"Found {total_files} task types (JSONL files) to process.")
        
        file_count = 0
        for jsonl_file in jsonl_files:
            file_count += 1
            task_type_sanitized = os.path.basename(jsonl_file).replace('.jsonl', '')
            print(f"\n--- Processing Task Type ({file_count}/{total_files}): {task_type_sanitized} ---")
            
            trajectories_in_task = load_trajectories_from_file(jsonl_file, args.image_root_dir)
            
            for timestamp_id, trajectory_data in trajectories_in_task.items():
                print(f"  -> Processing Trajectory: {timestamp_id}")

                stages = sorted(trajectory_data["stages"], key=lambda x: x["progress"])
                main_trajectory_paths = [s["image_path"] for s in stages]
                
                if not main_trajectory_paths:
                    print("  Warning: No stages found for this trajectory. Skipping.")
                    continue

                if args.cross_trajectory_ref:
                    ref_trajectory_paths = find_cross_reference(timestamp_id, trajectories_in_task)
                else:
                    ref_trajectory_paths = find_self_reference(trajectory_data)

                if not ref_trajectory_paths:
                    print("  Warning: No reference trajectory found. Skipping.")
                    continue

                main_dir = prepare_temp_dir(main_trajectory_paths, run_temp_root)
                ref_dir = prepare_temp_dir(ref_trajectory_paths, run_temp_root)
                
                # Original task_type had slashes, which we need for the output filename.
                # We can't perfectly reconstruct it, so we use the sanitized one.
                # This is a change in output naming convention.
                output_filename = f"{task_type_sanitized}_{timestamp_id}.json"
                
                num_gpus = len(args.gpu_ids.split(','))
                command = [
                    'python', str(vlac_eval_script),
                    '--model_path', args.model_path,
                    '--data_dir', main_dir,
                    '--ref_dir', ref_dir,
                    '--task', trajectory_data['task_goal'],
                    '--output_dir', args.output_dir,
                    '--output_name', output_filename,
                    '--gpu_ids', args.gpu_ids,
                    '--num_gpus', str(num_gpus)
                ] + passthrough_args

                print(f"  Executing VLAC evaluation for {timestamp_id}...")

                try:
                    process = subprocess.run(command, check=True, capture_output=True, text=True)
                    print(f"    Successfully evaluated. Results saved in {args.output_dir}/{output_filename}")
                except subprocess.CalledProcessError as e:
                    print(f"    ERROR: Evaluation failed for {timestamp_id}.")
                    print(f"    Return Code: {e.returncode}")
                    print(f"    Stdout: {e.stdout}")
                    print(f"    Stderr: {e.stderr}")

    finally:
        print(f"\nCleaning up temporary directory: {run_temp_root}")
        shutil.rmtree(run_temp_root)

if __name__ == "__main__":
    main()