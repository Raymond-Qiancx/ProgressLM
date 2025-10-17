import argparse
import json
import os
import glob
import subprocess
import tempfile
import shutil
from pathlib import Path
from tqdm import tqdm

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
                # This warning can be noisy, print only if not in quiet mode from a higher level
                # For now, we keep it, as this function is not aware of a quiet flag.
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

        # First, count total number of trajectories for tqdm
        print("Pre-calculating total number of trajectories...")
        total_trajectories = 0
        for jsonl_file in jsonl_files:
            # A bit inefficient to load files twice, but necessary for a clean progress bar
            trajectories = load_trajectories_from_file(jsonl_file, args.image_root_dir)
            total_trajectories += len(trajectories)
        print(f"Found {total_trajectories} total trajectories to evaluate.")

        # Initialize trackers for metrics
        error_count = 0
        voc_scores = []
        neg_rates = []

        with tqdm(total=total_trajectories, desc="Overall Progress", unit="traj") as pbar:
            for jsonl_file in jsonl_files:
                trajectories_in_task = load_trajectories_from_file(jsonl_file, args.image_root_dir)
                
                for timestamp_id, trajectory_data in trajectories_in_task.items():
                    # Update postfix at the beginning of the loop for immediate feedback
                    avg_voc = sum(voc_scores) / len(voc_scores) if voc_scores else 0
                    avg_neg_rate = sum(neg_rates) / len(neg_rates) if neg_rates else 0
                    pbar.set_postfix(avg_VOC=f'{avg_voc:.3f}', avg_NegRate=f'{avg_neg_rate:.3f}', errors=error_count, refresh=True)

                    stages = sorted(trajectory_data["stages"], key=lambda x: x["progress"])
                    main_trajectory_paths = [s["image_path"] for s in stages]
                    
                    if not main_trajectory_paths:
                        error_count += 1
                        pbar.update(1)
                        continue

                    if args.cross_trajectory_ref:
                        ref_trajectory_paths = find_cross_reference(timestamp_id, trajectories_in_task)
                    else:
                        ref_trajectory_paths = find_self_reference(trajectory_data)

                    if not ref_trajectory_paths:
                        error_count += 1
                        pbar.update(1)
                        continue

                    main_dir = prepare_temp_dir(main_trajectory_paths, run_temp_root)
                    ref_dir = prepare_temp_dir(ref_trajectory_paths, run_temp_root)
                    
                    task_type_sanitized = os.path.basename(jsonl_file).replace('.jsonl', '')
                    output_filename = f"{task_type_sanitized}_{timestamp_id}.json"
                    output_path = os.path.join(args.output_dir, output_filename)

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
                        '--num_gpus', str(num_gpus),
                        '--quiet' # Suppress output from the child script
                    ] + passthrough_args

                    try:
                        # We still pass --quiet, but capture output here to debug if something goes wrong.
                        process = subprocess.run(command, check=True, capture_output=True, text=True)
                        
                        # Read results from the generated JSON to update metrics
                        with open(output_path, 'r') as f:
                            results = json.load(f)
                        
                        voc = results['metrics']['voc']
                        neg_rate = results['metrics']['negative_rate']
                        voc_scores.append(voc)
                        neg_rates.append(neg_rate)

                    except subprocess.CalledProcessError as e:
                        error_count += 1
                        # Print the error to the main console for debugging
                        pbar.write(f"\n--- ERROR processing {timestamp_id} ---")
                        pbar.write(f"  Return Code: {e.returncode}")
                        if e.stdout:
                            pbar.write(f"  Stdout: {e.stdout.strip()}")
                        if e.stderr:
                            pbar.write(f"  Stderr: {e.stderr.strip()}")
                        pbar.write("--- END ERROR ---")
                    except (FileNotFoundError, json.JSONDecodeError) as e:
                        error_count += 1
                        pbar.write(f"\n--- ERROR processing {timestamp_id} (Post-processing failed) ---")
                        pbar.write(f"  Error: {e}")
                        pbar.write("--- END ERROR ---")
                    
                    pbar.update(1)

    finally:
        print(f"\nCleaning up temporary directory: {run_temp_root}")
        shutil.rmtree(run_temp_root)
    
    # Final summary
    print("\n" + "="*80)
    print("Evaluation Summary")
    print("="*80)
    final_avg_voc = sum(voc_scores) / len(voc_scores) if voc_scores else 0
    final_avg_neg_rate = sum(neg_rates) / len(neg_rates) if neg_rates else 0
    print(f"Total Trajectories Evaluated: {total_trajectories}")
    print(f"Successful Evaluations: {len(voc_scores)}")
    print(f"Failed Evaluations (Errors): {error_count}")
    print(f"\nFinal Average VOC: {final_avg_voc:.4f}")
    print(f"Final Average Negative Rate: {final_avg_neg_rate:.4f}")
    print("="*80)

if __name__ == "__main__":
    main()
