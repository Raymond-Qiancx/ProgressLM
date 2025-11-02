# Text Object Replacement Pipeline

This pipeline creates negative cases for robotic manipulation tasks by replacing objects in task instructions while maintaining structural coherence.

## Overview

The text object replacement pipeline analyzes task goals and step-by-step instructions, identifies the main object being manipulated, and replaces it with a confusing or similar object to create failure cases.

## Files

### Core Components

1. **text_object_replacement_prompt.py** - Prompt construction
   - System prompt for object replacement
   - Builds prompts from dataset items
   - Supports optional image input for visual context

2. **text_object_replacement_dataset.py** - Dataset loading
   - Loads JSONL datasets
   - Validates required fields
   - Preserves all original metadata
   - Optional image path construction

3. **run_text_object_replacement_single.py** - Main inference script
   - Single-process batch inference
   - Optimized for 72B models with model parallelism
   - XML response parsing
   - Comprehensive error handling

### Execution Scripts

4. **eval/scripts/eval_text_object_replacement_72b.sh** - Shell script
   - Configuration management
   - Path validation
   - Execution with logging
   - Summary generation

## Input Format

Expected JSONL format with all fields preserved in output:

```json
{
  "id": "h5_agilex_3rgb/10_packplate_2/2024_09_28-17_07_01-172863393748093664.00",
  "task_goal": "with both arms placing two plates into a rack",
  "text_demo": [
    "[left] move towards the green plate and [right] grab the green plate",
    "[left] grab the green plate",
    "[left] move the green plate towards the rack and [right] move away from the green plate",
    "[left] put the green plate on the rack",
    "[left]move away from the green plate and [right] move towards the beige plate",
    "[right] grab the beige plate",
    "[left] grab the beige plate",
    "[left] move the beige plate towards the rack and [right] move away from the beige plate",
    "[left] put the beige plate on the rack",
    "[left] move away from the beige plate"
  ],
  "total_steps": "10",
  "stage_to_estimate": "camera_front_0062.jpg",
  "closest_idx": 1,
  "progress_score": "10%",
  "data_source": "h5_agilex_3rgb"
}
```

## Replacement Strategy

The model follows a priority-based replacement strategy:

1. **Priority 1** (Preferred): Replace with another object visible in the image
   - Example: If both "green plate" and "beige plate" are visible, replace "green plate" with "beige plate"
   - Creates maximum confusion by using contextually available alternatives

2. **Priority 2** (Fallback): Replace with a physically reasonable but different object
   - Example: Replace "apple" with "orange" (reasonable), not "car" or "water bottle"
   - Maintains logical coherence while creating a failure case

## Output Format

Each result contains:

```json
{
  "original_object": "green plate",
  "replacement_object": "beige plate",
  "replacement_strategy": "Priority 1",
  "reasoning": "This creates confusion since both plates are present...",
  "edited_goal": "with both arms placing two plates into a rack (beige plate focused)",
  "edited_demo": [
    "[left] move towards the beige plate and [right] grab the beige plate",
    "[left] grab the beige plate",
    ...
  ],
  "raw_response": "Full model response...",
  "meta_data": {
    "id": "h5_agilex_3rgb/10_packplate_2/...",
    "task_goal": "with both arms placing two plates into a rack",
    "text_demo": [...],
    "total_steps": "10",
    "stage_to_estimate": "camera_front_0062.jpg",
    "closest_idx": 1,
    "progress_score": "10%",
    "data_source": "h5_agilex_3rgb",
    "status": "success"
  }
}
```

**Important**: All original fields from the input JSONL are preserved in `meta_data`.

## Usage

### Quick Start

```bash
cd /home/vcj9002/jianshu/chengxuan/ProgressLM/eval/scripts

# Edit configuration in the script:
# - MODEL_PATH
# - DATASET_PATH
# - IMAGE_ROOT (optional, for image-based analysis)
# - OUTPUT_DIR

# Run inference
bash eval_text_object_replacement_72b.sh
```

### Python Command Line

```bash
cd /home/vcj9002/jianshu/chengxuan/ProgressLM/eval/qwen25vl

export CUDA_VISIBLE_DEVICES=0,1,2,3

python run_text_object_replacement_single.py \
    --model-path /path/to/Qwen2.5-VL-72B-Instruct \
    --dataset-path /path/to/dataset.jsonl \
    --output-file /path/to/output.jsonl \
    --image-root /path/to/images \
    --batch-size 48 \
    --num-inferences 1 \
    --temperature 0.7 \
    --limit 10  # Optional: test with small subset
```

### Key Parameters

- `--model-path`: Path to Qwen2.5-VL model
- `--dataset-path`: Input JSONL file
- `--output-file`: Output JSONL file
- `--image-root`: (Optional) Root directory for images
- `--batch-size`: Batch size (default: 1 for 72B)
- `--num-inferences`: Number of inferences per sample (default: 1)
- `--temperature`: Sampling temperature (default: 0.7)
- `--limit`: Limit samples for testing (default: -1 for all)

## Testing

Test with a small subset first:

```bash
python run_text_object_replacement_single.py \
    --model-path /path/to/model \
    --dataset-path /path/to/dataset.jsonl \
    --output-file /path/to/test_output.jsonl \
    --limit 10 \
    --verbose
```

## Model Requirements

- **Recommended**: Qwen2.5-VL-72B-Instruct or Qwen2.5-VL-32B-Instruct
- **GPUs**: 4x GPUs for 72B model (model parallelism)
- **Memory**: ~80GB VRAM for 72B model across 4 GPUs

## Output Files

1. **{output_file}.jsonl** - Main results (one JSON per line)
2. **{output_file}_summary.json** - Statistics and summary
3. **{output_file}.log** - Execution log (when using shell script)

## Editing Guidelines

The model is instructed to:

1. Keep original sentence format and structure
2. Preserve ALL markers (`[right]`, `[left]`, `[towards]`, etc.)
3. Only change object names
4. Maintain action verbs and spatial descriptions
5. Ensure replacement creates a valid negative case

## Error Handling

- Image validation errors: Marked as failed, recorded in response
- Parsing errors: If XML tags are missing or malformed
- Batch errors: Entire batch marked as failed
- All errors tracked in statistics and summary

## Performance

- **Single-process mode**: Optimized for 72B models
- **Model parallelism**: Automatic distribution across GPUs
- **Batch processing**: Configurable batch size
- **Periodic saving**: Results saved every 10 batches
- **Progress tracking**: Real-time progress bar with error rate

## Comparison with Adversarial Editing

| Feature | Adversarial Editing | Text Object Replacement |
|---------|---------------------|-------------------------|
| Input | Image + Text | Text (+ optional Image) |
| Output | Image editing prompt | Modified text instructions |
| Focus | Visual manipulation | Textual manipulation |
| Strategy | Color/Object/Occlusion | Object replacement |
| Use Case | Image-based failures | Instruction-based failures |

## Examples

### Input
```
Task Goal: "with both arms placing two plates into a rack"
Text Demo: ["[left] move towards the green plate and [right] grab the green plate", ...]
```

### Output
```
Original Object: "green plate"
Replacement Object: "beige plate"
Edited Goal: "with both arms placing two plates into a rack (beige plate focused)"
Edited Demo: ["[left] move towards the beige plate and [right] grab the beige plate", ...]
```

## Troubleshooting

1. **Import errors**: Ensure all dependencies are installed
2. **CUDA errors**: Check GPU availability and CUDA_VISIBLE_DEVICES
3. **Memory errors**: Reduce batch size or use smaller model
4. **Parsing errors**: Check model output format, adjust temperature
5. **Path errors**: Verify dataset path and image root exist

## Citation

If you use this pipeline, please cite the original ProgressLM work and acknowledge the Qwen2.5-VL model.
