# SFT Data Transfer for LLaMA-Factory

This directory contains tools to convert Text Demo and Visual Demo datasets from their original format + CoT responses into LLaMA-Factory's ShareGPT image-text interleaved format.

## 📋 Overview

The conversion process merges two types of files:
1. **Original data** (`*_sft.jsonl`): Contains task descriptions, demo content, and metadata
2. **CoT responses** (`*_cot.jsonl`): Contains model-generated reasoning and answers

**Output**: ShareGPT format JSON files suitable for Qwen2.5-VL SFT training in LLaMA-Factory.

## 📂 File Structure

```
sft_data_transfer/
├── README.md                    # This file
├── utils.py                     # Shared utility functions
├── convert_text_demo.py         # Text Demo converter
├── convert_visual_demo.py       # Visual Demo converter
├── validate_output.py           # Output validation tool
├── run_convert_text.sh          # Batch Text Demo conversion
└── run_convert_visual.sh        # Batch Visual Demo conversion
```

## 🔄 Data Format Conversion

### Text Demo

**Input (Original Data):**
```json
{
  "id": "h5_agilex_3rgb/10_packplate_2/2024_09_28-16_39_32-172863588507336032.00",
  "task_goal": "place two plates on the plate rack",
  "text_demo": [
    "[left] approaches the green plate while [right] grabs the green plate",
    "[left] grab the green plate while [right] away from the green plate",
    ...
  ],
  "total_steps": 10,
  "stage_to_estimate": "camera_front_0061.jpg",
  "closest_idx": 1,
  "progress_score": "10%"
}
```

**Input (CoT Response):**
```json
{
  "ref": "1",
  "score": "10%",
  "response": "<ref_think>...</ref_think>\n<ref>1</ref>\n<score_think>...</score_think>\n<score>10%</score>",
  "meta_data": {"id": "...", "status": "success"}
}
```

**Output (LLaMA-Factory Format):**
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Our goal is place two plates on the plate rack.\n\nHere is the demonstration:\nStep 1. [left] approaches the green plate...\nThe Progress for now is 10%.\n\n...\n\nHere is the current state that you need to estimate:\n<image>\n\nYour task: ..."
    },
    {
      "role": "assistant",
      "content": "<ref_think>...</ref_think>\n<ref>1</ref>\n<score_think>...</score_think>\n<score>10%</score>"
    }
  ],
  "images": ["h5_agilex_3rgb/10_packplate_2/.../camera_front_0061.jpg"]
}
```

### Visual Demo

**Input (Original Data):**
```json
{
  "id": "h5_agilex_3rgb/15_steamegg/2024_09_12-15_33_10-172955278034739456.00",
  "task_goal": "placing an egg into a pot",
  "visual_demo": [
    "camera_front_0000.jpg",
    "camera_front_0121.jpg",
    ...
  ],
  "total_steps": "8",
  "stage_to_estimate": ["camera_front_0143.jpg"],
  "progress_score": "16%"
}
```

**Output (LLaMA-Factory Format):**
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Our goal is placing an egg into a pot.\n\nHere is the demonstration:\n<image> 0% <image> 12% <image> 25% ... <image> 100%\n\nHere is the current state that you need to estimate:\n<image>\n\nYour task: ..."
    },
    {
      "role": "assistant",
      "content": "<ref_think>...</ref_think>\n<ref>3</ref>\n<score_think>...</score_think>\n<score>16%</score>"
    }
  ],
  "images": [
    "h5_agilex_3rgb/15_steamegg/.../camera_front_0000.jpg",
    "h5_agilex_3rgb/15_steamegg/.../camera_front_0121.jpg",
    ...
    "h5_agilex_3rgb/15_steamegg/.../camera_front_0143.jpg"
  ]
}
```

**Key Feature:** Image paths use format `{id}/{filename}` (no `images/` prefix).

## 🚀 Quick Start

### 1. Update Configuration

Edit the shell scripts to set your paths:

**`run_convert_text.sh`:**
```bash
# Original data directory
ORIGINAL_DIR="/Users/cxqian/Codes/ProgressLM/data/train/text_demo"

# CoT responses directory (UPDATE THIS!)
COT_DIR="/path/to/your/cot/responses/text_demo"

# Output directory
OUTPUT_DIR="/Users/cxqian/Codes/ProgressLM/LLaMA-Factory/data"
```

**`run_convert_visual.sh`:**
```bash
# Original data directory
ORIGINAL_DIR="/Users/cxqian/Codes/ProgressLM/data/train/visual_demo"

# CoT responses directory (UPDATE THIS!)
COT_DIR="/path/to/your/cot/responses/visual_demo"

# Output directory
OUTPUT_DIR="/Users/cxqian/Codes/ProgressLM/LLaMA-Factory/data"
```

### 2. Run Batch Conversion

**Convert all Text Demo datasets:**
```bash
cd /Users/cxqian/Codes/ProgressLM/cold_start/sft_data_transfer
bash run_convert_text.sh
```

**Convert all Visual Demo datasets:**
```bash
bash run_convert_visual.sh
```

### 3. Check Results

Output files will be saved to `OUTPUT_DIR` with names like:
- `text_h5_agilex_3rgb_llamafactory.json`
- `visual_h5_agilex_3rgb_llamafactory.json`

## 🔧 Manual Conversion

For individual datasets:

### Text Demo

```bash
python convert_text_demo.py \
    --original-data /path/to/text_h5_agilex_3rgb_sft.jsonl \
    --cot-responses /path/to/text_h5_agilex_3rgb_cot.jsonl \
    --output-file /path/to/output.json \
    --filter-success \
    --verbose
```

### Visual Demo

```bash
python convert_visual_demo.py \
    --original-data /path/to/visual_h5_agilex_3rgb_sft.jsonl \
    --cot-responses /path/to/visual_h5_agilex_3rgb_cot.jsonl \
    --output-file /path/to/output.json \
    --filter-success \
    --verbose
```

### Options

- `--filter-success`: Only include samples with `status='success'` in CoT responses
- `--verbose`: Print detailed conversion logs
- `--original-data`: Path to original JSONL file
- `--cot-responses`: Path to CoT responses JSONL file
- `--output-file`: Path to output JSON file

## ✅ Validation

Validate converted files:

```bash
python validate_output.py \
    --input-file /path/to/output.json \
    --show-samples 5 \
    --structure \
    --verbose
```

**Checks performed:**
- ✅ Required fields present (`messages`, `images`)
- ✅ Message structure (user + assistant)
- ✅ Image tag count matches images array length
- ✅ XML tags present in assistant response
- ✅ Non-empty content

## 📊 Understanding the Output

### Statistics

After conversion, you'll see:

```
=== Text Demo Conversion Summary ===
Dataset: text_h5_agilex_3rgb_sft
Total original samples: 1000
Total CoT responses loaded: 950
Matched samples: 945
  - Success status: 940
  - Failed status: 5
Unmatched samples: 55
Final output samples: 940
Image tag validation: 940/940 passed
Output file: /path/to/output.json
```

### Matching Logic

Samples are matched using:
1. **Primary:** `id` + `progress_score`
2. **Fallback:** `id` + `closest_idx`
3. **Last resort:** `id` only (if unique)

## 🎯 Using Converted Data in LLaMA-Factory

### 1. Register Datasets

Edit `LLaMA-Factory/data/dataset_info.json`:

```json
{
  "text_h5_agilex_3rgb_sft": {
    "file_name": "text_h5_agilex_3rgb_llamafactory.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "images": "images"
    }
  },
  "visual_h5_agilex_3rgb_sft": {
    "file_name": "visual_h5_agilex_3rgb_llamafactory.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "images": "images"
    }
  }
}
```

### 2. Update Training Config

Edit `LLaMA-Factory/our_scripts/qwen2_5vl_lora_sft_custom.yaml`:

```yaml
# Use single dataset
dataset: text_h5_agilex_3rgb_sft

# Or mix multiple datasets
dataset: text_h5_agilex_3rgb_sft,visual_h5_agilex_3rgb_sft,mllm_demo

template: qwen2_vl
```

### 3. Run Training

```bash
cd /Users/cxqian/Codes/ProgressLM/LLaMA-Factory
bash our_scripts/train_qwen2_5vl_lora_sft.sh
```

## 🔍 Key Features

### 1. Image Path Format

All image paths use format: `{id}/{filename}`

**Example:**
```
h5_agilex_3rgb/10_packplate_2/2024_09_28-16_39_32-172863588507336032.00/camera_front_0061.jpg
```

No `images/` prefix is added.

### 2. Image-Text Interleaving

**Text Demo:**
- User message: 1× `<image>` tag
- Images array: 1 image

**Visual Demo:**
- User message: N+1× `<image>` tags (N demo images + 1 current state)
- Images array: N+1 images in same order

### 3. Clean SFT Format

**Removed from original prompts:**
- ❌ `TEXT_DEMO_SYSTEM_PROMPT_TRAIN`
- ❌ `VISUAL_DEMO_SYSTEM_PROMPT_TRAIN`
- ❌ Ground-truth hints (`"Critical Rule"`, known answers)

**Preserved:**
- ✅ Task goal
- ✅ Demo content (text steps or images)
- ✅ Current state image
- ✅ Task instructions
- ✅ XML-formatted responses

## 📝 Example Workflow

```bash
# 1. Update CoT directory paths in scripts
vim run_convert_text.sh    # Set COT_DIR
vim run_convert_visual.sh  # Set COT_DIR

# 2. Run conversions
bash run_convert_text.sh
bash run_convert_visual.sh

# 3. Validate outputs
python validate_output.py \
    --input-file ../LLaMA-Factory/data/text_h5_agilex_3rgb_llamafactory.json \
    --structure

# 4. Register in LLaMA-Factory
vim ../LLaMA-Factory/data/dataset_info.json

# 5. Update training config
vim ../LLaMA-Factory/our_scripts/qwen2_5vl_lora_sft_custom.yaml

# 6. Start training
cd ../LLaMA-Factory
bash our_scripts/train_qwen2_5vl_lora_sft.sh
```

## ⚠️ Common Issues

### Issue: "CoT directory not found"

**Solution:** Update `COT_DIR` in the shell scripts to point to your CoT responses directory.

### Issue: "No CoT match found"

**Causes:**
- `id` mismatch between original and CoT files
- Missing `status='success'` in CoT (when using `--filter-success`)
- Different `progress_score` or `closest_idx` values

**Solution:** Check that IDs match exactly between files.

### Issue: "Image tag count mismatch"

**Cause:** Bug in conversion logic.

**Solution:** File a bug report with sample data.

### Issue: "Validation failed"

**Solution:** Run validation with `--verbose` to see detailed errors:
```bash
python validate_output.py --input-file output.json --verbose
```

## 🛠️ Development

### Adding New Datasets

Add to `DATASETS` array in shell scripts:

```bash
DATASETS=(
    "new_dataset|new_dataset_sft.jsonl|new_dataset_cot.jsonl"
)
```

### Modifying Prompts

Edit instruction strings in converter files:
- `convert_text_demo.py`: `TEXT_DEMO_INSTRUCTION`
- `convert_visual_demo.py`: `VISUAL_DEMO_INSTRUCTION`

### Custom Validation

Add checks in `validate_output.py`:

```python
def custom_validation(sample):
    # Your validation logic
    pass
```

## 📚 References

- [LLaMA-Factory Documentation](https://github.com/hiyouga/LLaMA-Factory)
- [Qwen2-VL Model](https://github.com/QwenLM/Qwen2-VL)
- [ShareGPT Format Spec](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md)

## 🐛 Troubleshooting

For issues or questions:
1. Check validation output for specific errors
2. Verify input file formats match expected structure
3. Ensure CoT responses have `status='success'`
4. Check that image paths are constructed correctly

## 📄 License

Same as parent ProgressLM project.
