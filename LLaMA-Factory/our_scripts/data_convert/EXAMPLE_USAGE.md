# SFT Data Conversion - Quick Example

This guide walks through a complete example of converting one dataset.

## 📝 Step-by-Step Example

### Step 1: Prepare Your Data

Assume you have:
- **Original data**: `/Users/cxqian/Codes/ProgressLM/data/train/text_demo/text_h5_agilex_3rgb_sft.jsonl`
- **CoT responses**: `/path/to/cot/text_h5_agilex_3rgb_cot.jsonl`

### Step 2: Run Conversion

```bash
cd /Users/cxqian/Codes/ProgressLM/cold_start/sft_data_transfer

python convert_text_demo.py \
    --original-data /Users/cxqian/Codes/ProgressLM/data/train/text_demo/text_h5_agilex_3rgb_sft.jsonl \
    --cot-responses /path/to/cot/text_h5_agilex_3rgb_cot.jsonl \
    --output-file /Users/cxqian/Codes/ProgressLM/LLaMA-Factory/data/text_h5_agilex_3rgb_sft.json \
    --filter-success \
    --verbose
```

**Expected Output:**
```
Loading original data from: .../text_h5_agilex_3rgb_sft.jsonl
Loaded 1000 original samples

Loading CoT responses from: .../text_h5_agilex_3rgb_cot.jsonl
Loaded 950 CoT responses covering 945 unique IDs

Processing 1000 original samples...

Saving 940 samples to: .../text_h5_agilex_3rgb_sft.json

==================================================
Conversion Summary
==================================================
Dataset: text_h5_agilex_3rgb_sft
Total original samples: 1000
Total CoT responses loaded: 950
Matched samples: 945
  - Success status: 940
  - Failed/Other status: 5
Unmatched samples: 55
Final output samples: 940
Image tag validation: 940/940 passed
Output file: .../text_h5_agilex_3rgb_sft.json
==================================================

✅ Text Demo conversion completed successfully!
```

### Step 3: Validate Output

```bash
python validate_output.py \
    --input-file /Users/cxqian/Codes/ProgressLM/LLaMA-Factory/data/text_h5_agilex_3rgb_sft.json \
    --show-samples 3 \
    --structure
```

**Expected Output:**
```
Validating file: .../text_h5_agilex_3rgb_sft.json
============================================================
Total samples: 940

✅ Sample 1:
  Images: 1, <image> tags: 1

✅ Sample 2:
  Images: 1, <image> tags: 1

✅ Sample 3:
  Images: 1, <image> tags: 1

============================================================
Validation Summary
============================================================
Total samples: 940
Valid samples: 940 (100.0%)
Invalid samples: 0 (0.0%)

✅ Validation PASSED: All samples are valid!

============================================================
Dataset Structure Analysis
============================================================
Image counts per sample:
  Min: 1
  Max: 1
  Average: 1.0

Image count distribution:
  1 images: 940 samples

Message content lengths (chars, sampled):
  User avg: 1250
  Assistant avg: 320
============================================================
```

### Step 4: Register in LLaMA-Factory

Edit `LLaMA-Factory/data/dataset_info.json`:

```json
{
  "text_h5_agilex_3rgb_sft": {
    "file_name": "text_h5_agilex_3rgb_sft.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "images": "images"
    }
  }
}
```

### Step 5: Configure Training

Edit `LLaMA-Factory/our_scripts/qwen2_5vl_lora_sft_custom.yaml`:

```yaml
### dataset
dataset: text_h5_agilex_3rgb_sft
template: qwen2_vl
```

### Step 6: Start Training

```bash
cd /Users/cxqian/Codes/ProgressLM/LLaMA-Factory
bash our_scripts/train_qwen2_5vl_lora_sft.sh
```

## 🎨 Visual Demo Example

For visual demo:

```bash
python convert_visual_demo.py \
    --original-data /Users/cxqian/Codes/ProgressLM/data/train/visual_demo/visual_h5_agilex_3rgb_sft.jsonl \
    --cot-responses /path/to/cot/visual_h5_agilex_3rgb_cot.jsonl \
    --output-file /Users/cxqian/Codes/ProgressLM/LLaMA-Factory/data/visual_h5_agilex_3rgb_sft.json \
    --filter-success \
    --verbose
```

## 🔄 Batch Conversion Example

For multiple datasets at once:

```bash
# 1. Edit run_convert_text.sh to set COT_DIR
vim run_convert_text.sh

# Change this line:
COT_DIR="/path/to/your/cot/responses/text_demo"

# 2. Run batch conversion
bash run_convert_text.sh
```

This will process all configured datasets automatically.

## 📊 Sample Output File

**File: `text_h5_agilex_3rgb_sft.json`**

```json
[
  {
    "messages": [
      {
        "role": "user",
        "content": "Our goal is place two plates on the plate rack.\n\nHere is the demonstration:\nStep 1. [left] approaches the green plate while [right] grabs the green plate\nThe Progress for now is 10%.\n\n...\n\nHere is the current state that you need to estimate:\n<image>\n\nYour task:\n1. Analyze the text_demo to understand..."
      },
      {
        "role": "assistant",
        "content": "<ref_think>The current state shows the left hand approaching the green plate...</ref_think>\n<ref>1</ref>\n<score_think>Based on the visual similarity...</score_think>\n<score>10%</score>"
      }
    ],
    "images": [
      "h5_agilex_3rgb/10_packplate_2/2024_09_28-16_39_32-172863588507336032.00/camera_front_0061.jpg"
    ]
  },
  ...
]
```

## ✅ Checklist

Before training:

- [ ] CoT responses directory path updated in scripts
- [ ] Conversion completed successfully (no errors)
- [ ] Validation passed (100% valid samples)
- [ ] Dataset registered in `dataset_info.json`
- [ ] Training config updated to use new dataset
- [ ] Image paths are accessible from training environment

## 💡 Tips

1. **Start Small**: Convert one dataset first to verify everything works
2. **Check Paths**: Ensure image paths in output are correct for your setup
3. **Filter Success**: Use `--filter-success` to only include high-quality samples
4. **Validate Always**: Always run validation after conversion
5. **Test Training**: Do a quick test run (1-2 epochs) before full training

## 🐛 If Something Goes Wrong

### Low Match Rate (< 80%)

Check:
- IDs match exactly between original and CoT files
- CoT responses have correct `meta_data.id` field
- Progress scores or closest_idx values align

### Validation Errors

Run with verbose to see details:
```bash
python validate_output.py --input-file output.json --verbose
```

### Training Fails to Load Dataset

Check:
- Dataset registered correctly in `dataset_info.json`
- Output file exists at specified path
- `formatting` is set to `"sharegpt"`
- Image paths are accessible from training script

## 📚 Next Steps

After successful conversion:

1. **Verify a few samples manually** - Open the JSON and check formatting
2. **Run small training test** - Use `max_samples: 100` to test quickly
3. **Monitor first epoch** - Check loss curves and sample outputs
4. **Scale up** - If all looks good, run full training

Happy training! 🚀
