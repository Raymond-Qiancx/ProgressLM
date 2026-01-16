# ProgressLM: Towards Progress Reasoning in Vision-Language Models

### Under Construction

## Installation



### SFT Environment

We use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for supervised fine-tuning. It supports LoRA, QLoRA, and full fine-tuning with various model architectures including Qwen2.5-VL.

```bash
conda create -n progsft python=3.11 -y
conda activate progsft
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```


### RL Environment

We use [EasyR1](https://github.com/hiyouga/EasyR1) for reinforcement learning with GRPO (Group Relative Policy Optimization). It provides distributed training support with FSDP and efficient rollout generation.

```bash
conda create -n progrl python=3.11 -y
conda activate progrl
cd EasyR1
pip install -e .
```


### Evaluation Environment

```bash
conda create -n progresslm python=3.11 -y
conda activate progresslm
pip install -r eval/requirement.txt
```

## Datasets

### prog-bench

Programmatic task benchmarks for progress reasoning evaluation.

| File | Description |
|------|-------------|
| `text-normal.jsonl` | Text demonstration normal samples |
| `text-unanswerable.jsonl` | Unanswerable text samples |
| `visual_same_view.jsonl` | Same-view visual demonstrations |
| `visual_cross_view.jsonl` | Cross-view visual demonstrations |
| `visual-unanswerable.jsonl` | Unanswerable visual samples |

### human-bench

Human activity benchmarks for progress reasoning evaluation.

| File | Description |
|------|-------------|
| `text_demo_human_activities.jsonl` | Text demonstrations of human activities |
| `visual_demo_human_activities.jsonl` | Visual demonstrations of human activities |

## SFT Training

### Data Preparation

Convert your CoT responses to LLaMA-Factory format:

```bash
cd LLaMA-Factory/our_scripts/data_convert

# Convert Text Demo data
python convert_text_demo.py \
    --original-data /path/to/text_demo.jsonl \
    --cot-responses /path/to/cot_responses.jsonl \
    --output-file /path/to/output.json \
    --filter-success

# Convert Visual Demo data
python convert_visual_demo.py \
    --original-data /path/to/visual_demo.jsonl \
    --cot-responses /path/to/cot_responses.jsonl \
    --output-file /path/to/output.json \
    --filter-success

# Batch convert and merge all datasets
bash run_convert_and_merge.sh
```

### Configuration Files

| Config File | Description |
|-------------|-------------|
| `qwen2_5vl_lora_sft_small.yaml` | Qwen2.5-VL-3B LoRA SFT config |
| `qwen2_5vl_lora_sft_7b.yaml` | Qwen2.5-VL-7B LoRA SFT config |
| `qwen3vl_4b_lora_sft.yaml` | Qwen3-VL-4B LoRA SFT config |

### Running SFT Training

```bash
cd LLaMA-Factory

# Qwen2.5-VL-3B (Single GPU)
bash our_scripts/train_qwen2_5vl_lora_sft.sh

# Qwen2.5-VL-3B (Multi GPU)
CUDA_VISIBLE_DEVICES=0,1,2,3 bash our_scripts/train_qwen2_5vl_lora_sft.sh

# Qwen2.5-VL-7B
bash our_scripts/train_qwen2_5vl_lora_sft_7b.sh

# Qwen3-VL-4B
bash our_scripts/train_qwen3vl_4b_lora_sft.sh
```

### Merge LoRA Weights

After SFT training, merge LoRA weights into the base model:

```bash
# Merge Qwen2.5-VL-3B LoRA
llamafactory-cli export LLaMA-Factory/our_scripts/merge_qwen2_5vl_lora.yaml

# Merge Qwen2.5-VL-7B LoRA
llamafactory-cli export LLaMA-Factory/our_scripts/merge_qwen25vl_7b_lora.yaml
```

### Key SFT Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `model_name_or_path` | Base model path | `Qwen/Qwen2.5-VL-3B-Instruct` |
| `dataset` | Training dataset name | `progresslm_think` |
| `output_dir` | Output directory | `/path/to/output` |
| `lora_rank` | LoRA rank | `8` |
| `lora_alpha` | LoRA alpha | `16` |
| `learning_rate` | Learning rate | `1.0e-4` |
| `num_train_epochs` | Training epochs | `3.0` |
| `per_device_train_batch_size` | Batch size per GPU | `2` |
| `gradient_accumulation_steps` | Gradient accumulation | `8` |

## RL Training (GRPO)

### Configuration Files

| Config File | Description |
|-------------|-------------|
| `configs/visual_demo_grpo.yaml` | Qwen2.5-VL-3B GRPO config |
| `configs/visual_demo_grpo_7b.yaml` | Qwen2.5-VL-7B GRPO config |
| `configs/multinodes.yaml` | Multi-node training config |

### Running RL Training

```bash
cd EasyR1

# Qwen2.5-VL-3B (Single Node)
bash progresslm/run_grpo_3b.sh

# Qwen2.5-VL-3B (Multi Node)
bash progresslm/run_grpo_3b_multinode.sh

# Qwen2.5-VL-7B
bash progresslm/run_grpo_7b.sh
```

### Key RL Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `worker.actor.model.model_path` | SFT model path | `/path/to/sft_model` |
| `trainer.save_checkpoint_path` | Checkpoint path | `/path/to/rl_ckpt` |
| `algorithm.kl_coef` | KL penalty coefficient | `1.0e-2` |
| `algorithm.kl_target` | Target KL divergence | `0.1` |
| `worker.rollout.n` | Number of rollouts | `4` |
| `worker.rollout.temperature` | Sampling temperature | `0.6` |
| `worker.actor.optim.lr` | Learning rate | `1.0e-6` |
| `trainer.total_epochs` | Training epochs | `2` |

## Evaluation

### Supported Models

| Model | Directory | Description |
|-------|-----------|-------------|
| Qwen2.5-VL | `eval/qwen25vl/` | Qwen2.5-VL series (3B, 7B, 72B) |
| Qwen3-VL | `eval/qwen3vl/` | Qwen3-VL series |
| InternVL | `eval/internvl/` | InternVL series |
| OpenAI GPT | `eval/openai/` | GPT-4V, GPT-4o via API |

### Benchmark Scripts

Evaluation scripts are organized in `eval/qwen25vl/scripts/benchmarks/`:

| Benchmark | Description | Scripts |
|-----------|-------------|---------|
| `normal_text/` | Text demonstration (normal) | `eval_text_normal_sft_3b.sh`, `eval_text_normal_rl_3b.sh`, ... |
| `normal_view/` | Visual demonstration (same view) | `visual_eval_one_view_3B_SFT.sh`, `visual_eval_one_view_3B_RL.sh`, ... |
| `multi_view/` | Visual demonstration (cross view) | Multi-view evaluation scripts |
| `nega_text/` | Text unanswerable samples | Negative text evaluation scripts |
| `edit_nega/` | Visual unanswerable samples | Negative visual evaluation scripts |
| `human/` | Human activity benchmarks | Human activity evaluation scripts |

### Running Evaluation

#### Text Demo Evaluation (prog-bench)

```bash
cd eval/qwen25vl/scripts/benchmarks/normal_text

# SFT Model (3B)
bash eval_text_normal_sft_3b.sh

# RL Model (3B)
bash eval_text_normal_rl_3b.sh

# SFT Model (7B)
bash eval_text_normal_sft_7b.sh

# Large Models (72B)
bash eval_text_normal_72b.sh
```

#### Visual Demo Evaluation (prog-bench)

```bash
cd eval/qwen25vl/scripts/benchmarks/normal_view

# SFT Model (3B)
bash visual_eval_one_view_3B_SFT.sh

# RL Model (3B)
bash visual_eval_one_view_3B_RL.sh

# SFT Model (7B)
bash visual_eval_one_view_7B_SFT.sh

# Large Models (72B)
bash visual_eval_one_view_72B.sh
```

#### Human Activity Evaluation (human-bench)

```bash
cd eval/qwen25vl/scripts/benchmarks/human

# Text Demo - Human Activities
bash text_eval_human_rl_3b.sh

# Visual Demo - Human Activities
bash visual_eval_human_3B_RL.sh
```

#### Nothink Mode Evaluation

For models without thinking process:

```bash
# Text Demo Nothink
cd eval/qwen25vl/scripts/benchmarks/normal_text
bash nothink_3b.sh
bash nothink_7b.sh
bash nothink_72b.sh

# Visual Demo Nothink
cd eval/qwen25vl/scripts/benchmarks/normal_view
bash visual_eval_one_view_nothink_3B.sh
bash visual_eval_one_view_nothink_7B.sh
bash visual_eval_one_view_nothink_72B.sh
```

### Manual Evaluation Command

```bash
cd eval/qwen25vl/codes

# Text Demo Evaluation
python run_text_demo.py \
    --model-path /path/to/model \
    --dataset-path /path/to/text_demo.jsonl \
    --output-file /path/to/results.jsonl \
    --image-root /path/to/images \
    --batch-size 100 \
    --temperature 0.6 \
    --max-new-tokens 40000

# Visual Demo Evaluation
python run_visual_demo.py \
    --model-path /path/to/model \
    --dataset-path /path/to/visual_demo.jsonl \
    --output-file /path/to/results.jsonl \
    --image-root /path/to/images \
    --batch-size 50 \
    --temperature 0.6 \
    --max-new-tokens 40000

# Nothink Mode
python run_text_demo_nothink.py \
    --model-path /path/to/model \
    --dataset-path /path/to/text_demo.jsonl \
    --output-file /path/to/results.jsonl

python run_visual_demo_nothink.py \
    --model-path /path/to/model \
    --dataset-path /path/to/visual_demo.jsonl \
    --output-file /path/to/results.jsonl
```

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **VOC** | Trajectory Order Consistency - Spearman correlation of predicted progress order |
| **Score Error** | Normalized error of predicted progress score |
| **Ref Error** | Normalized error of predicted reference step index |
| **N/A Recall** | Recall for unanswerable samples (predicting "n/a" correctly) |

### Other Model Evaluation

#### InternVL

```bash
cd eval/internvl/codes
python run_text_demo.py --model-path /path/to/internvl --dataset-path /path/to/data.jsonl
python run_visual_demo.py --model-path /path/to/internvl --dataset-path /path/to/data.jsonl
```

#### OpenAI API

```bash
cd eval/openai/codes
export OPENAI_API_KEY=your_api_key
python run_text_demo.py --dataset-path /path/to/data.jsonl
python run_visual_demo.py --dataset-path /path/to/data.jsonl
```
