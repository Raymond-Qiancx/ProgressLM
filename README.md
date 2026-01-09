# ProgressLM

## Project Structure

```
ProgressLM/
├── EasyR1/                 # RL training framework (GRPO)
├── LLaMA-Factory/          # SFT training framework
├── cold_start/             # Data generation and cold start
├── dataset/                # Benchmark datasets
│   ├── prog-bench/         # Programmatic task benchmarks
│   └── human-bench/        # Human activity benchmarks
├── eval/                   # Evaluation code
│   ├── qwen25vl/           # Qwen2.5-VL evaluation
│   ├── qwen3vl/            # Qwen3-VL evaluation
│   ├── internvl/           # InternVL evaluation
│   └── openai/             # OpenAI API evaluation
└── README.md
```

## Installation

### SFT Environment

We use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for supervised fine-tuning. It supports LoRA, QLoRA, and full fine-tuning with various model architectures including Qwen2.5-VL.

```bash
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

For more details, please refer to the [LLaMA-Factory documentation](https://github.com/hiyouga/LLaMA-Factory).

### RL Environment

We use [EasyR1](https://github.com/hiyouga/EasyR1) for reinforcement learning with GRPO (Group Relative Policy Optimization). It provides distributed training support with FSDP and efficient rollout generation.

```bash
cd EasyR1
pip install -e .
```

For more details, please refer to the [EasyR1 documentation](https://github.com/hiyouga/EasyR1).

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

### Configuration Files

| Config File | Description |
|-------------|-------------|
| `LLaMA-Factory/our_scripts/qwen2_5vl_lora_sft_small.yaml` | Qwen2.5-VL-3B LoRA SFT config |
| `LLaMA-Factory/our_scripts/qwen2_5vl_lora_sft_7b.yaml` | Qwen2.5-VL-7B LoRA SFT config |
| `LLaMA-Factory/our_scripts/qwen3vl_4b_lora_sft.yaml` | Qwen3-VL-4B LoRA SFT config |

### Running Commands

```bash
# Single GPU
bash LLaMA-Factory/our_scripts/train_qwen2_5vl_lora_sft.sh

# Multi GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 bash LLaMA-Factory/our_scripts/train_qwen2_5vl_lora_sft.sh

# 7B Model
bash LLaMA-Factory/our_scripts/train_qwen2_5vl_lora_sft_7b.sh
```

### Key Configuration Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `model_name_or_path` | Base model path | `Qwen/Qwen2.5-VL-3B-Instruct` |
| `dataset` | Training dataset name | `progresslm_think` |
| `output_dir` | Output directory for checkpoints | `/path/to/output` |
| `lora_rank` | LoRA rank | `8` |
| `lora_alpha` | LoRA alpha | `16` |
| `lora_target` | LoRA target modules | `all` |
| `learning_rate` | Learning rate | `1.0e-4` |
| `num_train_epochs` | Number of training epochs | `3.0` |
| `per_device_train_batch_size` | Batch size per device | `2` |
| `gradient_accumulation_steps` | Gradient accumulation steps | `8` |

## RL Training (GRPO)

### Configuration Files

| Config File | Description |
|-------------|-------------|
| `EasyR1/progresslm/configs/visual_demo_grpo.yaml` | Qwen2.5-VL-3B GRPO config |
| `EasyR1/progresslm/configs/visual_demo_grpo_7b.yaml` | Qwen2.5-VL-7B GRPO config |
| `EasyR1/progresslm/configs/multinodes.yaml` | Multi-node training config |

### Running Commands

```bash
# Single Node (3B Model)
bash EasyR1/progresslm/run_grpo_3b.sh

# Multi Node (3B Model)
bash EasyR1/progresslm/run_grpo_3b_multinode.sh
```

### Key Configuration Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `worker.actor.model.model_path` | SFT model path | `/path/to/sft_model` |
| `trainer.save_checkpoint_path` | Checkpoint save path | `/path/to/rl_ckpt` |
| `algorithm.adv_estimator` | Advantage estimator | `grpo` |
| `algorithm.kl_coef` | KL penalty coefficient | `1.0e-2` |
| `algorithm.kl_target` | Target KL divergence | `0.1` |
| `worker.rollout.n` | Number of rollouts | `4` |
| `worker.rollout.temperature` | Sampling temperature | `0.6` |
| `worker.actor.optim.lr` | Learning rate | `1.0e-6` |
| `trainer.total_epochs` | Total training epochs | `2` |

## Evaluation

### Supported Models

| Model | Directory | Description |
|-------|-----------|-------------|
| Qwen2.5-VL | `eval/qwen25vl/` | Qwen2.5-VL series (3B, 7B, 72B) |
| Qwen3-VL | `eval/qwen3vl/` | Qwen3-VL series |
| InternVL | `eval/internvl/` | InternVL series |
| OpenAI GPT | `eval/openai/` | GPT-4V, GPT-4o via API |

### Evaluation Scripts

| Script | Description |
|--------|-------------|
| `run_text_demo.py` | Text demonstration evaluation |
| `run_visual_demo.py` | Visual demonstration evaluation |
| `run_text_demo_nothink.py` | Text evaluation without thinking |
| `run_visual_demo_nothink.py` | Visual evaluation without thinking |

### Running Examples

```bash
# Qwen2.5-VL Evaluation
cd eval/qwen25vl/codes
python run_text_demo.py --model_path <model_path> --data_path <data_path>
python run_visual_demo.py --model_path <model_path> --data_path <data_path>

# InternVL Evaluation
cd eval/internvl/codes
python run_text_demo.py --model_path <model_path> --data_path <data_path>
python run_visual_demo.py --model_path <model_path> --data_path <data_path>

# OpenAI API Evaluation
cd eval/openai/codes
python run_text_demo.py --data_path <data_path>
python run_visual_demo.py --data_path <data_path>
```

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **VOC** | Trajectory Order Consistency - measures the Spearman correlation of predicted progress order |
| **Score Error** | Normalized error of predicted progress score |
| **Ref Error** | Normalized error of predicted reference step index |

## Merge LoRA Weights

After SFT training, merge LoRA weights into the base model:

```bash
# Merge Qwen2.5-VL-3B LoRA
llamafactory-cli export LLaMA-Factory/our_scripts/merge_qwen2_5vl_lora.yaml

# Merge Qwen2.5-VL-7B LoRA
llamafactory-cli export LLaMA-Factory/our_scripts/merge_qwen25vl_7b_lora.yaml
```
