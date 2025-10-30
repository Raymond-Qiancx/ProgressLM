#!/bin/bash

# Example Configuration for Qwen3VL Visual Demo Progress Estimation
# Copy this file and modify according to your setup

# ===========================
# Example 1: Single GPU (Quick Test)
# ===========================
example_single_gpu() {
    export CUDA_VISIBLE_DEVICES=0

    python run.py \
        --model-path "Qwen/Qwen3-VL-2B-Instruct" \
        --dataset-path "/path/to/dataset.jsonl" \
        --output-file "./outputs/test_results.jsonl" \
        --image-root "/path/to/images" \
        --batch-size 8 \
        --num-inferences 1 \
        --limit 10 \
        --temperature 0.7 \
        --verbose
}

# ===========================
# Example 2: Multi-GPU (4 GPUs)
# ===========================
example_multi_gpu() {
    export CUDA_VISIBLE_DEVICES=0,1,2,3

    python run.py \
        --model-path "Qwen/Qwen3-VL-8B-Instruct" \
        --dataset-path "/path/to/dataset.jsonl" \
        --output-file "./outputs/results.jsonl" \
        --image-root "/path/to/images" \
        --batch-size 16 \
        --num-inferences 4 \
        --temperature 0.7 \
        --top-p 0.9 \
        --top-k 50 \
        --max-new-tokens 512
}

# ===========================
# Example 3: Large Model (32B) with Lower Resolution
# ===========================
example_large_model() {
    export CUDA_VISIBLE_DEVICES=0,1,2,3

    python run.py \
        --model-path "Qwen/Qwen3-VL-32B-Instruct" \
        --dataset-path "/path/to/dataset.jsonl" \
        --output-file "./outputs/32b_results.jsonl" \
        --image-root "/path/to/images" \
        --batch-size 4 \
        --num-inferences 4 \
        --temperature 0.7 \
        --min-pixels $((256*256)) \
        --max-pixels $((512*256))
}

# ===========================
# Example 4: High Resolution Inference
# ===========================
example_high_resolution() {
    export CUDA_VISIBLE_DEVICES=0,1

    python run.py \
        --model-path "Qwen/Qwen3-VL-8B-Instruct" \
        --dataset-path "/path/to/dataset.jsonl" \
        --output-file "./outputs/high_res_results.jsonl" \
        --image-root "/path/to/images" \
        --batch-size 4 \
        --num-inferences 4 \
        --temperature 0.7 \
        --min-pixels $((512*256)) \
        --max-pixels $((2560*256))
}

# ===========================
# Example 5: Greedy Decoding (Deterministic)
# ===========================
example_greedy_decoding() {
    export CUDA_VISIBLE_DEVICES=0,1,2,3

    python run.py \
        --model-path "Qwen/Qwen3-VL-8B-Instruct" \
        --dataset-path "/path/to/dataset.jsonl" \
        --output-file "./outputs/greedy_results.jsonl" \
        --image-root "/path/to/images" \
        --batch-size 16 \
        --num-inferences 1 \
        --temperature 0.01 \
        --top-p 0.001 \
        --top-k 1
}

# ===========================
# Example 6: Creative/Diverse Sampling
# ===========================
example_diverse_sampling() {
    export CUDA_VISIBLE_DEVICES=0,1,2,3

    python run.py \
        --model-path "Qwen/Qwen3-VL-8B-Instruct" \
        --dataset-path "/path/to/dataset.jsonl" \
        --output-file "./outputs/diverse_results.jsonl" \
        --image-root "/path/to/images" \
        --batch-size 16 \
        --num-inferences 10 \
        --temperature 1.0 \
        --top-p 0.95 \
        --top-k 100
}

# ===========================
# Run the example you want
# ===========================

# Uncomment the example you want to run:

# example_single_gpu
# example_multi_gpu
# example_large_model
# example_high_resolution
# example_greedy_decoding
# example_diverse_sampling

echo "Please uncomment one of the examples above to run."
echo "Available examples:"
echo "  - example_single_gpu: Quick test with 1 GPU"
echo "  - example_multi_gpu: Standard 4-GPU setup"
echo "  - example_large_model: 32B model with low resolution"
echo "  - example_high_resolution: High-quality inference"
echo "  - example_greedy_decoding: Deterministic output"
echo "  - example_diverse_sampling: Creative responses"
