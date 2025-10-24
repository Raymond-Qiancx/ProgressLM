# 72B Model Inference Scripts

这些脚本专门为运行 **Qwen2.5-VL-72B** 或 **Qwen2.5-VL-32B** 等大模型设计，使用**模型并行**方式在4张80G GPU上运行。

## 文件说明

### Python 脚本（位于 `../qwen25vl/`）
- `run_text_demo_single.py` - 文本演示的单进程推理脚本
- `run_visual_demo_single.py` - 视觉演示的单进程推理脚本

### Shell 启动脚本（位于当前目录）
- `eval_text_demo_72b.sh` - 文本演示的72B模型运行脚本
- `eval_visual_demo_72b.sh` - 视觉演示的72B模型运行脚本

## 与原版脚本的区别

### 原版脚本 (`eval_text_demo.sh`, `eval_visual_demo.sh`)
- 使用**数据并行**：每个GPU加载一个完整模型实例
- 多进程架构：4个GPU = 4个独立进程
- 适合小模型（3B, 7B）
- 每个GPU处理不同的数据批次

### 72B版本脚本 (`eval_text_demo_72b.sh`, `eval_visual_demo_72b.sh`)
- 使用**模型并行**：一个模型分布在多张GPU上
- 单进程架构：所有GPU协同工作
- 适合大模型（32B, 72B）
- 模型自动分布到4张GPU上

## 使用方法

### 1. 运行文本演示评估（72B模型）

```bash
cd /Users/cxqian/Codes/ProgressLM/eval/scripts
./eval_text_demo_72b.sh
```

### 2. 运行视觉演示评估（72B模型）

```bash
cd /Users/cxqian/Codes/ProgressLM/eval/scripts
./eval_visual_demo_72b.sh
```

## 配置说明

### GPU配置
```bash
GPU_IDS="0,1,2,3"  # 使用4张GPU进行模型并行
BATCH_SIZE=1       # 72B模型建议使用小batch（可根据显存调整）
```

### 模型路径
默认配置指向：
```bash
MODEL_PATH="/projects/b1222/userdata/jianshu/chengxuan/saved/models/Qwen2.5-VL-72B-Instruct"
```

### 数据集路径
- 文本演示：`/projects/b1222/userdata/jianshu/chengxuan/ProgressLM/data/eval/text/all_text.jsonl`
- 视觉演示：`/projects/b1222/userdata/jianshu/chengxuan/ProgressLM/data/eval/visual/visual_eval_all.jsonl`

### 输出路径
- 文本演示：`/projects/b1222/userdata/jianshu/chengxuan/saved/saved_results/progresslm/eval_text_72b/`
- 视觉演示：`/projects/b1222/userdata/jianshu/chengxuan/saved/saved_results/progresslm/eval_visual_72b/`

## 运行原理

### 模型加载过程
1. 设置环境变量 `CUDA_VISIBLE_DEVICES=0,1,2,3`，让所有4张GPU对模型可见
2. Python脚本检测模型路径包含 "72b"，自动使用 `device_map='auto'`
3. Hugging Face Transformers 自动将模型层分布到4张GPU上
4. 单进程按批次顺序处理数据

### 显存分配
72B模型（FP16）约需 144GB 显存：
- GPU 0: ~36GB（包含embedding和部分层）
- GPU 1: ~36GB（中间层）
- GPU 2: ~36GB（中间层）
- GPU 3: ~36GB（包含输出层）

实际分配由 Hugging Face 的 `device_map='auto'` 自动优化。

## 性能优化建议

### Batch Size调整
- **BATCH_SIZE=1**：最安全，适合72B模型
- **BATCH_SIZE=2**：如果显存充足（80G GPU），可以尝试
- **不建议 >2**：可能导致OOM（显存溢出）

### 推理速度
- 72B模型比3B/7B慢很多（约10-20倍）
- 单进程意味着无法并行处理数据
- 预计速度：约 0.5-2 samples/min（取决于图像数量和复杂度）

### 监控建议
运行时使用 `nvidia-smi` 或 `nvitop` 监控GPU状态：
```bash
# 在另一个终端运行
watch -n 1 nvidia-smi
```

## 故障排查

### 问题1：显存不足（OOM）
**解决方案：**
- 减小 BATCH_SIZE 到 1
- 减小 MAX_NEW_TOKENS
- 减小 MIN_PIXELS 或 MAX_PIXELS

### 问题2：模型未正确分布到多GPU
**检查：**
- 确认 `CUDA_VISIBLE_DEVICES=0,1,2,3` 设置正确
- 确认模型路径包含 "72b" 或 "32b"（不区分大小写）
- 或设置环境变量 `export AUTO_SPLIT=1`

### 问题3：进程卡住不动
**原因：**
- 可能是模型加载时间过长（72B模型需要几分钟）
- 查看日志确认是否在 "Loading model..." 阶段

**解决：**
- 耐心等待模型加载完成
- 查看 GPU 显存占用确认加载进度

## 与原多进程版本的对比

| 特性 | 多进程版本 | 单进程72B版本 |
|------|-----------|--------------|
| 适用模型 | 3B, 7B | 32B, 72B |
| GPU使用 | 数据并行 | 模型并行 |
| 进程数 | 4个 | 1个 |
| Batch处理 | 并行 | 串行 |
| 显存需求/GPU | ~20GB | ~36GB |
| 推理速度 | 快（4x并行） | 慢（单进程） |
| 适用场景 | 小模型 | 大模型 |

## 参考文档

- [Hugging Face - Model Parallelism](https://huggingface.co/docs/transformers/v4.15.0/parallelism)
- [Qwen2-VL Documentation](https://github.com/QwenLM/Qwen2-VL)
- Original scripts: `eval_text_demo.sh`, `eval_visual_demo.sh`
