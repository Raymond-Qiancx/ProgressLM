# VLAC Example 数据集评估使用指南

## 📋 概述

本目录包含了用于测试 Visual Demo Progress Estimation 模型的 VLAC 示例数据集和评估脚本。

## 🗂️ 文件说明

- **vlac_example_visual_demo.jsonl**: 生成的18条测试数据
- **eval_vlac_example.sh**: 评估脚本
- **convert_to_jsonl.py**: 数据转换脚本（已执行）
- **images/**: 图像数据
  - `ref/`: 参考演示图像（6个时间步 × 3个视角）
  - `test/`: 待评估图像（6个时间步 × 3个视角）

## 🚀 快速开始

### 1. 配置模型路径

编辑 `eval_vlac_example.sh`，设置您的模型路径：

```bash
# 修改第14行
MODEL_PATH="/path/to/your/model"

# 改为实际路径，例如：
MODEL_PATH="/projects/b1222/userdata/jianshu/chengxuan/saved/saved_results/progresslm/models/progresslm_sft_epoch2_model"
```

### 2. 运行评估

```bash
cd /Users/cxqian/Codes/ProgressLM/ref_codes/VLAC/evo_vlac/examples
bash eval_vlac_example.sh
```

### 3. 查看结果

结果将保存在 `results/` 目录下：
- `eval_vlac_example_{timestamp}.jsonl`: 详细评估结果
- `eval_vlac_example_{timestamp}_summary.json`: 汇总统计
- `eval_vlac_example_{timestamp}.log`: 运行日志

## ⚙️ 配置参数

### GPU 配置
```bash
GPU_IDS="0"        # 单GPU
GPU_IDS="0,1"      # 双GPU并行
```

### 批处理大小
```bash
BATCH_SIZE=2       # 推荐2-4（数据集只有18条）
```

### 推理次数
```bash
NUM_INFERENCES=1   # 每个样本推理1次（快速测试）
NUM_INFERENCES=4   # 每个样本推理4次（评估一致性）
```

### 样本限制
```bash
LIMIT=-1           # 处理全部18条数据
LIMIT=5            # 仅处理前5条（快速调试）
```

## 📊 数据集详情

- **总样本数**: 18条
- **任务**: "Scoop the rice into the rice cooker."
- **相机视角**: 3个（camera_0, camera_1, camera_2）
- **进度覆盖**: 0%, 16%, 33%, 50%, 66%, 83%
- **Visual Demo**: 每条记录包含6张参考图像
- **Stage to Estimate**: 每条记录评估1张测试图像

## 🎯 预期输出格式

每条结果包含：
```json
{
  "ref": 2,                          // 最接近的参考图像索引（1-based）
  "ref_think": "reasoning...",       // 参考选择推理过程
  "score": "33%",                    // 预测的进度分数
  "score_think": "reasoning...",     // 进度估计推理过程
  "ground_truth_ref": "1",           // 真实的最接近索引
  "ground_truth_score": "16%",       // 真实进度
  "response": "完整模型输出...",
  "meta_data": {
    "id": "vlac_example/595-565/camera_0_test_01",
    "task_goal": "Scoop the rice into the rice cooker.",
    "stage_to_estimate": "images/test/595-44-565-0.jpg",
    "status": "success"
  }
}
```

## 📈 评估指标

脚本会自动计算：

1. **Ref Accuracy**: 参考图像选择准确率
2. **Mean Score Error**: 进度估计平均误差
3. **Parse Error Rate**: 解析失败率
4. **Processing Time**: 处理时间统计

## 🔧 故障排除

### 模型路径错误
```
Error: Model directory not found: /path/to/your/model
```
→ 检查并更新脚本中的 `MODEL_PATH`

### 数据集未生成
```
Error: Dataset file not found
```
→ 运行 `python convert_to_jsonl.py` 生成数据集

### GPU内存不足
```
CUDA out of memory
```
→ 减小 `BATCH_SIZE` 或设置 `LIMIT` 限制样本数

### Python脚本未找到
```
Error: Python script not found
```
→ 确保在正确的项目目录结构下运行

## 💡 高级用法

### 多次推理评估一致性
```bash
NUM_INFERENCES=10  # 每个样本推理10次
```
这将生成 18 × 10 = 180 条结果，可用于分析模型输出的稳定性。

### 调试模式
```bash
LIMIT=3           # 只处理3条数据
VERBOSE=true      # 启用详细输出
```

### 不同温度测试
```bash
TEMPERATURE=0.1   # 更确定的输出
TEMPERATURE=0.9   # 更多样化的输出
```

## 📝 示例命令

### 快速测试（前3条数据）
```bash
# 编辑脚本，设置：
LIMIT=3
bash eval_vlac_example.sh
```

### 完整评估
```bash
# 编辑脚本，设置：
LIMIT=-1
NUM_INFERENCES=1
bash eval_vlac_example.sh
```

### 一致性评估（多次推理）
```bash
# 编辑脚本，设置：
LIMIT=-1
NUM_INFERENCES=5
bash eval_vlac_example.sh
```
