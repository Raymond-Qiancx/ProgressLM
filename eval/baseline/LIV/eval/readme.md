LIV 评估方案详细分析
LIV 模型核心能力
LIV 模型能做的事情：
┌─────────────┐                    ┌─────────────┐
│   图像      │ ──→ liv(vision) ──→│  1024维向量  │
└─────────────┘                    └─────────────┘

┌─────────────┐                    ┌─────────────┐
│   文本      │ ──→ liv(text)  ──→ │  1024维向量  │
└─────────────┘                    └─────────────┘

两个向量 ──→ sim() ──→ 相似度分数 (cosine similarity)
1. Text Demo 任务的 LIV 评估
输入
输入项	类型	示例
当前状态图像	Image Tensor [1, 3, H, W]	stage_to_estimate 图像
文本步骤列表	List[str]	["reach for object", "grasp object", "lift object"]
任务目标	str (可选)	"pick up the red block"
计算流程
Text Demo 评估流程:

当前状态图像                    文本步骤列表
     │                              │
     ▼                              ▼
┌──────────────┐            ┌──────────────────────┐
│ liv(vision)  │            │ clip.tokenize()      │
└──────────────┘            └──────────────────────┘
     │                              │
     ▼                              ▼
┌──────────────┐            ┌──────────────────────┐
│ img_emb      │            │ text_tokens          │
│ [1, 1024]    │            │ [N, 77]              │
└──────────────┘            └──────────────────────┘
                                    │
                                    ▼
                            ┌──────────────────────┐
                            │ liv(text)            │
                            └──────────────────────┘
                                    │
                                    ▼
                            ┌──────────────────────┐
                            │ text_embs            │
                            │ [N, 1024]            │
                            └──────────────────────┘
     │                              │
     └──────────┬───────────────────┘
                │
                ▼
        ┌───────────────┐
        │ sim(img, text)│  计算余弦相似度
        └───────────────┘
                │
                ▼
        ┌───────────────┐
        │ similarities  │  [N] 个相似度分数
        │ [0.2, 0.8, 0.3]│
        └───────────────┘
                │
                ▼
        ┌───────────────┐
        │ argmax + 1    │  找最大相似度的索引
        └───────────────┘
                │
                ▼
        pred_ref = 2 (1-based)
        pred_score = 2/3 = 66.7%
具体代码逻辑
# 输入
image: Tensor [1, 3, H, W]      # 当前状态图像
text_demo: ["step1", "step2", "step3"]  # 3个文本步骤

# Step 1: 编码图像
img_embedding = liv(input=image, modality="vision")  # [1, 1024]

# Step 2: 编码文本
text_tokens = clip.tokenize(text_demo)  # [3, 77]
text_embeddings = liv(input=text_tokens, modality="text")  # [3, 1024]

# Step 3: 计算相似度
similarities = liv.module.sim(img_embedding.expand(3, -1), text_embeddings)
# similarities = [0.15, 0.72, 0.31]  (示例)

# Step 4: 生成输出
pred_ref = similarities.argmax().item() + 1  # = 2 (1-based)
pred_score = pred_ref / len(text_demo)       # = 2/3 = 0.667
输出
输出项	类型	说明
pred_ref	int	最相似的文本步骤索引（1-based）
pred_score	float	进度分数 = pred_ref / total_steps
similarities	List[float]	每个步骤的相似度分数（可选，用于分析）
2. Visual Demo 任务的 LIV 评估
输入
输入项	类型	示例
当前状态图像	Image Tensor [1, 3, H, W]	stage_to_estimate 图像
演示图像序列	List[Image] [N, 3, H, W]	visual_demo 的 N 张图像
任务目标	str (可选)	"pick up the red block"
计算流程
Visual Demo 评估流程:

当前状态图像                    演示图像序列 (N张)
     │                              │
     ▼                              ▼
┌──────────────┐            ┌──────────────────────┐
│ liv(vision)  │            │ liv(vision)          │
└──────────────┘            └──────────────────────┘
     │                              │
     ▼                              ▼
┌──────────────┐            ┌──────────────────────┐
│ stage_emb    │            │ demo_embs            │
│ [1, 1024]    │            │ [N, 1024]            │
└──────────────┘            └──────────────────────┘
     │                              │
     └──────────┬───────────────────┘
                │
                ▼
        ┌───────────────┐
        │ sim(stage,    │  计算余弦相似度
        │     demo)     │
        └───────────────┘
                │
                ▼
        ┌───────────────────────┐
        │ similarities          │
        │ [0.9, 0.7, 0.5, 0.3, 0.2]│  N个相似度分数
        └───────────────────────┘
                │
                ▼
        ┌───────────────┐
        │ argmax + 1    │
        └───────────────┘
                │
                ▼
        pred_ref = 1 (1-based)
        pred_score = (1-1)/4 = 0%  (第1张是0%进度)
具体代码逻辑
# 输入
stage_image: Tensor [1, 3, H, W]  # 当前状态图像
demo_images: Tensor [5, 3, H, W]  # 5张演示图像 (0%, 25%, 50%, 75%, 100%)
total_steps = 4  # 演示图像数 - 1

# Step 1: 编码当前状态图像
stage_embedding = liv(input=stage_image, modality="vision")  # [1, 1024]

# Step 2: 编码所有演示图像
demo_embeddings = liv(input=demo_images, modality="vision")  # [5, 1024]

# Step 3: 计算相似度
similarities = liv.module.sim(stage_embedding.expand(5, -1), demo_embeddings)
# similarities = [0.85, 0.72, 0.45, 0.30, 0.20]  (示例)

# Step 4: 生成输出
pred_ref = similarities.argmax().item() + 1  # = 1 (1-based)
pred_score = (pred_ref - 1) / total_steps    # = 0/4 = 0.0 (0%)
输出
输出项	类型	说明
pred_ref	int	最相似的演示图像索引（1-based）
pred_score	float	进度分数 = (pred_ref - 1) / total_steps
similarities	List[float]	每个演示图像的相似度分数
总结对比
方面	Text Demo	Visual Demo
图像输入	当前状态 1张	当前状态 1张
参考输入	N 个文本步骤	N 张演示图像
LIV编码	vision + text	vision + vision
相似度计算	图像 vs 文本	图像 vs 图像
pred_ref	argmax(sim) + 1	argmax(sim) + 1
pred_score	ref / N	(ref - 1) / (N - 1)
评估指标
两种任务都可以计算相同的指标：
# 1. Ref Error: |pred_ref - gt_ref|
ref_error = abs(pred_ref - gt_closest_idx)

# 2. Score Error: |pred_score - gt_score|
score_error = abs(pred_score - gt_progress_score)

# 3. VOC (Spearman correlation): 轨迹内的排序一致性