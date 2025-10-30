from huggingface_hub import HfApi
import os

# === 请在这里配置 ===
HF_TOKEN = "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"           # 👈 你的 Hugging Face 访问令牌
REPO_ID = "Raymond-Qiancx/FRM_SFT_3B"     # 👈 目标仓库名（username/仓库名）
MODEL_PATH = "/projects/b1222/userdata/jianshu/chengxuan/saved/saved_results/progresslm/models/3b_sft_qwen25vl_4epoch"                 # 👈 本地模型文件夹路径

# === 初始化 API ===
api = HfApi(token=HF_TOKEN)

# # === 如果仓库不存在则自动创建 ===
# try:
#     api.create_repo(
#         name=REPO_ID.split("/")[-1],
#         repo_type="model",
#         private=False,     # 若希望私有仓库，请改为 True
#         exist_ok=True
#     )
#     print(f"✅ 仓库 {REPO_ID} 已存在或创建成功")
# except Exception as e:
#     print("❌ 创建仓库失败：", e)

# === 上传整个文件夹 ===
try:
    api.upload_folder(
        folder_path=MODEL_PATH,
        repo_id=REPO_ID,
        repo_type="model",
        path_in_repo="",       # 上传到根目录
    )
    print(f"✅ 模型文件夹已成功上传到 https://huggingface.co/{REPO_ID}")
except Exception as e:
    print("❌ 上传失败：", e)
