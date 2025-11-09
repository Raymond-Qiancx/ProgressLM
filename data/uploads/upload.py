from huggingface_hub import HfApi

# ======== 配置信息 ========
token = "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"  # ← 在这里填入你的 Hugging Face token
repo_id = "Raymond-Qiancx/MSCOCO"  # ← 例如 "vcj9002/my-dataset"
file_path = "/home/vcj9002/jianshu/chengxuan/images.tar.gz"
target_path = "images"  # 上传后在仓库中的路径
repo_type = "dataset"  # 可选值: "model" | "dataset" | "space"
# ===========================

api = HfApi()

api.upload_file(
    path_or_fileobj=file_path,
    path_in_repo=target_path,
    repo_id=repo_id,
    repo_type=repo_type,
    token=token,
)

print("✅ 文件上传成功！")
