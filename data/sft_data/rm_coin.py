# # # import json

# # # def remove_coin_images(input_path, output_path):
# # #     """
# # #     删除 jsonl 文件中 images 字段中包含 'coin' 的项，并生成新的文件。
# # #     处理结果直接打印在命令行。
# # #     """
# # #     modified_count = 0
# # #     total_count = 0

# # #     with open(input_path, 'r', encoding='utf-8') as infile, \
# # #          open(output_path, 'w', encoding='utf-8') as outfile:

# # #         for line_num, line in enumerate(infile, start=1):
# # #             line = line.strip()
# # #             if not line:
# # #                 continue
# # #             total_count += 1

# # #             try:
# # #                 data = json.loads(line)
# # #             except json.JSONDecodeError as e:
# # #                 print(f"[第 {line_num} 行] ❌ JSON 解析错误: {e}")
# # #                 continue

# # #             images = data.get("images", [])
# # #             if any("coin" in str(img) for img in images):
# # #                 data.pop("images", None)
# # #                 modified_count += 1
# # #                 print(f"[第 {line_num} 行] 🧹 删除 images 字段（含 'coin'）")

# # #             json.dump(data, outfile, ensure_ascii=False)
# # #             outfile.write('\n')

# # #     print("\n✅ 处理完成！")
# # #     print(f"总行数: {total_count}")
# # #     print(f"修改行数: {modified_count}")
# # #     print(f"输出文件: {output_path}")


# # # if __name__ == "__main__":
# # #     # 输入、输出文件路径，可按需修改
# # #     input_file = "/projects/b1222/userdata/jianshu/chengxuan/ProgressLM/data/sft_data/sft_steps_3to7.jsonl"
# # #     output_file = "/projects/b1222/userdata/jianshu/chengxuan/ProgressLM/data/sft_data/sft_no_coin.jsonl"

# # #     remove_coin_images(input_file, output_file)

# # import json
# # import re

# # def add_percent_to_scores(input_path, output_path):
# #     """
# #     在 jsonl 文件中为 <score></score> 标签内没有 '%' 的数值加上百分号。
# #     修改结果写入新的文件，并在命令行显示修改情况。
# #     """
# #     modified_count = 0
# #     total_count = 0

# #     # 匹配 <score>数字</score> 的正则（不含 %）
# #     score_pattern = re.compile(r"<score>(\s*[\d.]+)\s*</score>")

# #     with open(input_path, 'r', encoding='utf-8') as infile, \
# #          open(output_path, 'w', encoding='utf-8') as outfile:

# #         for line_num, line in enumerate(infile, start=1):
# #             line = line.strip()
# #             if not line:
# #                 continue
# #             total_count += 1

# #             try:
# #                 data = json.loads(line)
# #             except json.JSONDecodeError as e:
# #                 print(f"[第 {line_num} 行] ❌ JSON 解析错误: {e}")
# #                 continue

# #             modified = False

# #             # 如果 assistant 内容里有 <score> 标签，处理它
# #             if "messages" in data:
# #                 for msg in data["messages"]:
# #                     if msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
# #                         content = msg["content"]

# #                         # 查找所有没有百分号的 score 标签
# #                         def add_percent(match):
# #                             nonlocal modified
# #                             modified = True
# #                             value = match.group(1).strip()
# #                             return f"<score>{value}%</score>"

# #                         new_content = score_pattern.sub(add_percent, content)
# #                         msg["content"] = new_content

# #             if modified:
# #                 modified_count += 1
# #                 print(f"[第 {line_num} 行] ✅ 已为 <score> 补上 '%'")

# #             json.dump(data, outfile, ensure_ascii=False)
# #             outfile.write('\n')

# #     print("\n✅ 处理完成！")
# #     print(f"总行数: {total_count}")
# #     print(f"修改行数: {modified_count}")
# #     print(f"输出文件: {output_path}")


# # if __name__ == "__main__":
# #     # 输入与输出路径，可按需修改
# #     input_file = "/projects/b1222/userdata/jianshu/chengxuan/ProgressLM/data/sft_data/sft_no_coin.jsonl"
# #     output_file = "/projects/b1222/userdata/jianshu/chengxuan/ProgressLM/data/sft_data/sft_final.jsonl"

# #     add_percent_to_scores(input_file, output_file)

# # #!/usr/bin/env python3
# # import json
# # from tqdm import tqdm
# # import argparse
# # import re
# # import os

# # def count_image_tokens(text):
# #     """统计 <image> 出现次数"""
# #     return len(re.findall(r"<image>", text))

# # def check_sft_file(input_path, output_path=None):
# #     bad_samples = []
# #     total = 0
# #     kept = 0

# #     with open(input_path, "r", encoding="utf-8") as fin:
# #         lines = fin.readlines()

# #     if output_path:
# #         fout = open(output_path, "w", encoding="utf-8")

# #     for i, line in enumerate(tqdm(lines, desc="Checking samples")):
# #         total += 1
# #         try:
# #             obj = json.loads(line)
# #         except Exception as e:
# #             bad_samples.append((i+1, "JSONDecodeError", str(e)))
# #             continue

# #         # 提取所有消息文本
# #         messages = obj.get("messages", [])
# #         if not messages:
# #             bad_samples.append((i+1, "NoMessages", "Missing 'messages' field"))
# #             continue

# #         text = "".join(m.get("content", "") for m in messages)
# #         num_tokens = count_image_tokens(text)
# #         num_images = len(obj.get("images", []))

# #         if num_tokens != num_images:
# #             bad_samples.append((i+1, f"{num_tokens} <image>", f"{num_images} images"))
# #             continue

# #         if output_path:
# #             fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
# #             kept += 1

# #     if output_path:
# #         fout.close()

# #     print("\n=== 检查结果 ===")
# #     print(f"总样本数: {total}")
# #     print(f"错误样本数: {len(bad_samples)}")
# #     if output_path:
# #         print(f"已保存干净样本: {kept} -> {output_path}")

# #     if bad_samples:
# #         print("\n前 10 个问题样本:")
# #         for idx, token_info, img_info in bad_samples[:10]:
# #             print(f"  行 {idx}: {token_info} / {img_info}")

# #     return bad_samples


# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser(description="检查并清洗 SFT 数据集中的 <image> 不匹配样本")
# #     parser.add_argument("--input", required=True, help="输入 JSONL 文件路径")
# #     parser.add_argument("--output", help="输出清洗后的文件路径（可选）")
# #     args = parser.parse_args()

# #     if not os.path.exists(args.input):
# #         raise FileNotFoundError(f"找不到文件: {args.input}")

# #     check_sft_file(args.input, args.output)



# import json
# import re
# from tqdm import tqdm
# import argparse
# import os

# def count_image_tokens(text):
#     """统计 <image> 出现次数"""
#     return len(re.findall(r"<image>", text))

# def clean_sft_file(input_path, output_path):
#     total, kept, removed = 0, 0, 0

#     with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
#         for i, line in enumerate(tqdm(fin, desc="Cleaning dataset")):
#             total += 1
#             try:
#                 obj = json.loads(line)
#             except Exception as e:
#                 removed += 1
#                 continue

#             messages = obj.get("messages", [])
#             if not messages:
#                 removed += 1
#                 continue

#             text = "".join(m.get("content", "") for m in messages)
#             num_tokens = count_image_tokens(text)
#             num_images = len(obj.get("images", []))

#             # 条件：数量匹配才能保留
#             if num_tokens == num_images:
#                 fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
#                 kept += 1
#             else:
#                 removed += 1

#     print("\n=== 清理完成 ===")
#     print(f"总样本数: {total}")
#     print(f"保留样本数: {kept}")
#     print(f"删除样本数: {removed}")
#     print(f"干净文件已保存到: {output_path}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="一键清除 SFT 数据集中 <image> 数量不匹配的样本")
#     parser.add_argument("--input", required=True, help="输入 JSONL 文件路径")
#     parser.add_argument("--output", required=True, help="输出清洗后文件路径")
#     args = parser.parse_args()

#     if not os.path.exists(args.input):
#         raise FileNotFoundError(f"找不到文件: {args.input}")

#     clean_sft_file(args.input, args.output)

# import json

# file_path = "/projects/b1222/userdata/jianshu/chengxuan/ProgressLM/data/sft_data/sft_final.jsonl"

# count_total = 0
# count_len1 = 0

# with open(file_path, "r", encoding="utf-8") as f:
#     for line in f:
#         data = json.loads(line)
#         if "images" in data:
#             count_total += 1
#             if isinstance(data["images"], list) and len(data["images"]) == 1:
#                 count_len1 += 1

# if count_total > 0:
#     ratio = count_len1 / count_total
#     print(f"总样本数: {count_total}")
#     print(f"images长度为1的样本数: {count_len1}")
#     print(f"比例: {ratio:.2%}")
# else:
#     print("未找到包含images字段的数据。")
