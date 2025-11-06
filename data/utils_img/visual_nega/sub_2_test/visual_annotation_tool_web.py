#!/usr/bin/env python3
"""
可视化数据标注器 - Web版本（基于Gradio）
用于标注 edited_raw_all.jsonl 中的数据，显示图片和元数据，支持 Yes/No 标注
"""

import json
import os
import gradio as gr
from PIL import Image
import sys
from datetime import datetime
import shutil


class WebAnnotationTool:
    def __init__(self):
        # 配置文件路径
        self.jsonl_path = "/projects/p32958/chengxuan/ProgressLM/data/utils_img/visual_nega/sub_2_test/sub_2_labeled.jsonl"
        self.image_base_path = "/gpfs/projects/p32958/chengxuan/results/progresslm/negative/image/"
        self.output_path = "/projects/p32958/chengxuan/ProgressLM/data/utils_img/visual_nega/sub_2_test/annotated_output.jsonl"
        self.progress_path = "/projects/p32958/chengxuan/ProgressLM/data/utils_img/visual_nega/sub_2_test/annotation_progress.json"

        # 数据存储
        self.all_data = []
        self.current_index = 0
        self.annotations = {}  # {index: True/False}  True=Yes, False=No

        # 加载数据
        self.load_data()
        self.load_progress()

    def load_data(self):
        """加载JSONL数据"""
        print(f"正在加载数据: {self.jsonl_path}")
        try:
            with open(self.jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.all_data.append(json.loads(line))
            print(f"成功加载 {len(self.all_data)} 条记录")
        except Exception as e:
            print(f"错误: 加载数据失败: {e}")
            sys.exit(1)

    def load_progress(self):
        """加载标注进度"""
        if os.path.exists(self.progress_path):
            try:
                with open(self.progress_path, 'r', encoding='utf-8') as f:
                    progress = json.load(f)
                    self.current_index = progress.get('current_index', 0)
                    self.annotations = {int(k): v for k, v in progress.get('annotations', {}).items()}
                print(f"恢复进度: 从第 {self.current_index + 1} 条记录开始")
            except Exception as e:
                print(f"加载进度文件失败: {e}")

    def save_progress(self):
        """保存标注进度"""
        try:
            progress = {
                'current_index': self.current_index,
                'annotations': self.annotations
            }
            with open(self.progress_path, 'w', encoding='utf-8') as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            print(f"保存进度失败: {e}")

    def get_image_pair(self, meta_data):
        """获取原图和编辑后图片的路径"""
        try:
            image_id = meta_data.get('id', '')
            image_name = meta_data.get('image', '')

            if not image_id or not image_name:
                return None, None

            # 构建原图路径
            original_base_path = "/gpfs/projects/p32958/chengxuan/new_extracted_images/images/"
            original_path = os.path.join(original_base_path, image_id, image_name)

            # 构建编辑后图片路径
            if image_name.endswith('.jpg'):
                edited_image_name = image_name.replace('.jpg', '_edit.jpg')
            else:
                edited_image_name = image_name + '_edit.jpg'

            edited_path = os.path.join(self.image_base_path, image_id, edited_image_name)

            # 检查文件是否存在
            original_exists = os.path.exists(original_path)
            edited_exists = os.path.exists(edited_path)

            return (original_path if original_exists else None,
                    edited_path if edited_exists else None)
        except Exception as e:
            print(f"获取图片路径失败: {e}")
            return None, None

    def format_record_info(self, record):
        """格式化记录信息为Markdown"""
        info = []

        info.append("# 数据信息\n")

        info.append("## STRATEGY")
        info.append(f"**{record.get('strategy', 'N/A')}**\n")

        info.append("## PROMPT")
        info.append(f"{record.get('prompt', 'N/A')}\n")

        info.append("## RAW DEMO")
        info.append(f"{record.get('raw_demo', 'N/A')}\n")

        info.append("## META DATA")
        meta_data = record.get('meta_data', {})

        for key, value in meta_data.items():
            if key == 'text_demo' and isinstance(value, list):
                info.append(f"**{key}:**")
                for i, step in enumerate(value, 1):
                    info.append(f"{i}. {step}")
                info.append("")
            else:
                info.append(f"**{key}:** {value}")

        return "\n\n".join(info)

    def get_current_record(self):
        """获取当前记录的所有信息"""
        if self.current_index >= len(self.all_data):
            return None, None, None, None, None

        record = self.all_data[self.current_index]

        # 获取格式化的文本信息
        info_text = self.format_record_info(record)

        # 获取原图和编辑图
        meta_data = record.get('meta_data', {})
        original_path, edited_path = self.get_image_pair(meta_data)

        # 获取进度信息
        progress_text = f"### 记录 {self.current_index + 1} / {len(self.all_data)}"
        if self.current_index in self.annotations:
            status = "✓ YES" if self.annotations[self.current_index] else "✗ NO"
            progress_text += f" (已标注: {status})"

        # 获取统计信息
        yes_count = sum(1 for v in self.annotations.values() if v)
        no_count = sum(1 for v in self.annotations.values() if not v)
        total_annotated = len(self.annotations)
        stats_text = f"**已标注:** {total_annotated} | **YES:** {yes_count} | **NO:** {no_count}"

        return info_text, original_path, edited_path, progress_text, stats_text

    def annotate_yes(self):
        """标注为 YES"""
        self.annotations[self.current_index] = True
        self.save_progress()
        self.current_index += 1
        return self.get_current_record()

    def annotate_no(self):
        """标注为 NO"""
        self.annotations[self.current_index] = False
        self.save_progress()
        self.current_index += 1
        return self.get_current_record()

    def skip_record(self):
        """跳过当前记录"""
        self.current_index += 1
        return self.get_current_record()

    def previous_record(self):
        """上一条记录"""
        if self.current_index > 0:
            self.current_index -= 1
        return self.get_current_record()

    def next_record(self):
        """下一条记录"""
        if self.current_index < len(self.all_data) - 1:
            self.current_index += 1
        return self.get_current_record()

    def save_and_finish(self):
        """保存结果"""
        if not self.annotations:
            return "⚠️ 没有任何标注，无法保存！"

        # 保存标注结果
        yes_records = []
        for idx, keep in self.annotations.items():
            if keep:
                yes_records.append(self.all_data[idx])

        # 写入输出文件
        try:
            with open(self.output_path, 'w', encoding='utf-8') as f:
                for record in yes_records:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')

            # 保存统计信息
            stats_path = self.output_path.replace('.jsonl', '_stats.txt')
            with open(stats_path, 'w', encoding='utf-8') as f:
                yes_count = sum(1 for v in self.annotations.values() if v)
                no_count = sum(1 for v in self.annotations.values() if not v)
                total_annotated = len(self.annotations)

                f.write(f"标注统计信息\n")
                f.write(f"=" * 50 + "\n")
                f.write(f"总记录数: {len(self.all_data)}\n")
                f.write(f"已标注数: {total_annotated}\n")
                f.write(f"YES (保留): {yes_count}\n")
                f.write(f"NO (删除): {no_count}\n")
                f.write(f"未标注: {len(self.all_data) - total_annotated}\n")
                if total_annotated > 0:
                    f.write(f"保留率: {yes_count / total_annotated * 100:.2f}%\n")
                else:
                    f.write(f"保留率: N/A\n")

            # 备份进度文件（而不是删除）
            if os.path.exists(self.progress_path):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = self.progress_path.replace('.json', f'_backup_{timestamp}.json')
                shutil.copy2(self.progress_path, backup_path)
                print(f"进度文件已备份到: {backup_path}")

            yes_count = sum(1 for v in self.annotations.values() if v)
            no_count = sum(1 for v in self.annotations.values() if not v)

            result_text = f"""
✅ 标注结果已保存成功！

**输出文件:** {self.output_path}
**统计文件:** {stats_path}

**保留记录数 (YES):** {yes_count}
**删除记录数 (NO):** {no_count}
**总标注数:** {len(self.annotations)}

标注工作已完成！可以关闭浏览器。
"""
            return result_text

        except Exception as e:
            return f"❌ 保存失败: {e}"


def create_ui():
    """创建Gradio界面"""
    tool = WebAnnotationTool()

    with gr.Blocks(title="可视化数据标注器", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 📝 可视化数据标注器")
        gr.Markdown("使用 **YES** / **NO** 按钮标注数据，按 **保存并完成** 导出结果")

        with gr.Row():
            progress_display = gr.Markdown(value=tool.get_current_record()[3])

        with gr.Row():
            stats_display = gr.Markdown(value=tool.get_current_record()[4])

        with gr.Row():
            # 左侧：文本信息
            with gr.Column(scale=1):
                info_display = gr.Markdown(
                    value=tool.get_current_record()[0],
                    label="数据信息"
                )

            # 右侧：图片对比区
            with gr.Column(scale=1):
                with gr.Row():
                    # 原始图片
                    original_image_display = gr.Image(
                        value=tool.get_current_record()[1],
                        label="原始图片",
                        height=600
                    )
                    # 编辑后图片
                    edited_image_display = gr.Image(
                        value=tool.get_current_record()[2],
                        label="编辑后的图片",
                        height=600
                    )

        # 控制按钮
        with gr.Row():
            prev_btn = gr.Button("⬅️ 上一条", variant="secondary")
            yes_btn = gr.Button("✅ YES (保留)", variant="primary", size="lg")
            no_btn = gr.Button("❌ NO (删除)", variant="stop", size="lg")
            skip_btn = gr.Button("⏭️ 跳过", variant="secondary")

        with gr.Row():
            next_btn = gr.Button("➡️ 下一条", variant="secondary")
            save_btn = gr.Button("💾 保存并完成", variant="primary")

        # 保存结果显示
        result_display = gr.Markdown(visible=False)

        # 按钮事件
        def update_yes():
            info, original_img, edited_img, prog, stats = tool.annotate_yes()
            if info is None:
                return {
                    info_display: "✅ 所有数据已标注完成！",
                    original_image_display: None,
                    edited_image_display: None,
                    progress_display: "完成",
                    stats_display: stats if stats else ""
                }
            return {
                info_display: info,
                original_image_display: original_img,
                edited_image_display: edited_img,
                progress_display: prog,
                stats_display: stats
            }

        def update_no():
            info, original_img, edited_img, prog, stats = tool.annotate_no()
            if info is None:
                return {
                    info_display: "✅ 所有数据已标注完成！",
                    original_image_display: None,
                    edited_image_display: None,
                    progress_display: "完成",
                    stats_display: stats if stats else ""
                }
            return {
                info_display: info,
                original_image_display: original_img,
                edited_image_display: edited_img,
                progress_display: prog,
                stats_display: stats
            }

        def update_skip():
            info, original_img, edited_img, prog, stats = tool.skip_record()
            if info is None:
                return {
                    info_display: "✅ 所有数据已标注完成！",
                    original_image_display: None,
                    edited_image_display: None,
                    progress_display: "完成",
                    stats_display: stats if stats else ""
                }
            return {
                info_display: info,
                original_image_display: original_img,
                edited_image_display: edited_img,
                progress_display: prog,
                stats_display: stats
            }

        def update_prev():
            info, original_img, edited_img, prog, stats = tool.previous_record()
            return {
                info_display: info,
                original_image_display: original_img,
                edited_image_display: edited_img,
                progress_display: prog,
                stats_display: stats
            }

        def update_next():
            info, original_img, edited_img, prog, stats = tool.next_record()
            if info is None:
                return {
                    info_display: "✅ 已经是最后一条记录！",
                    original_image_display: None,
                    edited_image_display: None,
                    progress_display: "完成",
                    stats_display: stats if stats else ""
                }
            return {
                info_display: info,
                original_image_display: original_img,
                edited_image_display: edited_img,
                progress_display: prog,
                stats_display: stats
            }

        def save_results():
            result = tool.save_and_finish()
            return {
                result_display: gr.update(value=result, visible=True)
            }

        yes_btn.click(
            update_yes,
            outputs=[info_display, original_image_display, edited_image_display, progress_display, stats_display]
        )

        no_btn.click(
            update_no,
            outputs=[info_display, original_image_display, edited_image_display, progress_display, stats_display]
        )

        skip_btn.click(
            update_skip,
            outputs=[info_display, original_image_display, edited_image_display, progress_display, stats_display]
        )

        prev_btn.click(
            update_prev,
            outputs=[info_display, original_image_display, edited_image_display, progress_display, stats_display]
        )

        next_btn.click(
            update_next,
            outputs=[info_display, original_image_display, edited_image_display, progress_display, stats_display]
        )

        save_btn.click(
            save_results,
            outputs=[result_display]
        )

        gr.Markdown("""
        ---
        ### 💡 使用说明
        - **YES**: 保留当前记录并跳到下一条
        - **NO**: 删除当前记录并跳到下一条
        - **跳过**: 不标注，直接查看下一条
        - **上一条/下一条**: 浏览和修改已标注的记录
        - **保存并完成**: 导出所有标注为YES的记录到文件

        ### 📁 输出文件
        - `annotated_output.jsonl` - 所有标注为YES的记录
        - `annotated_output_stats.txt` - 详细统计信息
        """)

    return app


def main():
    """主函数"""
    print("=" * 60)
    print("可视化数据标注器 - Web版本")
    print("=" * 60)

    app = create_ui()

    print("\n🚀 启动Web服务器...")
    print("\n" + "=" * 60)
    print("📌 访问方式：")
    print("=" * 60)
    print("1. 本地访问: http://localhost:7860")
    print("2. 远程访问: 使用下方显示的公网地址")
    print("3. SSH端口转发: ssh -L 7860:localhost:7860 user@server")
    print("=" * 60)
    print("\n按 Ctrl+C 停止服务器\n")

    # 启动Gradio服务器
    # share=True 会生成一个公网链接（临时，72小时有效）
    app.launch(
        server_name="0.0.0.0",  # 允许外部访问
        server_port=7806,
        share=True,  # 生成公网链接
        show_error=True,
        allowed_paths=[
            "/gpfs/projects/p32958/chengxuan/results/progresslm/negative/image/",  # 编辑后的图片目录
            "/gpfs/projects/p32958/chengxuan/new_extracted_images/images/"  # 原始图片目录
        ]
    )


if __name__ == '__main__':
    main()
