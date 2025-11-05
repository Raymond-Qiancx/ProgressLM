#!/usr/bin/env python3
"""
可视化数据标注器
用于标注 edited_raw_all.jsonl 中的数据，显示图片和元数据，支持 Yes/No 标注
"""

import json
import os
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from PIL import Image, ImageTk
import sys


class VisualAnnotationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("可视化数据标注器")
        self.root.geometry("1200x900")

        # 配置文件路径
        self.jsonl_path = "/gpfs/projects/p32958/chengxuan/ProgressLM/data/raw/edit_imgs/edited_raw_all.jsonl"
        self.image_base_path = "/gpfs/projects/p32958/chengxuan/results/progresslm/negative/image/"
        self.output_path = "/gpfs/projects/p32958/chengxuan/ProgressLM/data/utils_img/visual_nega/annotated_output.jsonl"
        self.progress_path = "/gpfs/projects/p32958/chengxuan/ProgressLM/data/utils_img/visual_nega/annotation_progress.json"

        # 数据存储
        self.all_data = []
        self.current_index = 0
        self.annotations = {}  # {index: True/False}  True=Yes, False=No

        # 加载数据
        self.load_data()
        self.load_progress()

        # 创建UI
        self.create_ui()

        # 绑定快捷键
        self.root.bind('y', lambda e: self.annotate_yes())
        self.root.bind('Y', lambda e: self.annotate_yes())
        self.root.bind('n', lambda e: self.annotate_no())
        self.root.bind('N', lambda e: self.annotate_no())
        self.root.bind('<Left>', lambda e: self.previous_record())
        self.root.bind('<Right>', lambda e: self.next_record())
        self.root.bind('<Control-s>', lambda e: self.save_and_exit())

        # 显示第一条记录
        self.display_current_record()

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
            messagebox.showerror("错误", f"加载数据失败: {e}")
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

    def create_ui(self):
        """创建UI界面"""
        # 主容器
        main_container = ttk.Frame(self.root, padding="10")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_container.columnconfigure(0, weight=1)
        main_container.rowconfigure(1, weight=1)

        # 顶部进度条
        self.progress_label = ttk.Label(main_container, text="", font=('Arial', 12, 'bold'))
        self.progress_label.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)

        # 中间内容区域（左右分栏）
        content_frame = ttk.Frame(main_container)
        content_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=1)
        content_frame.rowconfigure(0, weight=1)

        # 左侧：文本信息区域
        left_frame = ttk.LabelFrame(content_frame, text="数据信息", padding="10")
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        left_frame.rowconfigure(0, weight=1)
        left_frame.columnconfigure(0, weight=1)

        self.text_area = scrolledtext.ScrolledText(left_frame, wrap=tk.WORD, width=50, height=30)
        self.text_area.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 右侧：图片显示区域
        right_frame = ttk.LabelFrame(content_frame, text="编辑后的图片", padding="10")
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        right_frame.rowconfigure(0, weight=1)
        right_frame.columnconfigure(0, weight=1)

        self.image_label = ttk.Label(right_frame, text="图片加载中...", anchor='center')
        self.image_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 底部控制按钮
        control_frame = ttk.Frame(main_container, padding="10")
        control_frame.grid(row=2, column=0, sticky=(tk.W, tk.E))

        # 按钮布局
        ttk.Button(control_frame, text="← 上一条 (Left)", command=self.previous_record).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="✓ YES (Y)", command=self.annotate_yes,
                   style='Success.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="✗ NO (N)", command=self.annotate_no,
                   style='Danger.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="跳过 →", command=self.skip_record).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="保存并退出 (Ctrl+S)", command=self.save_and_exit).pack(side=tk.RIGHT, padx=5)

        # 统计信息
        self.stats_label = ttk.Label(control_frame, text="", font=('Arial', 10))
        self.stats_label.pack(side=tk.RIGHT, padx=20)

        # 配置按钮样式
        style = ttk.Style()
        style.configure('Success.TButton', foreground='green')
        style.configure('Danger.TButton', foreground='red')

    def display_current_record(self):
        """显示当前记录"""
        if not self.all_data or self.current_index >= len(self.all_data):
            messagebox.showinfo("完成", "所有数据已标注完成！")
            self.save_and_exit()
            return

        record = self.all_data[self.current_index]

        # 更新进度标签
        progress_text = f"记录 {self.current_index + 1} / {len(self.all_data)}"
        if self.current_index in self.annotations:
            status = "✓ YES" if self.annotations[self.current_index] else "✗ NO"
            progress_text += f"  (已标注: {status})"
        self.progress_label.config(text=progress_text)

        # 更新统计信息
        yes_count = sum(1 for v in self.annotations.values() if v)
        no_count = sum(1 for v in self.annotations.values() if not v)
        total_annotated = len(self.annotations)
        self.stats_label.config(
            text=f"已标注: {total_annotated} | YES: {yes_count} | NO: {no_count}"
        )

        # 显示文本信息
        self.text_area.delete('1.0', tk.END)
        self.text_area.insert(tk.END, "=" * 60 + "\n")
        self.text_area.insert(tk.END, "STRATEGY\n")
        self.text_area.insert(tk.END, "=" * 60 + "\n")
        self.text_area.insert(tk.END, f"{record.get('strategy', 'N/A')}\n\n")

        self.text_area.insert(tk.END, "=" * 60 + "\n")
        self.text_area.insert(tk.END, "PROMPT\n")
        self.text_area.insert(tk.END, "=" * 60 + "\n")
        self.text_area.insert(tk.END, f"{record.get('prompt', 'N/A')}\n\n")

        self.text_area.insert(tk.END, "=" * 60 + "\n")
        self.text_area.insert(tk.END, "RAW DEMO\n")
        self.text_area.insert(tk.END, "=" * 60 + "\n")
        self.text_area.insert(tk.END, f"{record.get('raw_demo', 'N/A')}\n\n")

        self.text_area.insert(tk.END, "=" * 60 + "\n")
        self.text_area.insert(tk.END, "META DATA\n")
        self.text_area.insert(tk.END, "=" * 60 + "\n")

        meta_data = record.get('meta_data', {})
        for key, value in meta_data.items():
            if key == 'text_demo' and isinstance(value, list):
                self.text_area.insert(tk.END, f"{key}:\n")
                for i, step in enumerate(value, 1):
                    self.text_area.insert(tk.END, f"  {i}. {step}\n")
            else:
                self.text_area.insert(tk.END, f"{key}: {value}\n")

        # 显示图片
        self.display_image(meta_data)

    def display_image(self, meta_data):
        """显示编辑后的图片"""
        try:
            # 构建图片路径
            image_id = meta_data.get('id', '')
            image_name = meta_data.get('image', '')

            if not image_id or not image_name:
                self.image_label.config(text="图片信息缺失", image='')
                return

            # 将 .jpg 替换为 _edit.jpg
            if image_name.endswith('.jpg'):
                edited_image_name = image_name.replace('.jpg', '_edit.jpg')
            else:
                edited_image_name = image_name + '_edit.jpg'

            image_path = os.path.join(self.image_base_path, image_id, edited_image_name)

            if not os.path.exists(image_path):
                self.image_label.config(
                    text=f"图片不存在:\n{image_path}",
                    image='',
                    foreground='red'
                )
                return

            # 加载并显示图片
            img = Image.open(image_path)

            # 调整图片大小以适应显示区域
            max_width = 600
            max_height = 700
            img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)

            # 转换为PhotoImage
            photo = ImageTk.PhotoImage(img)

            # 保持引用防止垃圾回收
            self.current_photo = photo

            self.image_label.config(image=photo, text='', foreground='black')

        except Exception as e:
            self.image_label.config(
                text=f"图片加载失败:\n{str(e)}",
                image='',
                foreground='red'
            )

    def annotate_yes(self):
        """标注为 YES（保留）"""
        self.annotations[self.current_index] = True
        self.save_progress()
        self.next_record()

    def annotate_no(self):
        """标注为 NO（删除）"""
        self.annotations[self.current_index] = False
        self.save_progress()
        self.next_record()

    def skip_record(self):
        """跳过当前记录"""
        self.next_record()

    def next_record(self):
        """显示下一条记录"""
        if self.current_index < len(self.all_data) - 1:
            self.current_index += 1
            self.display_current_record()
        else:
            messagebox.showinfo("提示", "已经是最后一条记录了")

    def previous_record(self):
        """显示上一条记录"""
        if self.current_index > 0:
            self.current_index -= 1
            self.display_current_record()
        else:
            messagebox.showinfo("提示", "已经是第一条记录了")

    def save_and_exit(self):
        """保存结果并退出"""
        if not self.annotations:
            result = messagebox.askyesno("确认", "没有任何标注，确定要退出吗？")
            if not result:
                return

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
                f.write(f"保留率: {yes_count / total_annotated * 100:.2f}%\n" if total_annotated > 0 else "保留率: N/A\n")

            messagebox.showinfo(
                "保存成功",
                f"标注结果已保存!\n\n"
                f"输出文件: {self.output_path}\n"
                f"统计文件: {stats_path}\n\n"
                f"保留记录数: {yes_count}\n"
                f"删除记录数: {no_count}"
            )

            # 删除进度文件
            if os.path.exists(self.progress_path):
                os.remove(self.progress_path)

            self.root.quit()

        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {e}")


def main():
    """主函数"""
    root = tk.Tk()
    app = VisualAnnotationTool(root)
    root.mainloop()


if __name__ == '__main__':
    main()
