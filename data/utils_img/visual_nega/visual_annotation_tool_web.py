#!/usr/bin/env python3
"""
Visual Data Annotation Tool - Web Version (Gradio-based)
For annotating data in annotated_raw.jsonl, displaying images and metadata, supporting Yes/No annotation
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
        # Configuration file paths
        self.base_dir = "/gpfs/projects/p32958/chengxuan/ProgressLM/data/utils_img/visual_nega/sub_1_sub"
        self.jsonl_path = os.path.join(self.base_dir, "annotated_raw.jsonl")
        self.image_base_path = "/gpfs/projects/p32958/chengxuan/results/progresslm/negative/image/"
        self.output_path = os.path.join(self.base_dir, "annotated_output.jsonl")
        self.progress_path = os.path.join(self.base_dir, "annotation_progress.json")

        # Data storage
        self.all_data = []
        self.current_index = 0
        self.annotations = {}  # {index: True/False}  True=Yes, False=No

        # Load data
        self.load_data()
        self.load_progress()

    def load_data(self):
        """Load JSONL data"""
        print(f"Loading data: {self.jsonl_path}")
        try:
            with open(self.jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.all_data.append(json.loads(line))
            print(f"Successfully loaded {len(self.all_data)} records")
        except Exception as e:
            print(f"Error: Failed to load data: {e}")
            sys.exit(1)

    def load_progress(self):
        """Load annotation progress"""
        if os.path.exists(self.progress_path):
            try:
                with open(self.progress_path, 'r', encoding='utf-8') as f:
                    progress = json.load(f)
                    self.current_index = progress.get('current_index', 0)
                    self.annotations = {int(k): v for k, v in progress.get('annotations', {}).items()}
                print(f"Restored progress: Starting from record {self.current_index + 1}")
            except Exception as e:
                print(f"Failed to load progress file: {e}")

    def save_progress(self):
        """Save annotation progress"""
        try:
            progress = {
                'current_index': self.current_index,
                'annotations': self.annotations
            }
            with open(self.progress_path, 'w', encoding='utf-8') as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            print(f"Failed to save progress: {e}")

    def get_image_pair(self, meta_data):
        """Get original and edited image paths"""
        try:
            image_id = meta_data.get('id', '')
            image_name = meta_data.get('image', '')

            if not image_id or not image_name:
                return None, None

            # Build original image path
            original_base_path = "/projects/p32958/chengxuan/data/images/"
            original_path = os.path.join(original_base_path, image_id, image_name)

            # Build edited image path
            if image_name.endswith('.jpg'):
                edited_image_name = image_name.replace('.jpg', '_edit.jpg')
            else:
                edited_image_name = image_name + '_edit.jpg'

            edited_path = os.path.join(self.image_base_path, image_id, edited_image_name)

            # Check if files exist
            original_exists = os.path.exists(original_path)
            edited_exists = os.path.exists(edited_path)

            return (original_path if original_exists else None,
                    edited_path if edited_exists else None)
        except Exception as e:
            print(f"Failed to get image path: {e}")
            return None, None

    def format_record_info(self, record):
        """Format record info as Markdown"""
        info = []

        info.append("# Data Information\n")

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
        """Get all information of current record"""
        if self.current_index >= len(self.all_data):
            return None, None, None, None, None

        record = self.all_data[self.current_index]

        # Get formatted text info
        info_text = self.format_record_info(record)

        # Get original and edited images
        meta_data = record.get('meta_data', {})
        original_path, edited_path = self.get_image_pair(meta_data)

        # Get progress info
        progress_text = f"### Record {self.current_index + 1} / {len(self.all_data)}"
        if self.current_index in self.annotations:
            status = "✓ YES" if self.annotations[self.current_index] else "✗ NO"
            progress_text += f" (Annotated: {status})"

        # Get statistics info
        yes_count = sum(1 for v in self.annotations.values() if v)
        no_count = sum(1 for v in self.annotations.values() if not v)
        total_annotated = len(self.annotations)
        stats_text = f"**Annotated:** {total_annotated} | **YES:** {yes_count} | **NO:** {no_count}"

        return info_text, original_path, edited_path, progress_text, stats_text

    def annotate_yes(self):
        """Annotate as YES"""
        self.annotations[self.current_index] = True
        self.save_progress()
        self.current_index += 1
        return self.get_current_record()

    def annotate_no(self):
        """Annotate as NO"""
        self.annotations[self.current_index] = False
        self.save_progress()
        self.current_index += 1
        return self.get_current_record()

    def skip_record(self):
        """Skip current record"""
        self.current_index += 1
        return self.get_current_record()

    def previous_record(self):
        """Previous record"""
        if self.current_index > 0:
            self.current_index -= 1
        return self.get_current_record()

    def next_record(self):
        """Next record"""
        if self.current_index < len(self.all_data) - 1:
            self.current_index += 1
        return self.get_current_record()

    def save_and_finish(self):
        """Save results"""
        if not self.annotations:
            return "⚠️ No annotations to save!"

        # Save annotation results
        yes_records = []
        for idx, keep in self.annotations.items():
            if keep:
                yes_records.append(self.all_data[idx])

        # Write to output file
        try:
            with open(self.output_path, 'w', encoding='utf-8') as f:
                for record in yes_records:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')

            # Save statistics
            stats_path = self.output_path.replace('.jsonl', '_stats.txt')
            with open(stats_path, 'w', encoding='utf-8') as f:
                yes_count = sum(1 for v in self.annotations.values() if v)
                no_count = sum(1 for v in self.annotations.values() if not v)
                total_annotated = len(self.annotations)

                f.write(f"Annotation Statistics\n")
                f.write(f"=" * 50 + "\n")
                f.write(f"Total Records: {len(self.all_data)}\n")
                f.write(f"Annotated: {total_annotated}\n")
                f.write(f"YES (Keep): {yes_count}\n")
                f.write(f"NO (Delete): {no_count}\n")
                f.write(f"Not Annotated: {len(self.all_data) - total_annotated}\n")
                if total_annotated > 0:
                    f.write(f"Keep Rate: {yes_count / total_annotated * 100:.2f}%\n")
                else:
                    f.write(f"Keep Rate: N/A\n")

            # Backup progress file (instead of deleting)
            if os.path.exists(self.progress_path):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = self.progress_path.replace('.json', f'_backup_{timestamp}.json')
                shutil.copy2(self.progress_path, backup_path)
                print(f"Progress file backed up to: {backup_path}")

            yes_count = sum(1 for v in self.annotations.values() if v)
            no_count = sum(1 for v in self.annotations.values() if not v)

            result_text = f"""
✅ Annotation results saved successfully!

**Output File:** {self.output_path}
**Statistics File:** {stats_path}

**Records Kept (YES):** {yes_count}
**Records Deleted (NO):** {no_count}
**Total Annotated:** {len(self.annotations)}

Annotation complete! You can close the browser.
"""
            return result_text

        except Exception as e:
            return f"❌ Save failed: {e}"


def create_ui():
    """Create Gradio interface"""
    tool = WebAnnotationTool()

    with gr.Blocks(title="Visual Data Annotation Tool", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 📝 Visual Data Annotation Tool")
        gr.Markdown("Use **YES** / **NO** buttons to annotate data, click **Save & Finish** to export results")

        with gr.Row():
            progress_display = gr.Markdown(value=tool.get_current_record()[3])

        with gr.Row():
            stats_display = gr.Markdown(value=tool.get_current_record()[4])

        with gr.Row():
            # Left side: Text info
            with gr.Column(scale=1):
                info_display = gr.Markdown(
                    value=tool.get_current_record()[0],
                    label="Data Information"
                )

            # Right side: Image comparison area
            with gr.Column(scale=1):
                with gr.Row():
                    # Original image
                    original_image_display = gr.Image(
                        value=tool.get_current_record()[1],
                        label="Original Image",
                        height=600
                    )
                    # Edited image
                    edited_image_display = gr.Image(
                        value=tool.get_current_record()[2],
                        label="Edited Image",
                        height=600
                    )

        # Control buttons
        with gr.Row():
            prev_btn = gr.Button("⬅️ Previous", variant="secondary")
            yes_btn = gr.Button("✅ YES (Keep)", variant="primary", size="lg")
            no_btn = gr.Button("❌ NO (Delete)", variant="stop", size="lg")
            skip_btn = gr.Button("⏭️ Skip", variant="secondary")

        with gr.Row():
            next_btn = gr.Button("➡️ Next", variant="secondary")
            save_btn = gr.Button("💾 Save & Finish", variant="primary")

        # Save result display
        result_display = gr.Markdown(visible=False)

        # Button events
        def update_yes():
            info, original_img, edited_img, prog, stats = tool.annotate_yes()
            if info is None:
                return {
                    info_display: "✅ All data has been annotated!",
                    original_image_display: None,
                    edited_image_display: None,
                    progress_display: "Complete",
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
                    info_display: "✅ All data has been annotated!",
                    original_image_display: None,
                    edited_image_display: None,
                    progress_display: "Complete",
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
                    info_display: "✅ All data has been annotated!",
                    original_image_display: None,
                    edited_image_display: None,
                    progress_display: "Complete",
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
                    info_display: "✅ This is the last record!",
                    original_image_display: None,
                    edited_image_display: None,
                    progress_display: "Complete",
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
        ### 💡 Instructions
        - **YES**: Keep current record and move to next
        - **NO**: Delete current record and move to next
        - **Skip**: Skip without annotation, view next record
        - **Previous/Next**: Browse and modify annotated records
        - **Save & Finish**: Export all YES records to file

        ### 📁 Output Files
        - `annotated_output.jsonl` - All records marked as YES
        - `annotated_output_stats.txt` - Detailed statistics
        """)

    return app


def main():
    """Main function"""
    print("=" * 60)
    print("Visual Data Annotation Tool - Web Version")
    print("=" * 60)

    app = create_ui()

    print("\n🚀 Starting Web Server...")
    print("\n" + "=" * 60)
    print("📌 Access Methods:")
    print("=" * 60)
    print("1. Local: http://localhost:7860")
    print("2. Remote: Use the public URL shown below")
    print("3. SSH Port Forward: ssh -L 7860:localhost:7860 user@server")
    print("=" * 60)
    print("\nPress Ctrl+C to stop the server\n")

    # Launch Gradio server
    # share=True generates a public URL (temporary, valid for 72 hours)
    app.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=True,  # Generate public URL
        show_error=True,
        allowed_paths=[
            "/gpfs/projects/p32958/chengxuan/results/progresslm/negative/image/",  # Edited images directory
            "/projects/p32958/chengxuan/data/images/"  # Original images directory
        ]
    )


if __name__ == '__main__':
    main()
