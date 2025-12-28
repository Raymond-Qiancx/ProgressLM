#!/usr/bin/env python3
"""
JSONL Results Visualization Tool

A Gradio-based visualization tool for browsing JSONL result files.
Features:
- Image display from stage_to_estimate path
- Task goal, text demo, and model response display
- Filtering by data source, status, and score range
- Slider navigation with prev/next buttons
"""

import json
import gradio as gr
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


class ResultsViewer:
    """Manages the loading, filtering, and display of JSONL results."""

    def __init__(self):
        self.all_data: List[Dict[str, Any]] = []
        self.filtered_data: List[Dict[str, Any]] = []
        self.data_sources: List[str] = []
        self.statuses: List[str] = []

    def load_jsonl(self, file_path: str) -> Tuple[str, List[str], List[str], int, str]:
        """Load a JSONL file and return metadata for UI updates."""
        self.all_data = []
        self.filtered_data = []

        if not file_path or not file_path.strip():
            return "Please enter a file path", ["All"], ["All"], 0, ""

        path = Path(file_path.strip())
        if not path.exists():
            return f"File not found: {file_path}", ["All"], ["All"], 0, ""

        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.all_data.append(json.loads(line))
        except Exception as e:
            return f"Error loading file: {str(e)}", ["All"], ["All"], 0, ""

        # Extract unique data sources and statuses
        sources = set()
        statuses = set()
        for item in self.all_data:
            meta = item.get("meta_data", {})
            if "data_source" in meta:
                sources.add(meta["data_source"])
            if "status" in meta:
                statuses.add(meta["status"])

        self.data_sources = ["All"] + sorted(list(sources))
        self.statuses = ["All"] + sorted(list(statuses))
        self.filtered_data = self.all_data.copy()

        status_msg = f"Loaded {len(self.all_data)} cases from {path.name}"
        return status_msg, self.data_sources, self.statuses, len(self.filtered_data) - 1, f"Total: {len(self.all_data)} | Filtered: {len(self.filtered_data)}"

    def apply_filters(
        self,
        source_filter: str,
        status_filter: str,
        score_correct_only: bool
    ) -> Tuple[int, str]:
        """Apply filters and return updated slider max and count text."""
        self.filtered_data = []

        for item in self.all_data:
            meta = item.get("meta_data", {})

            # Source filter
            if source_filter != "All" and meta.get("data_source") != source_filter:
                continue

            # Status filter
            if status_filter != "All" and meta.get("status") != status_filter:
                continue

            # Score correctness filter
            if score_correct_only:
                if item.get("score") != item.get("ground_truth_score"):
                    continue

            self.filtered_data.append(item)

        max_idx = max(0, len(self.filtered_data) - 1)
        count_text = f"Total: {len(self.all_data)} | Filtered: {len(self.filtered_data)}"
        return max_idx, count_text

    def get_case(self, index: int) -> Tuple[Optional[str], str, str, str, str, str]:
        """Get a specific case by index."""
        if not self.filtered_data or index < 0 or index >= len(self.filtered_data):
            return None, "No data", "No data", "No data", "No data", "Case 0 / 0"

        item = self.filtered_data[index]
        meta = item.get("meta_data", {})

        # Image path
        image_path = meta.get("stage_to_estimate", "")
        if image_path and Path(image_path).exists():
            image = image_path
        else:
            image = None

        # Task goal
        task_goal = meta.get("task_goal", "N/A")

        # Text demo - format as numbered list
        text_demo_list = meta.get("text_demo", [])
        if isinstance(text_demo_list, list):
            text_demo = "\n".join([f"{i+1}. {step}" for i, step in enumerate(text_demo_list)])
        else:
            text_demo = str(text_demo_list)

        # Response
        response = item.get("response", "N/A")

        # Metadata
        metadata_lines = [
            f"**ID:** {meta.get('id', 'N/A')}",
            f"**Data Source:** {meta.get('data_source', 'N/A')}",
            f"**Status:** {meta.get('status', 'N/A')}",
            "",
            f"**Score:** {item.get('score', 'N/A')} | **Ground Truth:** {item.get('ground_truth_score', 'N/A')}",
            f"**Ref Score:** {item.get('ref_score', 'N/A')} | **Pred Score:** {item.get('pred_score', 'N/A')}",
            "",
            f"**Progress Score:** {meta.get('progress_score', 'N/A')}",
            f"**Closest Idx:** {item.get('closest_idx', 'N/A')} | **Total Steps:** {meta.get('total_steps', 'N/A')}",
            "",
            f"**Ref False Positive:** {item.get('ref_false_positive', 'N/A')}",
            f"**Score False Positive:** {item.get('score_false_positive', 'N/A')}",
        ]
        metadata = "\n".join(metadata_lines)

        # Index display
        index_display = f"Case {index + 1} / {len(self.filtered_data)}"

        return image, task_goal, text_demo, response, metadata, index_display


# Global viewer instance
viewer = ResultsViewer()


def load_file(file_path: str):
    """Load file and update UI components."""
    status, sources, statuses, max_idx, count_text = viewer.load_jsonl(file_path)

    # Get first case
    if viewer.filtered_data:
        image, task_goal, text_demo, response, metadata, idx_display = viewer.get_case(0)
    else:
        image, task_goal, text_demo, response, metadata, idx_display = None, "", "", "", "", "Case 0 / 0"

    return (
        status,  # status_text
        gr.update(choices=sources, value="All"),  # source_dropdown
        gr.update(choices=statuses, value="All"),  # status_dropdown
        gr.update(maximum=max_idx, value=0),  # slider
        count_text,  # count_text
        image,  # image
        task_goal,  # task_goal
        text_demo,  # text_demo
        response,  # response
        metadata,  # metadata
        idx_display,  # index_display
    )


def filter_data(source_filter: str, status_filter: str, score_correct_only: bool, current_idx: int):
    """Apply filters and update display."""
    max_idx, count_text = viewer.apply_filters(source_filter, status_filter, score_correct_only)

    # Reset to first case
    new_idx = 0
    if viewer.filtered_data:
        image, task_goal, text_demo, response, metadata, idx_display = viewer.get_case(0)
    else:
        image, task_goal, text_demo, response, metadata, idx_display = None, "", "", "", "", "Case 0 / 0"

    return (
        gr.update(maximum=max_idx, value=new_idx),  # slider
        count_text,  # count_text
        image,  # image
        task_goal,  # task_goal
        text_demo,  # text_demo
        response,  # response
        metadata,  # metadata
        idx_display,  # index_display
    )


def update_display(index: int):
    """Update display for a specific case index."""
    image, task_goal, text_demo, response, metadata, idx_display = viewer.get_case(int(index))
    return image, task_goal, text_demo, response, metadata, idx_display


def go_prev(current_idx: int):
    """Go to previous case."""
    new_idx = max(0, int(current_idx) - 1)
    image, task_goal, text_demo, response, metadata, idx_display = viewer.get_case(new_idx)
    return new_idx, image, task_goal, text_demo, response, metadata, idx_display


def go_next(current_idx: int):
    """Go to next case."""
    max_idx = len(viewer.filtered_data) - 1
    new_idx = min(max_idx, int(current_idx) + 1)
    image, task_goal, text_demo, response, metadata, idx_display = viewer.get_case(new_idx)
    return new_idx, image, task_goal, text_demo, response, metadata, idx_display


def create_ui():
    """Create the Gradio UI."""
    with gr.Blocks(title="JSONL Results Viewer", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# JSONL Results Viewer")

        # File input section
        with gr.Row():
            file_path_input = gr.Textbox(
                label="JSONL File Path",
                placeholder="/path/to/results.jsonl",
                scale=4
            )
            load_btn = gr.Button("Load", variant="primary", scale=1)

        status_text = gr.Textbox(label="Status", interactive=False)

        # Filters section
        with gr.Row():
            source_dropdown = gr.Dropdown(
                choices=["All"],
                value="All",
                label="Data Source",
                scale=1
            )
            status_dropdown = gr.Dropdown(
                choices=["All"],
                value="All",
                label="Status",
                scale=1
            )
            score_correct_checkbox = gr.Checkbox(
                label="Score Correct Only",
                value=False,
                scale=1
            )

        count_text = gr.Textbox(label="Case Count", interactive=False)

        # Navigation section
        with gr.Row():
            case_slider = gr.Slider(
                minimum=0,
                maximum=0,
                step=1,
                value=0,
                label="Case Index",
                scale=4
            )
            prev_btn = gr.Button("Prev", scale=1)
            next_btn = gr.Button("Next", scale=1)

        index_display = gr.Textbox(label="Current Case", interactive=False)

        # Content display section
        with gr.Row():
            with gr.Column(scale=1):
                image_display = gr.Image(label="Stage Image", type="filepath")

            with gr.Column(scale=1):
                task_goal_text = gr.Textbox(label="Task Goal", lines=2, interactive=False)
                text_demo_text = gr.Textbox(label="Text Demo", lines=8, interactive=False)

        with gr.Row():
            with gr.Column(scale=1):
                response_text = gr.Textbox(label="Model Response", lines=12, interactive=False)

            with gr.Column(scale=1):
                metadata_text = gr.Markdown(label="Metadata")

        # Event handlers
        load_btn.click(
            fn=load_file,
            inputs=[file_path_input],
            outputs=[
                status_text,
                source_dropdown,
                status_dropdown,
                case_slider,
                count_text,
                image_display,
                task_goal_text,
                text_demo_text,
                response_text,
                metadata_text,
                index_display,
            ]
        )

        # Also load on Enter key
        file_path_input.submit(
            fn=load_file,
            inputs=[file_path_input],
            outputs=[
                status_text,
                source_dropdown,
                status_dropdown,
                case_slider,
                count_text,
                image_display,
                task_goal_text,
                text_demo_text,
                response_text,
                metadata_text,
                index_display,
            ]
        )

        # Filter change handlers
        for filter_component in [source_dropdown, status_dropdown, score_correct_checkbox]:
            filter_component.change(
                fn=filter_data,
                inputs=[source_dropdown, status_dropdown, score_correct_checkbox, case_slider],
                outputs=[
                    case_slider,
                    count_text,
                    image_display,
                    task_goal_text,
                    text_demo_text,
                    response_text,
                    metadata_text,
                    index_display,
                ]
            )

        # Slider change handler
        case_slider.change(
            fn=update_display,
            inputs=[case_slider],
            outputs=[
                image_display,
                task_goal_text,
                text_demo_text,
                response_text,
                metadata_text,
                index_display,
            ]
        )

        # Navigation button handlers
        prev_btn.click(
            fn=go_prev,
            inputs=[case_slider],
            outputs=[
                case_slider,
                image_display,
                task_goal_text,
                text_demo_text,
                response_text,
                metadata_text,
                index_display,
            ]
        )

        next_btn.click(
            fn=go_next,
            inputs=[case_slider],
            outputs=[
                case_slider,
                image_display,
                task_goal_text,
                text_demo_text,
                response_text,
                metadata_text,
                index_display,
            ]
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(share=True, server_name="0.0.0.0")
