#!/usr/bin/env python3
"""
Visual Demo Visualization Tool

A Gradio-based visualization tool for browsing JSONL result files with visual demos.
Features:
- Display all visual_demo images in order (gallery view)
- Display stage_to_estimate image (current state)
- Filter by visual demo count (number of demo images)
- Filtering by data source and status
- Slider navigation with prev/next buttons
"""

import json
import gradio as gr
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


class VisualDemoViewer:
    """Manages the loading, filtering, and display of JSONL results with visual demos."""

    def __init__(self):
        self.all_data: List[Dict[str, Any]] = []
        self.filtered_data: List[Dict[str, Any]] = []
        self.data_sources: List[str] = []
        self.statuses: List[str] = []
        self.demo_counts: List[str] = []  # Unique visual_demo counts

    def load_jsonl(self, file_path: str) -> Tuple[str, List[str], List[str], List[str], int, str]:
        """Load a JSONL file and return metadata for UI updates."""
        self.all_data = []
        self.filtered_data = []

        if not file_path or not file_path.strip():
            return "Please enter a file path", ["All"], ["All"], ["All"], 0, ""

        path = Path(file_path.strip())
        if not path.exists():
            return f"File not found: {file_path}", ["All"], ["All"], ["All"], 0, ""

        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.all_data.append(json.loads(line))
        except Exception as e:
            return f"Error loading file: {str(e)}", ["All"], ["All"], ["All"], 0, ""

        # Extract unique data sources, statuses, and demo counts
        sources = set()
        statuses = set()
        demo_counts = set()
        for item in self.all_data:
            meta = item.get("meta_data", {})
            if "data_source" in meta:
                sources.add(meta["data_source"])
            if "status" in meta:
                statuses.add(meta["status"])
            # Extract visual_demo count
            visual_demo = meta.get("visual_demo", [])
            demo_counts.add(len(visual_demo))

        self.data_sources = ["All"] + sorted(list(sources))
        self.statuses = ["All"] + sorted(list(statuses))
        self.demo_counts = ["All"] + [str(c) for c in sorted(demo_counts)]
        self.filtered_data = self.all_data.copy()

        status_msg = f"Loaded {len(self.all_data)} cases from {path.name}"
        return status_msg, self.data_sources, self.statuses, self.demo_counts, max(0, len(self.filtered_data) - 1), f"Total: {len(self.all_data)} | Filtered: {len(self.filtered_data)}"

    def apply_filters(
        self,
        source_filter: str,
        status_filter: str,
        demo_count_filter: str,
        score_correct_only: bool,
        ref_correct_only: bool
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

            # Demo count filter
            if demo_count_filter != "All":
                visual_demo = meta.get("visual_demo", [])
                if len(visual_demo) != int(demo_count_filter):
                    continue

            # Score correctness filter
            if score_correct_only:
                if item.get("score") != item.get("ground_truth_score"):
                    continue

            # Ref correctness filter
            if ref_correct_only:
                if item.get("ref") != item.get("closest_idx"):
                    continue

            self.filtered_data.append(item)

        max_idx = max(0, len(self.filtered_data) - 1)
        count_text = f"Total: {len(self.all_data)} | Filtered: {len(self.filtered_data)}"
        return max_idx, count_text

    def get_case(self, index: int) -> Tuple[List[Tuple[str, str]], Optional[str], str, str, str, str]:
        """Get a specific case by index.

        Returns:
            - visual_demo_gallery: List of (image_path, label) tuples for gallery
            - stage_to_estimate_image: Single image path
            - task_goal: Task goal text
            - response: Model response
            - metadata: Formatted metadata string
            - index_display: Current case index display
        """
        if not self.filtered_data or index < 0 or index >= len(self.filtered_data):
            return [], None, "No data", "No data", "No data", "Case 0 / 0"

        item = self.filtered_data[index]
        meta = item.get("meta_data", {})

        # Visual demo images - create gallery items
        visual_demo_list = meta.get("visual_demo", [])
        visual_demo_gallery = []
        for i, img_path in enumerate(visual_demo_list):
            if img_path and Path(img_path).exists():
                visual_demo_gallery.append((img_path, f"Demo {i+1}"))

        # Stage to estimate image
        stage_image_path = meta.get("stage_to_estimate", "")
        if stage_image_path and Path(stage_image_path).exists():
            stage_image = stage_image_path
        else:
            stage_image = None

        # Task goal
        task_goal = meta.get("task_goal", "N/A")

        # Response
        response = item.get("response", "N/A")

        # Metadata
        metadata_lines = [
            f"**ID:** {meta.get('id', 'N/A')}",
            f"**Data Source:** {meta.get('data_source', 'N/A')}",
            f"**Status:** {meta.get('status', 'N/A')}",
            "",
            f"**Score:** {item.get('score', 'N/A')} | **Ground Truth:** {item.get('ground_truth_score', 'N/A')}",
            f"**Ref:** {item.get('ref', 'N/A')} | **Closest Idx:** {item.get('closest_idx', 'N/A')}",
            f"**Ref Score:** {item.get('ref_score', 'N/A')} | **Pred Score:** {item.get('pred_score', 'N/A')}",
            "",
            f"**Progress Score:** {meta.get('progress_score', 'N/A')}",
            f"**Total Steps:** {meta.get('total_steps', 'N/A')}",
            f"**Delta:** {meta.get('delta', 'N/A')}",
            "",
            f"**Ref False Positive:** {item.get('ref_false_positive', 'N/A')}",
            f"**Score False Positive:** {item.get('score_false_positive', 'N/A')}",
        ]
        metadata = "\n".join(metadata_lines)

        # Index display
        index_display = f"Case {index + 1} / {len(self.filtered_data)}"

        return visual_demo_gallery, stage_image, task_goal, response, metadata, index_display


# Global viewer instance
viewer = VisualDemoViewer()


def load_file(file_path: str):
    """Load file and update UI components."""
    status, sources, statuses, demo_counts, max_idx, count_text = viewer.load_jsonl(file_path)

    # Get first case
    if viewer.filtered_data:
        gallery, stage_img, task_goal, response, metadata, idx_display = viewer.get_case(0)
    else:
        gallery, stage_img, task_goal, response, metadata, idx_display = [], None, "", "", "", "Case 0 / 0"

    return (
        status,  # status_text
        gr.update(choices=sources, value="All"),  # source_dropdown
        gr.update(choices=statuses, value="All"),  # status_dropdown
        gr.update(choices=demo_counts, value="All"),  # demo_count_dropdown
        gr.update(maximum=max_idx, value=0),  # slider
        count_text,  # count_text
        gallery,  # visual_demo_gallery
        stage_img,  # stage_image
        task_goal,  # task_goal
        response,  # response
        metadata,  # metadata
        idx_display,  # index_display
    )


def filter_data(source_filter: str, status_filter: str, demo_count_filter: str, score_correct_only: bool, ref_correct_only: bool, current_idx: int):
    """Apply filters and update display."""
    max_idx, count_text = viewer.apply_filters(source_filter, status_filter, demo_count_filter, score_correct_only, ref_correct_only)

    # Reset to first case
    if viewer.filtered_data:
        gallery, stage_img, task_goal, response, metadata, idx_display = viewer.get_case(0)
    else:
        gallery, stage_img, task_goal, response, metadata, idx_display = [], None, "", "", "", "Case 0 / 0"

    return (
        gr.update(maximum=max_idx, value=0),  # slider
        count_text,  # count_text
        gallery,  # visual_demo_gallery
        stage_img,  # stage_image
        task_goal,  # task_goal
        response,  # response
        metadata,  # metadata
        idx_display,  # index_display
    )


def update_display(index: int):
    """Update display for a specific case index."""
    gallery, stage_img, task_goal, response, metadata, idx_display = viewer.get_case(int(index))
    return gallery, stage_img, task_goal, response, metadata, idx_display


def go_prev(current_idx: int):
    """Go to previous case."""
    new_idx = max(0, int(current_idx) - 1)
    gallery, stage_img, task_goal, response, metadata, idx_display = viewer.get_case(new_idx)
    return new_idx, gallery, stage_img, task_goal, response, metadata, idx_display


def go_next(current_idx: int):
    """Go to next case."""
    max_idx = len(viewer.filtered_data) - 1
    new_idx = min(max_idx, int(current_idx) + 1)
    gallery, stage_img, task_goal, response, metadata, idx_display = viewer.get_case(new_idx)
    return new_idx, gallery, stage_img, task_goal, response, metadata, idx_display


def create_ui():
    """Create the Gradio UI."""
    with gr.Blocks(title="Visual Demo Viewer", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Visual Demo Results Viewer")
        gr.Markdown("View visual demo sequences and stage estimation images from JSONL results.")

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
            demo_count_dropdown = gr.Dropdown(
                choices=["All"],
                value="All",
                label="Visual Demo Count",
                scale=1
            )
            score_correct_checkbox = gr.Checkbox(
                label="Score Correct Only",
                value=False,
                scale=1
            )
            ref_correct_checkbox = gr.Checkbox(
                label="Ref Correct Only",
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

        # Task goal
        task_goal_text = gr.Textbox(label="Task Goal", lines=2, interactive=False)

        # Visual Demo Gallery section
        gr.Markdown("### Visual Demo Sequence")
        visual_demo_gallery = gr.Gallery(
            label="Visual Demo Images (in order)",
            show_label=True,
            columns=4,
            rows=2,
            height="auto",
            object_fit="contain"
        )

        # Stage to Estimate section
        gr.Markdown("### Stage to Estimate (Current State)")
        with gr.Row():
            with gr.Column(scale=1):
                stage_image = gr.Image(
                    label="Current State Image",
                    type="filepath",
                    height=400
                )
            with gr.Column(scale=1):
                metadata_text = gr.Markdown(label="Metadata")

        # Response section
        response_text = gr.Textbox(label="Model Response", lines=8, interactive=False)

        # Event handlers
        load_btn.click(
            fn=load_file,
            inputs=[file_path_input],
            outputs=[
                status_text,
                source_dropdown,
                status_dropdown,
                demo_count_dropdown,
                case_slider,
                count_text,
                visual_demo_gallery,
                stage_image,
                task_goal_text,
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
                demo_count_dropdown,
                case_slider,
                count_text,
                visual_demo_gallery,
                stage_image,
                task_goal_text,
                response_text,
                metadata_text,
                index_display,
            ]
        )

        # Filter change handlers
        for filter_component in [source_dropdown, status_dropdown, demo_count_dropdown, score_correct_checkbox, ref_correct_checkbox]:
            filter_component.change(
                fn=filter_data,
                inputs=[source_dropdown, status_dropdown, demo_count_dropdown, score_correct_checkbox, ref_correct_checkbox, case_slider],
                outputs=[
                    case_slider,
                    count_text,
                    visual_demo_gallery,
                    stage_image,
                    task_goal_text,
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
                visual_demo_gallery,
                stage_image,
                task_goal_text,
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
                visual_demo_gallery,
                stage_image,
                task_goal_text,
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
                visual_demo_gallery,
                stage_image,
                task_goal_text,
                response_text,
                metadata_text,
                index_display,
            ]
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(share=True, server_name="0.0.0.0")
