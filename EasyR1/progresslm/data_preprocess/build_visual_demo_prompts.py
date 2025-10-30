#!/usr/bin/env python3
"""Convert ProgressLM visual demo JSONL data into EasyR1-ready prompts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _default_progresslm_root(script_path: Path) -> Path | None:
    """Guess the ProgressLM repository location relative to this script."""
    for parent in script_path.parents:
        candidate = parent.parent / "chengxuan" / "ProgressLM"
        if candidate.exists():
            return candidate
    return None


def _add_progresslm_to_syspath(progresslm_root: Path) -> None:
    """Make sure ProgressLM eval helpers are importable."""
    eval_pkg = progresslm_root / "eval" / "qwen25vl"
    if not eval_pkg.exists():
        raise FileNotFoundError(
            f"Could not locate ProgressLM eval package at {eval_pkg}. "
            "Please verify --progresslm-root."
        )
    sys.path.insert(0, str(eval_pkg))


def collect_prompt_segments(
    item: Dict[str, Any],
    *,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> Tuple[str, List[str], List[Dict[str, Any]]]:
    """Build the user prompt string and gather associated image paths."""
    from visual_demo_prompt import VISUAL_DEMO_INSTRUCTION_PART3

    task_goal: str = item["task_goal"]
    visual_demo: List[str] = item["visual_demo"]
    total_steps: int = int(item["total_steps"])
    stage_to_estimate: str = item["stage_to_estimate"]

    if total_steps <= 0 or len(visual_demo) != total_steps + 1:
        raise ValueError(
            f"visual_demo length ({len(visual_demo)}) must equal total_steps+1 ({total_steps + 1})"
        )

    # Evenly spaced percentages from 0% to 100%
    percent_denominator = max(total_steps, 1)
    demo_percentages = [
        round((idx / percent_denominator) * 100) for idx in range(len(visual_demo))
    ]
    demo_sequence = " ".join(
        f"<image> {percentage}%" for percentage in demo_percentages
    )

    prompt_sections = [
        f"Our goal is {task_goal}.",
        f"Here is the demonstration:\n{demo_sequence}",
        "Here is the current state that you need to estimate:\n<image>",
        VISUAL_DEMO_INSTRUCTION_PART3,
    ]
    prompt = "\n\n\n".join(prompt_sections)

    # Images must match the appearance order of <image> tags in the prompt.
    image_paths: List[str] = list(visual_demo) + [stage_to_estimate]

    segments: List[Dict[str, Any]] = [
        {"type": "text", "value": prompt_sections[0]},
        {"type": "text", "value": "Here is the demonstration:"},
    ]
    for path, percentage in zip(visual_demo, demo_percentages):
        image_entry: Dict[str, Any] = {"type": "image", "value": path}
        if min_pixels is not None:
            image_entry["min_pixels"] = min_pixels
        if max_pixels is not None:
            image_entry["max_pixels"] = max_pixels
        segments.append(image_entry)
        segments.append({"type": "text", "value": f"{percentage}%"})

    segments.append({"type": "text", "value": "Here is the current state that you need to estimate:"})
    stage_entry: Dict[str, Any] = {"type": "image", "value": stage_to_estimate}
    if min_pixels is not None:
        stage_entry["min_pixels"] = min_pixels
    if max_pixels is not None:
        stage_entry["max_pixels"] = max_pixels
    segments.append(stage_entry)
    segments.append({"type": "text", "value": VISUAL_DEMO_INSTRUCTION_PART3})

    return prompt, image_paths, segments


def iter_dataset_items(
    dataset_path: Path,
    *,
    num_inferences: int,
    image_root: Path | None,
) -> Iterable[Dict[str, Any]]:
    from visual_demo_dataset import load_visual_demo_dataset

    return load_visual_demo_dataset(
        str(dataset_path),
        num_inferences=num_inferences,
        image_root=str(image_root) if image_root else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ProgressLM visual demo JSONL to EasyR1 prompt format."
    )
    parser.add_argument("--input", type=Path, required=True, help="Source visual demo JSONL file")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSONL file")
    parser.add_argument(
        "--progresslm-root",
        type=Path,
        default=None,
        help="Root directory of the ProgressLM repo (auto-detected when omitted)",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Optional root directory to prepend to relative image paths",
    )
    parser.add_argument(
        "--num-inferences",
        type=int,
        default=1,
        help="Replicate each record this many times (matches ProgressLM loader behaviour)",
    )
    parser.add_argument("--min-pixels", type=int, default=None, help="Optional min pixel constraint")
    parser.add_argument("--max-pixels", type=int, default=None, help="Optional max pixel constraint")
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many records")

    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    progresslm_root = args.progresslm_root or _default_progresslm_root(script_path)
    if progresslm_root is None:
        raise FileNotFoundError(
            "Unable to auto-detect ProgressLM repo location. Provide --progresslm-root explicitly."
        )

    _add_progresslm_to_syspath(progresslm_root)

    from visual_demo_prompt import VISUAL_DEMO_SYSTEM_PROMPT  # noqa: E402

    dataset_iter = iter_dataset_items(
        args.input,
        num_inferences=args.num_inferences,
        image_root=args.image_root,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    with args.output.open("w", encoding="utf-8") as fout:
        for item in dataset_iter:
            prompt, image_paths, segments = collect_prompt_segments(
                item,
                min_pixels=args.min_pixels,
                max_pixels=args.max_pixels,
            )

            progress_fraction = float(item.get("progress_score"))
            progress_percent = round(progress_fraction * 100)
            progress_score_str = f"{progress_percent}%"

            closest_idx = int(item.get("closest_idx"))
            delta = item.get("delta")
            demo_count = len(item["visual_demo"])

            canonical_response = (
                "<ref_think>Reasoning traces are not available in the source annotations; "
                "use the provided reference frame as guidance.</ref_think>\n"
                f"<ref>{closest_idx}</ref>\n"
                f"<score_think>Ground-truth progress label is {progress_score_str}"
            )
            if delta is not None:
                canonical_response += f" (delta {delta})"
            canonical_response += ".</score_think>\n"
            canonical_response += f"<score>{progress_score_str}</score>"

            answer = {
                "ref": closest_idx,
                "score_fraction": progress_fraction,
                "score_percent": progress_percent,
                "score_text": progress_score_str,
                "demo_count": demo_count,
                "delta": delta,
                "target_response": canonical_response,
            }

            record = {
                "id": item["id"],
                "task_goal": item["task_goal"],
                "system_prompt": VISUAL_DEMO_SYSTEM_PROMPT,
                "user_prompt": prompt,
                "answer": answer,
                "images": image_paths,
                "messages": segments,
                "metadata": {
                    "closest_idx": item.get("closest_idx"),
                    "progress_score": item.get("progress_score"),
                    "stage_to_estimate": item.get("stage_to_estimate"),
                },
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            processed += 1

            if args.limit is not None and processed >= args.limit:
                break

    print(f"Wrote {processed} samples to {args.output}")


if __name__ == "__main__":
    main()
