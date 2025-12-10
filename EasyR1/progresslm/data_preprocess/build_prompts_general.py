#!/usr/bin/env python3
"""Convert ProgressLM demo JSONL data into EasyR1-ready prompts (visual or text)."""

from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Tuple

DatasetType = Literal["visual", "text"]


def _resolve_image_path(
    image_root: Path | None,
    sample_id: str,
    image_path: str,
) -> str:
    """Return a path that includes the image root and sample id when needed."""
    if image_root is None or not image_path:
        return image_path

    path_obj = Path(image_path)
    if path_obj.is_absolute():
        return str(path_obj)

    return str(image_root / Path(sample_id) / path_obj)


def _resolve_image_list(
    image_root: Path | None,
    sample_id: str,
    image_paths: Iterable[str],
) -> List[str]:
    """Expand a sequence of image paths to include the image root and sample id."""
    return [
        _resolve_image_path(image_root, sample_id, img_path) for img_path in image_paths
    ]


def _default_progresslm_root(script_path: Path) -> Path | None:
    """Guess the ProgressLM repository location relative to this script."""
    for parent in script_path.parents:
        candidate = parent.parent / "chengxuan" / "ProgressLM"
        if candidate.exists():
            return candidate
    return None


def _add_progresslm_to_syspath(progresslm_root: Path) -> None:
    """Ensure ProgressLM eval helpers are importable."""
    eval_pkg = progresslm_root / "eval" / "qwen25vl"
    if not eval_pkg.exists():
        raise FileNotFoundError(
            f"Could not locate ProgressLM eval package at {eval_pkg}. "
            "Please verify --progresslm-root."
        )
    sys.path.insert(0, str(eval_pkg))


def _detect_dataset_type(dataset_path: Path) -> DatasetType:
    """Inspect the first non-empty JSON line to infer dataset modality."""
    with dataset_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            if "visual_demo" in sample:
                return "visual"
            if "text_demo" in sample:
                return "text"
            break
    raise ValueError(
        "Unable to infer dataset type from the input file. "
        "Please pass --dataset-type explicitly."
    )


def collect_visual_prompt_segments(
    item: Dict[str, Any],
    *,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> Tuple[str, List[str], List[Dict[str, Any]]]:
    """Build the user prompt and segments for a visual demo sample."""
    from visual_demo_prompt import VISUAL_DEMO_SYSTEM_PROMPT, VISUAL_DEMO_INSTRUCTION_PART3

    task_goal: str = item["task_goal"]
    visual_demo: List[str] = item["visual_demo"]
    total_steps: int = int(item["total_steps"])
    stage_to_estimate: str = item["stage_to_estimate"]

    if total_steps <= 0 or len(visual_demo) != total_steps + 1:
        raise ValueError(
            f"visual_demo length ({len(visual_demo)}) must equal total_steps+1 ({total_steps + 1})"
        )

    percent_denominator = max(total_steps, 1)
    demo_percentages = [
        round((idx / percent_denominator) * 100) for idx in range(len(visual_demo))
    ]
    demo_sequence = " ".join(
        f"<image> {percentage}%" for percentage in demo_percentages
    )

    prompt_sections = [
        VISUAL_DEMO_SYSTEM_PROMPT,
        f"The overall task goal is {task_goal}.",
        f"Here is the demonstration:\n{demo_sequence}",
        "Here is the current state that you need to estimate:\n<image>",
        VISUAL_DEMO_INSTRUCTION_PART3,
    ]
    prompt = "\n\n\n".join(prompt_sections)

    image_paths: List[str] = list(visual_demo) + [stage_to_estimate]

    segments: List[Dict[str, Any]] = [
        {"type": "text", "value": VISUAL_DEMO_SYSTEM_PROMPT},
        {"type": "text", "value": f"The overall task goal is {task_goal}."},
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

    segments.append(
        {"type": "text", "value": "Here is the current state that you need to estimate:"}
    )
    stage_entry: Dict[str, Any] = {"type": "image", "value": stage_to_estimate}
    if min_pixels is not None:
        stage_entry["min_pixels"] = min_pixels
    if max_pixels is not None:
        stage_entry["max_pixels"] = max_pixels
    segments.append(stage_entry)
    segments.append({"type": "text", "value": VISUAL_DEMO_INSTRUCTION_PART3})

    return prompt, image_paths, segments


def collect_text_prompt_segments(
    item: Dict[str, Any],
    *,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> Tuple[str, List[str], List[Dict[str, Any]]]:
    """Build the user prompt and segments for a text demo sample."""
    from text_demo_prompt import (
        TEXT_DEMO_SYSTEM_PROMPT,
        TEXT_DEMO_INSTRUCTION_PART1,
        TEXT_DEMO_INSTRUCTION_PART2,
        TEXT_DEMO_INSTRUCTION_PART3,
        build_text_demo_prompt_from_item,
    )

    task_goal: str = item["task_goal"]
    text_demo: List[str] = item["text_demo"]
    total_steps: int = int(item["total_steps"])
    stage_to_estimate: str = item["stage_to_estimate"]

    if total_steps <= 0 or len(text_demo) != total_steps:
        raise ValueError(
            f"text_demo length ({len(text_demo)}) must equal total_steps ({total_steps})"
        )

    demo_percentages = [
        round((idx / total_steps) * 100) for idx in range(1, len(text_demo) + 1)
    ]
    demo_sequence = " ".join(
        f"{step.strip()} {percentage}%"
        for step, percentage in zip(text_demo, demo_percentages)
    )

    prompt_sections = [
        TEXT_DEMO_SYSTEM_PROMPT,
        f"The overall task goal is {task_goal}.",
        f"{TEXT_DEMO_INSTRUCTION_PART1}\n{demo_sequence}",
        f"{TEXT_DEMO_INSTRUCTION_PART2}\n<image>",
        TEXT_DEMO_INSTRUCTION_PART3,
    ]
    prompt = "\n\n\n".join(prompt_sections)

    stage_entry: Dict[str, Any] = {"type": "image", "value": stage_to_estimate}
    if min_pixels is not None:
        stage_entry["min_pixels"] = min_pixels
    if max_pixels is not None:
        stage_entry["max_pixels"] = max_pixels

    segments = build_text_demo_prompt_from_item(
        item,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    return prompt, [stage_to_estimate], segments


def build_answer_payload(
    item: Dict[str, Any],
    *,
    dataset_type: DatasetType,
) -> Dict[str, Any]:
    progress_score = item.get("progress_score")
    closest_idx = item.get("closest_idx")
    delta = item.get("delta")
    demo_key = "visual_demo" if dataset_type == "visual" else "text_demo"
    demo_count = len(item.get(demo_key, []))

    # Handle "n/a" cases
    if progress_score == "n/a" or closest_idx == "n/a":
        canonical_response = (
            "<ref_think>This is an abnormal situation where the current state does not match "
            "the task goal or visual demonstration, or the operation has failed.</ref_think>\n"
            "<ref>n/a</ref>\n"
            "<score_think>Cannot provide valid progress estimation due to abnormal situation.</score_think>\n"
            "<score>n/a</score>"
        )
        return {
            "ref": "n/a",
            "score_fraction": "n/a",
            "score_percent": "n/a",
            "score_text": "n/a",
            "demo_count": demo_count,
            "delta": delta if (delta and delta != "n/a") else "n/a",  # Handle None/empty/n/a
            "target_response": canonical_response,
        }

    # Normal case with valid scores
    progress_fraction = float(progress_score)
    progress_percent = round(progress_fraction * 100)
    progress_score_str = f"{progress_percent}%"
    closest_idx_int = int(closest_idx)

    canonical_response = (
        "<ref_think>Reasoning traces are not available in the source annotations; "
        "use the provided reference step as guidance.</ref_think>\n"
        f"<ref>{closest_idx_int}</ref>\n"
        f"<score_think>Ground-truth progress label is {progress_score_str}"
    )
    if delta is not None and delta != "n/a":
        canonical_response += f" (delta {delta})"
    canonical_response += ".</score_think>\n"
    canonical_response += f"<score>{progress_score_str}</score>"

    return {
        "ref": str(closest_idx_int),  # Convert to string for PyArrow compatibility
        "score_fraction": str(progress_fraction),  # Convert to string for PyArrow compatibility
        "score_percent": str(progress_percent),  # Convert to string for PyArrow compatibility
        "score_text": progress_score_str,
        "demo_count": demo_count,
        "delta": delta if delta else "n/a",  # Handle None/empty delta
        "target_response": canonical_response,
    }


def _normalize_visual_item(
    raw_item: Dict[str, Any],
    *,
    line_num: int,
    image_root: Path | None,
) -> Dict[str, Any] | None:
    from visual_demo_dataset import parse_percentage

    required_fields = [
        "id",
        "task_goal",
        "visual_demo",
        "stage_to_estimate",
        "progress_score",
        "closest_idx",
        "total_steps",
    ]
    missing_fields = [field for field in required_fields if field not in raw_item]
    if missing_fields:
        print(f"Warning: Line {line_num} missing fields {missing_fields}, skipping")
        return None

    if not isinstance(raw_item["task_goal"], str) or not raw_item["task_goal"].strip():
        print(f"Warning: Line {line_num} has invalid task_goal (must be non-empty string), skipping")
        return None

    visual_demo = raw_item["visual_demo"]
    if not isinstance(visual_demo, list) or not visual_demo:
        print(f"Warning: Line {line_num} has invalid visual_demo (must be non-empty list), skipping")
        return None

    try:
        total_steps = int(raw_item["total_steps"])
        if total_steps + 1 != len(visual_demo):
            print(
                f"Warning: Line {line_num} total_steps+1 ({total_steps + 1}) "
                f"doesn't match visual_demo length ({len(visual_demo)}), skipping"
            )
            return None
    except (TypeError, ValueError):
        print(f"Warning: Line {line_num} has invalid total_steps (must be integer), skipping")
        return None

    # Allow closest_idx to be "n/a" or an integer
    closest_idx_raw = raw_item["closest_idx"]
    if closest_idx_raw == "n/a":
        closest_idx = "n/a"
    else:
        try:
            closest_idx = int(closest_idx_raw)
            if not (1 <= closest_idx <= total_steps + 1):
                print(
                    f"Warning: Line {line_num} has invalid closest_idx ({closest_idx}), "
                    f"must be 1-{total_steps + 1} or 'n/a', skipping"
                )
                return None
        except (TypeError, ValueError):
            print(f"Warning: Line {line_num} has invalid closest_idx (must be integer or 'n/a'), skipping")
            return None

    stage_to_estimate = raw_item["stage_to_estimate"]
    if isinstance(stage_to_estimate, list) and len(stage_to_estimate) == 1:
        stage_to_estimate = stage_to_estimate[0]
    elif not isinstance(stage_to_estimate, str):
        print(
            f"Warning: Line {line_num} has invalid stage_to_estimate "
            "(must be string or list with 1 element), skipping"
        )
        return None

    # Allow progress_score to be "n/a" or a percentage
    progress_score_raw = raw_item["progress_score"]
    if progress_score_raw == "n/a":
        progress_score = "n/a"
    else:
        try:
            progress_score = parse_percentage(progress_score_raw)
        except (TypeError, ValueError) as exc:
            print(
                f"Warning: Line {line_num} has invalid progress_score "
                f"({raw_item.get('progress_score')}): {exc}, skipping"
            )
            return None

    normalized = dict(raw_item)
    normalized["total_steps"] = total_steps
    normalized["closest_idx"] = closest_idx
    normalized["stage_to_estimate"] = stage_to_estimate
    normalized["progress_score"] = progress_score

    if image_root:
        image_root_str = str(image_root)
        normalized["visual_demo"] = [
            os.path.join(image_root_str, normalized["id"], img)
            if not os.path.isabs(img)
            else img
            for img in visual_demo
        ]
        if not os.path.isabs(stage_to_estimate):
            normalized["stage_to_estimate"] = os.path.join(
                image_root_str, normalized["id"], stage_to_estimate
            )

    return normalized


def _normalize_text_item(
    raw_item: Dict[str, Any],
    *,
    line_num: int,
    image_root: Path | None,
) -> Dict[str, Any] | None:
    from text_demo_dataset import parse_percentage

    required_fields = [
        "id",
        "task_goal",
        "text_demo",
        "stage_to_estimate",
        "progress_score",
        "closest_idx",
        "total_steps",
    ]
    missing_fields = [field for field in required_fields if field not in raw_item]
    if missing_fields:
        print(f"Warning: Line {line_num} missing fields {missing_fields}, skipping")
        return None

    text_demo = raw_item["text_demo"]
    if not isinstance(text_demo, list) or not text_demo:
        print(f"Warning: Line {line_num} has invalid text_demo (must be non-empty list), skipping")
        return None

    if not isinstance(raw_item["task_goal"], str) or not raw_item["task_goal"].strip():
        print(f"Warning: Line {line_num} has invalid task_goal (must be non-empty string), skipping")
        return None

    try:
        total_steps = int(raw_item["total_steps"])
        if total_steps != len(text_demo):
            print(
                f"Warning: Line {line_num} total_steps ({total_steps}) "
                f"doesn't match text_demo length ({len(text_demo)}), skipping"
            )
            return None
    except (TypeError, ValueError):
        print(f"Warning: Line {line_num} has invalid total_steps (must be integer), skipping")
        return None

    # Allow closest_idx to be "n/a" or an integer
    closest_idx_raw = raw_item["closest_idx"]
    if closest_idx_raw == "n/a":
        closest_idx = "n/a"
    else:
        try:
            closest_idx = int(closest_idx_raw)
            if not (1 <= closest_idx <= len(text_demo)):
                print(
                    f"Warning: Line {line_num} has invalid closest_idx ({closest_idx}), "
                    f"must be 1-{len(text_demo)} or 'n/a', skipping"
                )
                return None
        except (TypeError, ValueError):
            print(f"Warning: Line {line_num} has invalid closest_idx (must be integer or 'n/a'), skipping")
            return None

    stage_to_estimate = raw_item["stage_to_estimate"]
    if isinstance(stage_to_estimate, list) and len(stage_to_estimate) == 1:
        stage_to_estimate = stage_to_estimate[0]
    elif not isinstance(stage_to_estimate, str):
        print(
            f"Warning: Line {line_num} has invalid stage_to_estimate "
            "(must be string or list with 1 element), skipping"
        )
        return None

    # Allow progress_score to be "n/a" or a percentage
    progress_score_raw = raw_item["progress_score"]
    if progress_score_raw == "n/a":
        progress_score = "n/a"
    else:
        try:
            progress_score = parse_percentage(progress_score_raw)
        except (TypeError, ValueError) as exc:
            print(
                f"Warning: Line {line_num} has invalid progress_score "
                f"({raw_item.get('progress_score')}): {exc}, skipping"
            )
            return None

    normalized = dict(raw_item)
    normalized["total_steps"] = total_steps
    normalized["closest_idx"] = closest_idx
    normalized["stage_to_estimate"] = stage_to_estimate
    normalized["progress_score"] = progress_score

    if image_root and not os.path.isabs(stage_to_estimate):
        normalized["stage_to_estimate"] = os.path.join(
            str(image_root), normalized["id"], stage_to_estimate
        )

    return normalized


def load_dataset_records(
    dataset_path: Path,
    *,
    dataset_type: str,
    num_inferences: int,
    image_root: Path | None,
) -> List[Tuple[DatasetType, Dict[str, Any]]]:
    forced_type: DatasetType | None = None
    if dataset_type in ("visual", "text"):
        forced_type = dataset_type  # type: ignore[assignment]

    raw_records: List[Tuple[DatasetType, Dict[str, Any]]] = []

    with dataset_path.open("r", encoding="utf-8") as fin:
        for line_num, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Warning: Failed to parse JSON at line {line_num}: {exc}")
                continue

            if forced_type is not None:
                record_type = forced_type
            else:
                if "visual_demo" in item:
                    record_type = "visual"
                elif "text_demo" in item:
                    record_type = "text"
                else:
                    print(
                        f"Warning: Line {line_num} missing both visual_demo and text_demo; unable to infer type"
                    )
                    continue

            if record_type == "visual" and "visual_demo" not in item:
                print(
                    f"Warning: Line {line_num} expected visual_demo dataset but field missing, skipping"
                )
                continue
            if record_type == "text" and "text_demo" not in item:
                print(
                    f"Warning: Line {line_num} expected text_demo dataset but field missing, skipping"
                )
                continue

            if record_type == "visual":
                normalized = _normalize_visual_item(
                    item, line_num=line_num, image_root=image_root
                )
            else:
                normalized = _normalize_text_item(
                    item, line_num=line_num, image_root=image_root
                )

            if normalized is None:
                continue

            raw_records.append((record_type, normalized))

    print(f"Loaded {len(raw_records)} raw samples from {dataset_path}")

    expanded_records: List[Tuple[DatasetType, Dict[str, Any]]] = []
    for record_type, normalized in raw_records:
        for inference_idx in range(num_inferences):
            expanded_item = dict(normalized)
            expanded_item["_inference_idx"] = inference_idx
            expanded_records.append((record_type, expanded_item))

    print(f"Expanded to {len(expanded_records)} samples (×{num_inferences})")

    return expanded_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ProgressLM JSONL demos to EasyR1 prompt format."
    )
    parser.add_argument("--input", type=Path, required=True, help="Source demo JSONL file")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSONL file")
    parser.add_argument(
        "--dataset-type",
        choices=("visual", "text", "auto"),
        default="auto",
        help="Type of demo data to process (auto-detected when omitted).",
    )
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
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the dataset records before writing (uses fixed seed=42)"
    )

    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    progresslm_root = args.progresslm_root or _default_progresslm_root(script_path)
    if progresslm_root is None:
        raise FileNotFoundError(
            "Unable to auto-detect ProgressLM repo location. Provide --progresslm-root explicitly."
        )

    _add_progresslm_to_syspath(progresslm_root)

    dataset_records = load_dataset_records(
        args.input,
        dataset_type=args.dataset_type,
        num_inferences=args.num_inferences,
        image_root=args.image_root,
    )

    # Shuffle if requested (using fixed seed for reproducibility)
    if args.shuffle:
        import random
        random.seed(42)
        random.shuffle(dataset_records)
        print(f"Shuffled dataset with seed=42")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    with args.output.open("w", encoding="utf-8") as fout:
        for record_type, item in dataset_records:
            if record_type == "visual":
                prompt, image_paths, segments = collect_visual_prompt_segments(
                    item,
                    min_pixels=args.min_pixels,
                    max_pixels=args.max_pixels,
                )
            else:
                prompt, image_paths, segments = collect_text_prompt_segments(
                    item,
                    min_pixels=args.min_pixels,
                    max_pixels=args.max_pixels,
                )

            answer = build_answer_payload(item, dataset_type=record_type)

            # Convert metadata values to strings for PyArrow compatibility
            closest_idx_val = item.get("closest_idx")
            progress_score_val = item.get("progress_score")
            delta_val = item.get("delta")

            metadata: Dict[str, Any] = {
                "closest_idx": str(closest_idx_val) if closest_idx_val is not None else "n/a",
                "progress_score": str(progress_score_val) if progress_score_val is not None else "n/a",
                "stage_to_estimate": item.get("stage_to_estimate"),
                "dataset_type": record_type,
            }
            if record_type == "text":
                metadata["text_demo"] = item.get("text_demo")
            else:
                metadata["visual_demo"] = item.get("visual_demo")
                metadata["delta"] = str(delta_val) if delta_val else "n/a"

            resolved_image_paths = _resolve_image_list(
                args.image_root, item["id"], image_paths
            )

            if args.image_root is not None:
                for segment in segments:
                    if segment.get("type") == "image":
                        segment["value"] = _resolve_image_path(
                            args.image_root, item["id"], segment["value"]
                        )

                if isinstance(metadata.get("visual_demo"), list):
                    metadata["visual_demo"] = _resolve_image_list(
                        args.image_root, item["id"], metadata["visual_demo"]
                    )

                stage_meta = metadata.get("stage_to_estimate")
                if isinstance(stage_meta, list):
                    metadata["stage_to_estimate"] = _resolve_image_list(
                        args.image_root, item["id"], stage_meta
                    )
                elif isinstance(stage_meta, str):
                    metadata["stage_to_estimate"] = _resolve_image_path(
                        args.image_root, item["id"], stage_meta
                    )

            record = {
                "id": item["id"],
                "task_goal": item["task_goal"],
                "system_prompt": "",
                "user_prompt": prompt,
                "answer": answer,
                "images": resolved_image_paths,
                "messages": segments,
                "metadata": metadata,
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            processed += 1

            if args.limit is not None and processed >= args.limit:
                break

    print(f"Wrote {processed} samples to {args.output}")


if __name__ == "__main__":
    main()


"""
python progresslm/data_preprocess/build_prompts_general.py   --input /projects/b1222/userdata/jianshu/chengxuan/ProgressLM/data/train/rl/rl_sampled_35k.jsonl --image-root /projects/b1222/userdata/jianshu/chengxuan/ProgressLM/data/images  --output /projects/b1222/userdata/jianshu/code/EasyR1/progresslm/data_preprocess/rl_sampled_35k_easyr1.jsonl
"""
