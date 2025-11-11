"""Reward shaping for ProgressLM visual progress estimation."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

FORMAT_PATTERN = re.compile(
    r"\s*<ref_think>.*?</ref_think>\s*"
    r"<ref>\s*(?:\d+|n/a)\s*</ref>\s*"
    r"<score_think>.*?</score_think>\s*"
    r"<score>\s*(?:[0-9]+(?:\.[0-9]+)?%?|n/a)\s*</score>\s*$",
    re.DOTALL,
)
REF_PATTERN = re.compile(r"<ref>\s*(\d+|n/a)\s*</ref>")
SCORE_PATTERN = re.compile(r"<score>\s*([0-9]+(?:\.[0-9]+)?%?|n/a)\s*</score>")


def _load_ground_truth(raw: Any) -> Dict[str, Any]:
    """Normalize ground-truth payload into a dictionary."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Ground-truth string must be valid JSON when using visual_demo_reward."
            ) from exc
    raise TypeError(
        "Ground-truth must be a dict or JSON string for visual_demo_reward."
    )


def _compute_format_reward(response: str) -> float:
    """Return 1 when the response matches the required XML layout."""
    return 1.0 if FORMAT_PATTERN.fullmatch(response.strip()) else 0.0


def _extract_ref(response: str) -> int | str | None:
    """Extract ref value from response. Returns int, "n/a", or None."""
    match = REF_PATTERN.search(response)
    if match:
        value = match.group(1)
        if value == "n/a":
            return "n/a"
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _extract_score_percent(response: str) -> float | str | None:
    """Extract score value from response. Returns float, "n/a", or None."""
    match = SCORE_PATTERN.search(response)
    if match:
        value = match.group(1)
        if value == "n/a":
            return "n/a"
        try:
            # Remove % if present
            value = value.rstrip('%')
            return float(value)
        except ValueError:
            return None
    return None


def compute_score(reward_inputs: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    """
    Compute format/ref/score rewards for a batch of responses.

    Returns a list of dictionaries. Each dictionary contains:
      - overall: mean of the three component rewards
      - format: binary reward for tag formatting
      - accuracy: alias for score reward (for logger compatibility)
      - score: score reward (1 - relative error w.r.t. ground-truth progress)
      - ref: reference-frame reward (1 - scaled position error)
      - ref_error / score_error: diagnostic metrics for logging

    Handles "n/a" cases:
      - GT is "n/a" and pred is "n/a" → high reward (correct abnormal detection)
      - GT is "n/a" and pred is value → low reward (false positive)
      - GT is value and pred is "n/a" → low reward (false negative)
    """
    scores: List[Dict[str, float]] = []

    for reward_input in reward_inputs:
        response = reward_input["response"]
        gt_payload = _load_ground_truth(reward_input["ground_truth"])

        format_score = _compute_format_reward(response)

        # Parse ground truth - handle string types from our data format
        gt_ref_raw = gt_payload["ref"]
        gt_ref_is_na = (gt_ref_raw == "n/a")
        gt_ref = None if gt_ref_is_na else int(gt_ref_raw)

        demo_count = max(int(gt_payload.get("demo_count", 1)), 1)

        gt_score_raw = gt_payload["score_percent"]
        gt_score_is_na = (gt_score_raw == "n/a")
        gt_score_percent = None if gt_score_is_na else float(gt_score_raw)

        # Extract predictions
        pred_ref = _extract_ref(response)
        pred_score = _extract_score_percent(response)

        # Compute ref reward
        if gt_ref_is_na:
            # Ground truth is n/a
            if pred_ref == "n/a":
                ref_reward = 0.7  # Correct: predicted abnormal, but softer reward
                ref_error = 0.0
            else:
                ref_reward = -1.0  # Wrong: predicted value when should be n/a (false positive)
                ref_error = 1.0
        else:
            # Ground truth is a valid number
            if pred_ref == "n/a":
                ref_reward = 0.0  # Wrong: predicted n/a when should be value
                ref_error = 1.0
            elif pred_ref is None:
                ref_reward = 0.0  # Failed to extract
                ref_error = 1.0
            else:
                max_offset = max(demo_count - 1, 1)
                ref_error = min(abs(pred_ref - gt_ref) / max_offset, 1.0)
                ref_reward = max(1.0 - ref_error, 0.0)

        # Compute score reward
        if gt_score_is_na:
            # Ground truth is n/a
            if pred_score == "n/a":
                score_reward = 0.7  # Correct: predicted abnormal, but softer reward
                score_error = 0.0
            else:
                score_reward = -1.0  # Wrong: predicted value when should be n/a (false positive)
                score_error = 1.0
        else:
            # Ground truth is a valid number
            if pred_score == "n/a":
                score_reward = -1.0  # Wrong: predicted n/a when should be value
                score_error = 1.0
            elif pred_score is None:
                score_reward = -1.0  # Failed to extract
                score_error = 1.0
            else:
                score_error = min(abs(pred_score - gt_score_percent) / 100.0, 1.0)
                score_reward = max(1.0 - score_error, 0.0)

        overall = (
            0.1 * format_score
            + 0.6 * ref_reward
            + 0.3 * score_reward
        )

        scores.append(
            {
                "overall": overall,
                "format": format_score,
                "accuracy": score_reward,
                "score": score_reward,
                "ref": ref_reward,
                "ref_error": ref_error,
                "score_error": score_error,
            }
        )

    return scores
