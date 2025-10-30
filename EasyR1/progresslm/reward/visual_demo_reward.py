"""Reward shaping for ProgressLM visual progress estimation."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

FORMAT_PATTERN = re.compile(
    r"\s*<ref_think>.*?</ref_think>\s*"
    r"<ref>\s*\d+\s*</ref>\s*"
    r"<score_think>.*?</score_think>\s*"
    r"<score>\s*[0-9]+(?:\.[0-9]+)?%?\s*</score>\s*$",
    re.DOTALL,
)
REF_PATTERN = re.compile(r"<ref>\s*(\d+)\s*</ref>")
SCORE_PATTERN = re.compile(r"<score>\s*([0-9]+(?:\.[0-9]+)?)%?\s*</score>")


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


def _extract_ref(response: str) -> int | None:
    match = REF_PATTERN.search(response)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def _extract_score_percent(response: str) -> float | None:
    match = SCORE_PATTERN.search(response)
    if match:
        try:
            return float(match.group(1))
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
    """
    scores: List[Dict[str, float]] = []

    for reward_input in reward_inputs:
        response = reward_input["response"]
        gt_payload = _load_ground_truth(reward_input["ground_truth"])

        format_score = _compute_format_reward(response)

        gt_ref = int(gt_payload["ref"])
        demo_count = max(int(gt_payload.get("demo_count", 1)), 1)
        gt_score_percent = float(gt_payload["score_percent"])

        pred_ref = _extract_ref(response)
        if pred_ref is None:
            ref_error = 1.0
        else:
            max_offset = max(demo_count - 1, 1)
            ref_error = min(abs(pred_ref - gt_ref) / max_offset, 1.0)
        ref_reward = max(1.0 - ref_error**2, 0.0)

        pred_score = _extract_score_percent(response)
        if pred_score is None:
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
