"""Reward shaping for ProgressLM visual progress estimation with n/a handling."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

NA_TOKEN = "n/a"

FORMAT_PATTERN = re.compile(
    r"\s*<ref_think>.*?</ref_think>\s*"
    r"<ref>\s*(?:\d+|n/a)\s*</ref>\s*"
    r"<score_think>.*?</score_think>\s*"
    r"<score>\s*(?:[0-9]+(?:\.[0-9]+)?%?|n/a)\s*</score>\s*$",
    re.DOTALL | re.IGNORECASE,
)
REF_PATTERN = re.compile(r"<ref>\s*([^<]+)\s*</ref>", re.IGNORECASE)
SCORE_PATTERN = re.compile(r"<score>\s*([^<]+)\s*</score>", re.IGNORECASE)
SCORE_THINK_PATTERN = re.compile(r"<score_think>(.*?)</score_think>", re.DOTALL | re.IGNORECASE)


def _load_ground_truth(raw: Any) -> Dict[str, Any]:
    """Normalize ground-truth payload into a dictionary."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Ground-truth string must be valid JSON when using progresslm_reward."
            ) from exc
    raise TypeError("Ground-truth must be a dict or JSON string for progresslm_reward.")


def _compute_format_reward(response: str) -> float:
    """Return 1 when the response matches the required XML layout."""
    return 1.0 if FORMAT_PATTERN.fullmatch(response.strip()) else 0.0


def _extract_tag_value(pattern: re.Pattern[str], response: str) -> str | None:
    match = pattern.search(response)
    if match:
        return match.group(1).strip()
    return None


def _parse_ref_value(raw: Any) -> Tuple[bool, int | None]:
    """Return (is_na, value) for ref-like fields."""
    if isinstance(raw, (int, float)):
        if isinstance(raw, float) and not raw.is_integer():
            return False, None
        return False, int(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if text.lower() == NA_TOKEN:
            return True, None
        try:
            return False, int(text)
        except ValueError:
            try:
                as_float = float(text)
            except ValueError:
                return False, None
            return (False, int(as_float)) if as_float.is_integer() else (False, None)
    return False, None


def _parse_score_value(raw: Any) -> Tuple[bool, float | None]:
    """Return (is_na, value) for score-like fields."""
    if isinstance(raw, (int, float)):
        return False, float(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if text.lower() == NA_TOKEN:
            return True, None
        if text.endswith("%"):
            text = text[:-1].strip()
        try:
            return False, float(text)
        except ValueError:
            pass
    return False, None


def compute_score(reward_inputs: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    """
    Compute format/ref/score rewards for a batch of responses.

    Returns a list of dictionaries. Each dictionary contains:
      - overall: mean of the three component rewards (with weighting)
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

        gt_ref_is_na, gt_ref_value = _parse_ref_value(gt_payload["ref"])
        demo_count = max(int(gt_payload.get("demo_count", 1)), 1)
        gt_score_is_na, gt_score_value = _parse_score_value(
            gt_payload["score_percent"]
        )

        pred_ref_raw = _extract_tag_value(REF_PATTERN, response)
        pred_score_raw = _extract_tag_value(SCORE_PATTERN, response)

        pred_ref_is_na, pred_ref_value = _parse_ref_value(pred_ref_raw) if pred_ref_raw is not None else (False, None)
        pred_score_is_na, pred_score_value = _parse_score_value(pred_score_raw) if pred_score_raw is not None else (False, None)

        ref_error = 1.0
        ref_reward = 0.0
        score_error = 1.0
        score_reward = 0.0

        mismatch_na_state = (
            (gt_ref_is_na != pred_ref_is_na)
            or (gt_score_is_na != pred_score_is_na)
        )

        if gt_ref_is_na and pred_ref_is_na:
            ref_error = 0.0
            ref_reward = 1.0
        elif (
            not gt_ref_is_na
            and gt_ref_value is not None
            and pred_ref_value is not None
        ):
            max_offset = max(demo_count - 1, 1)
            ref_error = min(abs(pred_ref_value - gt_ref_value) / max_offset, 1.0)
            ref_reward = max(1.0 - ref_error**2, 0.0)

        if gt_score_is_na and pred_score_is_na:
            score_error = 0.0
            score_reward = 1.0
        elif (
            not gt_score_is_na
            and gt_score_value is not None
            and pred_score_value is not None
        ):
            score_error = min(
                abs(pred_score_value - gt_score_value) / 100.0,
                1.0,
            )
            score_reward = max(1.0 - score_error, 0.0)

        if mismatch_na_state:
            format_score = 0.0
            ref_reward = 0.0
            score_reward = 0.0
            ref_error = 1.0
            score_error = 1.0

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
