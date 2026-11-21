"""Judging Panel — VLM consensus over candidate masks.

Paper §Judging Panel:
    Score(M_g) = Σ_{j=1..N_VLM} Sign(V_j) · C_j
    Score > 0  ⇒  accept the candidate as a true nodule.

Each :class:`VLMJudge` wraps a single multimodal LLM and runs the exact
prompt from Figure 6 of the paper::

    #Task
    Given a lung CT image and its corresponding mask, align the two
    images, observe the masked region, and determine whether there is
    a lung nodule.

    ##Output Format
    {
      "Opinion":    "0/1, where 1 means yes and 0 means no.",
      "Confidence": "A value between 0 and 1, representing the
                     certainty of your judgment."
    }

The panel dispatches all judges in parallel per candidate, collects
their :class:`JudgeVote` s, applies the weighted-sum aggregation, and
returns a :class:`JudgedMask`.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field

from typing import TYPE_CHECKING

import numpy as np

from nodule_spotter.image_utils import InputMode, build_vlm_content
from nodule_spotter.types import ClusterResult, JudgedMask, JudgeVote

if TYPE_CHECKING:
    from agentbase.model.base import BaseLLM

logger = logging.getLogger(__name__)


JUDGE_PROMPT = """\
#Task
Given a lung CT image and its corresponding mask, align the two images, \
observe the masked region, and determine whether there is a lung nodule.

##Output Format
Respond with a single JSON object and nothing else:
{
  "Opinion": "0/1, where 1 means yes and 0 means no.",
  "Confidence": "A value between 0 and 1, representing the certainty of your judgment."
}
"""


# ── Single judge ──────────────────────────────────────────────────────

@dataclass
class VLMJudge:
    """One VLM acting as a single Judging-Panel member."""

    name: str
    model: "BaseLLM"
    input_mode: InputMode = "overlay"
    prompt: str = JUDGE_PROMPT

    async def vote(
        self, slice_image: np.ndarray, mask: np.ndarray
    ) -> JudgeVote:
        """Run the judge once and parse its structured verdict.

        Any parse failure or model error → sign=-1, confidence=0.0 so
        the candidate loses this vote rather than silently passing.
        """
        from agentbase.model.schema import Message  # lazy — keeps the
        # module importable without EasyAgent at module-load time.

        content = build_vlm_content(
            slice_image, mask, prompt_text=self.prompt, mode=self.input_mode
        )
        msgs = [Message.user(content)]
        try:
            response = await self.model.call_with_history(msgs)
            text = response.content or ""
        except Exception as e:  # noqa: BLE001
            logger.warning("judge %s crashed: %s", self.name, e)
            return JudgeVote(
                judge_name=self.name, sign=-1, confidence=0.0,
                raw_response=f"<error: {e}>",
            )

        opinion, confidence = _parse_verdict(text)
        if opinion is None or confidence is None:
            logger.info(
                "judge %s returned unparseable response: %r", self.name, text
            )
            return JudgeVote(
                judge_name=self.name, sign=-1, confidence=0.0,
                raw_response=text,
            )
        sign = 1 if opinion == 1 else -1
        return JudgeVote(
            judge_name=self.name,
            sign=sign,
            confidence=float(confidence),
            raw_response=text,
        )


# ── Panel ─────────────────────────────────────────────────────────────

@dataclass
class JudgingPanel:
    """A set of VLM judges with weighted-sum aggregation.

    Attributes
    ----------
    judges :
        The panel members. At least one is required.
    accept_threshold :
        Paper uses ``> 0``. Raise this to make the panel stricter
        (e.g. require an extra margin of agreement).
    drop_noise_clusters :
        If True (default), DBSCAN-noise clusters (``cluster_id == -1``)
        are auto-rejected without spending VLM calls on them — the
        paper treats those as outliers upstream.
    """

    judges: list[VLMJudge] = field(default_factory=list)
    accept_threshold: float = 0.0
    drop_noise_clusters: bool = True

    def __post_init__(self) -> None:
        if not self.judges:
            raise ValueError("JudgingPanel needs at least one VLMJudge")
        names = [j.name for j in self.judges]
        if len(set(names)) != len(names):
            raise ValueError(f"judge names must be unique; got {names}")

    # ── Per-candidate ────────────────────────────────────────────────

    async def evaluate(
        self,
        slice_image: np.ndarray,
        cluster: ClusterResult,
    ) -> JudgedMask:
        """Score one clustered candidate against the full panel."""
        if self.drop_noise_clusters and cluster.is_noise:
            return JudgedMask(cluster=cluster, votes=[], score=0.0, accepted=False)

        votes = await asyncio.gather(
            *[j.vote(slice_image, cluster.averaged_mask) for j in self.judges]
        )
        score = sum(v.weighted() for v in votes)
        return JudgedMask(
            cluster=cluster,
            votes=list(votes),
            score=float(score),
            accepted=score > self.accept_threshold,
        )

    # ── Batch convenience ────────────────────────────────────────────

    async def evaluate_all(
        self,
        slice_image: np.ndarray,
        clusters: list[ClusterResult],
    ) -> list[JudgedMask]:
        """Evaluate every cluster concurrently (one gather per panel call)."""
        return list(
            await asyncio.gather(
                *[self.evaluate(slice_image, c) for c in clusters]
            )
        )


# ── Output parsing ───────────────────────────────────────────────────

_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_verdict(text: str) -> tuple[int | None, float | None]:
    """Extract ``(opinion, confidence)`` from a judge's raw output.

    Accepts either a bare JSON object or one embedded in free text
    (common for chatty VLMs). The confidence string in the paper's
    prompt is free-form — we coerce values like ``"0.87"``, ``"87%"``
    or ``"high"`` into a float when possible, and otherwise give up
    and signal an unparseable response.
    """
    blob = _find_json_object(text)
    if blob is None:
        return None, None
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        return None, None
    if not isinstance(obj, dict):
        return None, None

    opinion = _coerce_opinion(obj.get("Opinion"))
    confidence = _coerce_confidence(obj.get("Confidence"))
    return opinion, confidence


def _find_json_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` substring, or None."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _coerce_opinion(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return 1 if int(value) == 1 else 0
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"1", "yes", "y", "true", "positive"}:
            return 1
        if s in {"0", "no", "n", "false", "negative"}:
            return 0
    return None


def _coerce_confidence(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    if isinstance(value, str):
        s = value.strip().rstrip("%")
        try:
            f = float(s)
        except ValueError:
            return None
        if f > 1.0:          # "87" or "87%" style
            f = f / 100.0
        return max(0.0, min(1.0, f))
    return None
