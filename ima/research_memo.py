"""Pinned research handoff delivered with every V6 planning snapshot."""
from hashlib import sha256
from pathlib import Path


MEMO_PATH = Path(__file__).resolve().parents[1] / "docs" / "V4_V5_RESEARCH_HANDOFF_TO_V6.md"


def research_memo_context():
    raw = MEMO_PATH.read_bytes()
    return {
        "path": "docs/V4_V5_RESEARCH_HANDOFF_TO_V6.md",
        "sha256": sha256(raw).hexdigest(),
        "content": raw.decode("utf-8"),
        "instruction": (
            "Read this entire handoff before allocating trials. Acknowledge its exact SHA256 "
            "in research_memo_sha256. Explain how relevant prior findings inform hypotheses "
            "or review_reason. Historical scores require matched current-dataset replays; "
            "historical attempt IDs are not current-campaign parents."
        ),
    }
