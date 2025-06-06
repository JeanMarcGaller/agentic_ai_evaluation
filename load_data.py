# === load_data.py ===

"""Helper functions to load data sets or user‑provided question files."""

# === Imports ===
from __future__ import annotations

import datetime as _dt
import json
import logging
import random
from pathlib import Path
from typing import Dict, Final, List

from datasets import load_dataset  # Hugging Face dataset loader

# === Logging ===
logger = logging.getLogger(__name__)


def get_hotpotqa_subset(num_samples: int = 3) -> List[Dict]:
    """
    Loads a random subset of the HotpotQA validation dataset.

    Args:
        num_samples (int): Number of questions to sample.

    Returns:
        list: Random sample of HotpotQA questions.
    """

    # Load distractor version of HotpotQA
    logger.info("Getting HotpotQA validation data.")
    dataset = load_dataset(
        "hotpot_qa",
        "distractor",
        split="validation",
        trust_remote_code=True,  # distractor needs custom loading script
    )

    # Sample number of examples
    subset = random.sample(list(dataset), num_samples)
    logger.info("Sampled %s questions from HotpotQA", num_samples)

    # Generate filename with timestamp
    timestamp: Final[str] = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path("data") / f"hotpotqa_subset_{timestamp}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(subset, indent=2), encoding="utf-8")
    logger.info("Saved HotpotQA subset to %s", path)

    return subset


def load_custom_questions(path: str | Path) -> List[Dict]:
    # Reads custom questions from a JSON file
    path = Path(path)
    logger.info("Loading custom questions from %s", path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Loaded %s custom questions", len(data))
    return data
