# === load_data.py ===

import json
import random

from datasets import load_dataset


def get_hotpotqa_subset(path="hotpotqa_subset.json", num_samples=10):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        dataset = load_dataset(
            "hotpot_qa", "distractor", split="validation", trust_remote_code=True
        )
        subset = random.sample(list(dataset), num_samples)
        with open(path, "w") as f:
            json.dump(subset, f, indent=2)
        return subset
