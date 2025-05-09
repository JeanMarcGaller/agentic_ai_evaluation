# === load_data.py ===

import json
import random
import os
from datasets import load_dataset


def get_hotpotqa_subset(num_samples, path="hotpotqa_subset.json", force_reload=False):
    if not force_reload and os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)

    dataset = load_dataset("hotpot_qa", "distractor", split="validation", trust_remote_code=True)
    subset = random.sample(list(dataset), num_samples)

    with open(path, "w") as f:
        json.dump(subset, f, indent=2)

    return subset

