# === load_data.py ===

import json
import random
import datetime
from datasets import load_dataset

def get_hotpotqa_subset(num_samples=3):
    dataset = load_dataset(
        "hotpot_qa", "distractor", split="validation", trust_remote_code=True
    )
    subset = random.sample(list(dataset), num_samples)

    # Optional: neuen Dateinamen mit Zeitstempel verwenden
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"hotpotqa_subset_{timestamp}.json"

    with open(path, "w") as f:
        json.dump(subset, f, indent=2)

    print(f"🆕 HotpotQA subset saved to {path}")
    return subset


