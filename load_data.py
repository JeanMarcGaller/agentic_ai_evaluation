# === load_data.py ===

import json
import random
import datetime
from datasets import load_dataset # From Hugging Face – loads datasets like HotpotQA

def get_hotpotqa_subset(num_samples=3): # TODO: Check num_samples usage in code
    """
       Loads a random subset of the HotpotQA validation dataset.

       Args:
           num_samples (int): Number of questions to sample.

       Returns:
           list: Random sample of HotpotQA questions. # TODO: Test Natural Questions, WebQuestions
       """

    # Load distractor version of HotpotQA
    dataset = load_dataset(
        "hotpot_qa",
        "distractor",
        split="validation",
        trust_remote_code=True # distractor needs custom loading script
    )

    # Sample number of examples
    subset = random.sample(list(dataset), num_samples)

    # Generate filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"data/hotpotqa_subset_{timestamp}.json"

    # Save subset of samples
    with open(path, "w") as f:
        json.dump(subset, f, indent=2)

    print(f"Sample subset saved {path}")
    return subset


def load_custom_questions(path: str) -> list[dict]:
    # Reads custom questions from a JSON file
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
