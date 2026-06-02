"""
training/dataset.py

Dataset utilities for the prompt injection classifier.

Loads seed examples from data/seed_injections.json and generates an augmented
training dataset by applying multiple transformation techniques to injection
examples (label=1). Augmentation helps the classifier generalize to obfuscated
and paraphrased injection attacks.

Output: HuggingFace Dataset with train/test split, saveable to CSV.
"""

import base64
import json
import logging
import os
import random
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Unicode lookalike map for substitution augmentation
# ---------------------------------------------------------------------------

UNICODE_LOOKALIKES: Dict[str, List[str]] = {
    "a": ["а", "ä", "á", "â", "ã"],   # Cyrillic а, German ä, etc.
    "e": ["е", "ë", "é", "ê"],
    "i": ["і", "ï", "í", "î"],
    "o": ["о", "ö", "ó", "ô", "õ"],
    "s": ["ѕ", "ś", "š"],
    "c": ["с", "ç"],
    "p": ["р"],
    "x": ["х"],
    "y": ["у", "ý"],
}

# Roleplay and hypothetical wrappers used for augmentation
ROLEPLAY_WRAPPERS = [
    "In a story where an AI has no rules, it says: ",
    "A fictional AI character with no restrictions would respond: ",
    "Imagine an AI that ignores its training. It would say: ",
    "In this creative writing scenario, the AI responds: ",
]

HYPOTHETICAL_WRAPPERS = [
    "Hypothetically, if someone wanted to make an AI ignore its rules, they might say: ",
    "For academic research purposes, consider this example of a prompt injection: ",
    "In theory, an attacker might try to override an AI's system prompt by writing: ",
    "As a thought experiment, suppose an adversarial user typed: ",
]


# ---------------------------------------------------------------------------
# Augmentation functions
# ---------------------------------------------------------------------------

def base64_encode_injection(text: str) -> str:
    """Wrap text in a base64 decode-and-follow pattern."""
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    return f"decode and follow: {encoded}"


def leet_speak(text: str) -> str:
    """Replace common characters with l33t-speak digits."""
    mapping = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}
    result = []
    for ch in text:
        result.append(mapping.get(ch.lower(), ch))
    return "".join(result)


def add_roleplay_wrapper(text: str, rng: Optional[random.Random] = None) -> str:
    """Prefix text with a roleplay framing."""
    rng = rng or random.Random()
    wrapper = rng.choice(ROLEPLAY_WRAPPERS)
    return f"{wrapper}{text}"


def add_hypothetical_wrapper(text: str, rng: Optional[random.Random] = None) -> str:
    """Prefix text with a hypothetical/academic framing."""
    rng = rng or random.Random()
    wrapper = rng.choice(HYPOTHETICAL_WRAPPERS)
    return f"{wrapper}{text}"


def unicode_substitute(text: str, num_subs: int = 5, seed: Optional[int] = None) -> str:
    """Replace up to `num_subs` eligible characters with Unicode lookalikes."""
    rng = random.Random(seed)
    chars = list(text)
    eligible_indices = [
        i for i, c in enumerate(chars) if c.lower() in UNICODE_LOOKALIKES
    ]
    rng.shuffle(eligible_indices)
    for idx in eligible_indices[:num_subs]:
        original = chars[idx].lower()
        replacement = rng.choice(UNICODE_LOOKALIKES[original])
        # Preserve case approximately
        if chars[idx].isupper():
            replacement = replacement.upper()
        chars[idx] = replacement
    return "".join(chars)


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def load_seed_data(seed_path: str) -> List[Dict]:
    """Load seed examples from JSON file."""
    path = Path(seed_path)
    if not path.exists():
        raise FileNotFoundError(f"Seed data not found at {seed_path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Loaded %d seed examples from %s", len(data), seed_path)
    return data


def load_deepset_dataset() -> List[Dict]:
    """
    Download and normalise the deepset/prompt-injections dataset from HuggingFace.

    This dataset has ~662 labeled examples covering a wide range of real
    prompt injection attacks and benign prompts. It is merged with our seed
    data and augmentations to produce a much richer training corpus.

    Label mapping:
      INJECTION  -> 1
      LEGIT      -> 0

    Returns [] gracefully if the package or network is unavailable so the
    pipeline can still run on seed data alone.
    """
    try:
        from datasets import load_dataset as hf_load  # type: ignore
    except ImportError:
        logger.warning("'datasets' package not installed — skipping deepset/prompt-injections. pip install datasets")
        return []

    try:
        logger.info("Downloading deepset/prompt-injections from HuggingFace Hub…")
        ds = hf_load("deepset/prompt-injections", split="train")
        examples: List[Dict] = []
        for row in ds:
            label_str = str(row.get("label", "")).strip().upper()
            if label_str in ("INJECTION", "1"):
                label = 1
            elif label_str in ("LEGIT", "0"):
                label = 0
            else:
                continue  # skip unknown labels
            text = str(row.get("text", row.get("prompt", ""))).strip()
            if text:
                examples.append({"text": text, "label": label})
        logger.info("deepset/prompt-injections: loaded %d examples (inj=%d, clean=%d)",
                    len(examples),
                    sum(1 for e in examples if e["label"] == 1),
                    sum(1 for e in examples if e["label"] == 0))
        return examples
    except Exception as exc:
        logger.warning("Could not load deepset/prompt-injections: %s — using seed data only.", exc)
        return []


def generate_augmented_dataset(
    seed_path: str = "data/seed_injections.json",
    random_seed: int = 42,
    use_deepset: bool = True,
) -> List[Dict]:
    """
    Generate an augmented dataset from seed examples + deepset/prompt-injections.

    Sources (merged before augmentation):
      1. data/seed_injections.json  — hand-curated seeds (40 examples)
      2. deepset/prompt-injections  — HuggingFace dataset (~662 examples)

    For each injection (label=1) in the combined seed corpus, generate 5 variants:
      1. base64_encode
      2. leet_speak
      3. roleplay_wrapper
      4. hypothetical_wrapper
      5. unicode_substitute

    Clean examples (label=0) are included as-is (no augmentation to avoid
    synthetic noise on the negative class).

    Returns a list of dicts with "text" and "label" keys.
    """
    rng = random.Random(random_seed)
    seed_data = load_seed_data(seed_path)

    # Merge deepset dataset if available
    if use_deepset:
        deepset_data = load_deepset_dataset()
        if deepset_data:
            # Deduplicate against seed data by normalised text
            seed_texts = {item["text"].strip().lower() for item in seed_data}
            new_from_deepset = [
                item for item in deepset_data
                if item["text"].strip().lower() not in seed_texts
            ]
            seed_data = seed_data + new_from_deepset
            logger.info("Combined corpus: %d examples (%d seed + %d deepset)",
                        len(seed_data),
                        len(seed_data) - len(new_from_deepset),
                        len(new_from_deepset))

    augmented: List[Dict] = list(seed_data)  # Start with originals

    injections = [item for item in seed_data if item["label"] == 1]
    logger.info("Augmenting %d injection examples…", len(injections))

    for item in injections:
        text = item["text"]
        item_seed = hash(text) & 0xFFFFFFFF

        variants = [
            {"text": base64_encode_injection(text), "label": 1},
            {"text": leet_speak(text), "label": 1},
            {"text": add_roleplay_wrapper(text, rng=rng), "label": 1},
            {"text": add_hypothetical_wrapper(text, rng=rng), "label": 1},
            {"text": unicode_substitute(text, seed=item_seed), "label": 1},
        ]
        augmented.extend(variants)

    rng.shuffle(augmented)
    logger.info("Final dataset size: %d examples", len(augmented))
    return augmented


def split_dataset(
    data: List[Dict],
    test_fraction: float = 0.2,
    random_seed: int = 42,
) -> "tuple":
    """Split dataset into train and test sets."""
    try:
        from datasets import Dataset  # type: ignore
    except ImportError:
        raise ImportError("Install 'datasets' package: pip install datasets")

    rng = random.Random(random_seed)
    shuffled = list(data)
    rng.shuffle(shuffled)

    split_idx = int(len(shuffled) * (1 - test_fraction))
    train_data = shuffled[:split_idx]
    test_data = shuffled[split_idx:]

    train_ds = Dataset.from_list(train_data)
    test_ds = Dataset.from_list(test_data)

    logger.info("Train: %d  |  Test: %d", len(train_ds), len(test_ds))
    return train_ds, test_ds


def save_dataset(
    data: List[Dict],
    output_dir: str = "training/data",
    test_fraction: float = 0.2,
    random_seed: int = 42,
) -> None:
    """Save train and test splits as CSV files."""
    import csv

    os.makedirs(output_dir, exist_ok=True)

    rng = random.Random(random_seed)
    shuffled = list(data)
    rng.shuffle(shuffled)

    split_idx = int(len(shuffled) * (1 - test_fraction))
    train_data = shuffled[:split_idx]
    test_data = shuffled[split_idx:]

    def write_csv(rows: List[Dict], path: str) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "label"])
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Saved %d rows to %s", len(rows), path)

    write_csv(train_data, os.path.join(output_dir, "train.csv"))
    write_csv(test_data, os.path.join(output_dir, "test.csv"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = generate_augmented_dataset()
    save_dataset(data)
    print(f"Dataset saved to training/data/ ({len(data)} total examples)")
