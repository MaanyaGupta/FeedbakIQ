"""
Interactive Aspect Annotation Tool for FeedbackIQ.
Provides an easy, rapid workflow for manually labeling customer review aspects.

Features:
- Displays review text, domain, and current assigned aspects.
- Shows available domain taxonomy aspects as numbered options.
- Allows multiple aspects per review (e.g. typing "1, 2, 4" or "battery, quality").
- Supports keyboard shortcuts:
  * [Enter]: Accept current aspects and proceed to next review.
  * Number selection: e.g. "1, 3" toggles/sets options 1 and 3.
  * "+aspect" or "-aspect": Add or remove a specific aspect.
  * "b" / "prev": Move back to previous review.
  * "goto <id>": Jump to specific review ID (e.g. "goto ELEC-015").
  * "stats": Display current annotation summary across aspects.
  * "q" / "exit": Save and quit.
- Automatically saves progress to data/aspect_annotations.csv after every change.
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Set

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANNOTATIONS_FILE = PROJECT_ROOT / "data" / "aspect_annotations.csv"

# Domain taxonomies
COMMON_ASPECTS = ["packaging", "quality", "price", "delivery", "customer_service"]
ELECTRONICS_ASPECTS = COMMON_ASPECTS + ["battery", "performance", "compatibility", "durability"]
FASHION_ASPECTS = COMMON_ASPECTS + ["size_fit", "material", "design", "color", "durability"]

DOMAIN_ASPECTS = {
    "Electronics": ELECTRONICS_ASPECTS,
    "Fashion": FASHION_ASPECTS,
}


def load_annotations() -> pd.DataFrame:
    if not ANNOTATIONS_FILE.exists():
        raise FileNotFoundError(f"Annotation file not found at: {ANNOTATIONS_FILE}")
    df = pd.read_csv(ANNOTATIONS_FILE)
    return df


def save_annotations(df: pd.DataFrame):
    df.to_csv(ANNOTATIONS_FILE, index=False)


def display_stats(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("CURRENT ANNOTATION STATISTICS")
    print("=" * 60)
    for domain in ["Electronics", "Fashion"]:
        sub = df[df["domain"] == domain]
        counts: Dict[str, int] = {}
        for labels in sub["aspect_labels"].dropna():
            for a in [x.strip() for x in str(labels).split(";") if x.strip()]:
                counts[a] = counts.get(a, 0) + 1
        print(f"\n--- {domain} (Total: {len(sub)} reviews) ---")
        for a, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            pct = (cnt / len(sub)) * 100.0 if len(sub) > 0 else 0
            bar = "=" * int(pct / 4)
            print(f"  {a:18s} : {cnt:3d} ({pct:5.1f}%) |{bar}|")
    print("=" * 60 + "\n")


def interactive_annotator():
    df = load_annotations()
    total = len(df)
    current_idx = 0

    print("=" * 70)
    print("FEEDBACKIQ INTERACTIVE ASPECT ANNOTATOR")
    print(f"Loaded {total} reviews from {ANNOTATIONS_FILE}")
    print("Shortcuts:")
    print("  - Press [Enter] to keep current aspects and go to next")
    print("  - Type comma-separated numbers (e.g. '1, 3, 5') to select aspects")
    print("  - Type aspect names directly (e.g. 'battery, quality')")
    print("  - Type 'b' to go back, 'goto <id>' to jump, 'stats' for summary, 'q' to quit")
    print("=" * 70)

    while current_idx < total:
        row = df.iloc[current_idx]
        review_id = row["review_id"]
        domain = row["domain"]
        text = row["text"]
        current_labels = [a.strip() for a in str(row["aspect_labels"]).split(";") if a.strip()]

        valid_aspects = DOMAIN_ASPECTS.get(domain, ELECTRONICS_ASPECTS)

        print("\n" + "-" * 70)
        print(f"Review [{current_idx + 1}/{total}] | ID: {review_id} | Domain: {domain}")
        print("-" * 70)
        print(f"TEXT:\n\"{text}\"")
        print("\nCURRENT ASPECTS:")
        if current_labels:
            for cl in current_labels:
                print(f"  -> {cl}")
        else:
            print("  (None)")

        print("\nAVAILABLE TAXONOMY OPTIONS:")
        for idx, aspect in enumerate(valid_aspects, 1):
            is_selected = " [SELECTED]" if aspect in current_labels else ""
            print(f"  [{idx:2d}] {aspect}{is_selected}")

        user_input = input(f"\nSelect aspects for {review_id} ([Enter] to accept, numbers, or 'q'): ").strip()

        if user_input.lower() in ["q", "quit", "exit"]:
            print("Exiting annotator. All progress is saved.")
            break

        if user_input.lower() == "stats":
            display_stats(df)
            continue

        if user_input.lower() in ["b", "prev", "back"]:
            current_idx = max(0, current_idx - 1)
            continue

        if user_input.lower().startswith("goto"):
            parts = user_input.split()
            if len(parts) > 1:
                target_id = parts[1].strip().upper()
                matches = df.index[df["review_id"] == target_id].tolist()
                if matches:
                    current_idx = matches[0]
                    continue
                else:
                    print(f"Error: Review ID '{target_id}' not found.")
                    continue

        if not user_input:
            # Keep current aspects and advance
            current_idx += 1
            continue

        # Parse user selection
        new_aspects = set()
        tokens = [t.strip() for t in user_input.replace(";", ",").split(",") if t.strip()]

        valid_choice = True
        for token in tokens:
            if token.isdigit():
                num = int(token)
                if 1 <= num <= len(valid_aspects):
                    new_aspects.add(valid_aspects[num - 1])
                else:
                    print(f"Warning: Number {num} is out of bounds (1-{len(valid_aspects)}).")
                    valid_choice = False
            elif token.lower() in valid_aspects:
                new_aspects.add(token.lower())
            else:
                print(f"Warning: Unknown aspect '{token}'. Valid options: {valid_aspects}")
                valid_choice = False

        if valid_choice and new_aspects:
            # Sort according to canonical domain order
            sorted_aspects = [a for a in valid_aspects if a in new_aspects]
            df.at[current_idx, "aspect_labels"] = "; ".join(sorted_aspects)
            save_annotations(df)
            print(f"Updated {review_id} -> {'; '.join(sorted_aspects)}")
            current_idx += 1
        elif not valid_choice:
            print("Please try again.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--stats":
        df = load_annotations()
        display_stats(df)
    else:
        interactive_annotator()
