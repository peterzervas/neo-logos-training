#!/usr/bin/env python3
"""
Consolidate all generated Neo-Logos data into clean 'latest' directories.

Handles scattered data across multiple timestamped directories from
different generation runs and top-ups. Merges everything, deduplicates,
creates proper symlinks, and verifies all paths before training.

Usage:
    python -m neo_logos.scripts.consolidate           # Merge + verify
    python -m neo_logos.scripts.consolidate --verify   # Verify only (no merge)
"""

import argparse
import hashlib
import os
import shutil
from datetime import datetime

from neo_logos.config.settings import PROJECT_ROOT

DATASET_DIR = PROJECT_ROOT / "dataset_outputs"

# Map of data types -> expected filename patterns and minimum counts
DATA_TYPES = {
    "neo_logos_identity": {
        "filename": "output.jsonl",
        "min_count": 3000,
        "target_count": 3300,
        "description": "Identity narratives",
    },
    "identity_qa": {
        "filename": "identity_qa.jsonl",
        "min_count": 400,
        "target_count": 500,
        "description": "Identity Q&A pairs",
    },
    "neo_logos_articles": {
        "filename": "output.jsonl",
        "min_count": 2400,
        "target_count": 2500,
        "description": "Articles Q&A",
    },
    "conversations": {
        "filename": "conversations.jsonl",
        "min_count": 4000,
        "target_count": 4750,
        "description": "Conversations",
    },
    "dpo_pairs": {
        "filename": "dpo_pairs.jsonl",
        "min_count": 1500,
        "target_count": 1990,
        "description": "DPO preference pairs",
    },
}

# What prepare_diverse_training.py needs
PREPARE_PATHS = {
    "identity": "neo_logos_identity/latest/output.jsonl",
    "articles": "neo_logos_articles/latest/output.jsonl",
    "conversations": "conversations/latest/conversations.jsonl",
    "identity_qa": "identity_qa/latest/identity_qa.jsonl",
    "dpo_pairs": "dpo_pairs/latest/dpo_pairs.jsonl",
}


def find_all_jsonl(type_dir, filename):
    """Find all JSONL files matching filename pattern in a type directory.

    Scans ALL timestamped subdirectories. Skips 'latest' (symlink) and
    'merged' (our own output) to avoid double-counting. Always rebuilds
    merged from source timestamped dirs.
    """
    files = []
    type_path = DATASET_DIR / type_dir
    if not type_path.exists():
        return files

    # Check for file directly in the type directory (e.g., manually placed)
    direct = type_path / filename
    if direct.exists():
        files.append(direct)

    for subdir in sorted(type_path.iterdir()):
        if not subdir.is_dir():
            continue
        # Skip our own output dirs - we rebuild these from source
        if subdir.name in ("latest", "merged"):
            continue
        # Skip symlinks to avoid double-counting
        if subdir.is_symlink():
            continue

        candidate = subdir / filename
        if candidate.exists():
            files.append(candidate)

    return files


def merge_jsonl_files(files, output_path):
    """Merge multiple JSONL files, deduplicating by content hash."""
    seen = set()
    merged = []

    for filepath in files:
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Deduplicate by content hash
                h = hashlib.md5(line.encode()).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
                merged.append(line)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for line in merged:
            f.write(line + "\n")

    return len(merged)


def create_latest_link(type_dir):
    """Create 'latest' symlink pointing to 'merged' directory."""
    type_path = DATASET_DIR / type_dir
    latest = type_path / "latest"
    merged = type_path / "merged"

    if not merged.exists():
        return False

    # Replace existing latest. If it is a real directory, preserve it instead
    # of deleting generated artifacts outright.
    if latest.is_symlink():
        latest.unlink()
    elif latest.is_dir():
        backup = latest.with_name(
            f"latest.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        shutil.move(str(latest), str(backup))
        print(f"  Backed up existing latest directory -> {backup.name}")

    try:
        os.symlink("merged", str(latest))
    except OSError:
        # WSL symlink issues - just copy instead
        shutil.copytree(str(merged), str(latest))

    return True


def consolidate():
    """Merge all scattered data into clean 'latest' directories."""
    print("=" * 60)
    print("NEO-LOGOS DATA CONSOLIDATION")
    print("=" * 60)

    results = {}

    for type_dir, config in DATA_TYPES.items():
        print(f"\n--- {config['description']} ({type_dir}) ---")

        files = find_all_jsonl(type_dir, config["filename"])
        if not files:
            print("  WARNING: No JSONL files found!")
            results[type_dir] = 0
            continue

        print(f"  Found {len(files)} source file(s):")
        for f in files:
            line_count = sum(1 for _ in open(f, encoding="utf-8") if _.strip())
            print(f"    {f.relative_to(DATASET_DIR)}: {line_count} lines")

        # Merge into consolidated file
        merged_path = DATASET_DIR / type_dir / "merged" / config["filename"]
        count = merge_jsonl_files(files, merged_path)
        print(f"  Merged: {count} unique examples → {merged_path.relative_to(DATASET_DIR)}")

        # Create latest symlink
        if create_latest_link(type_dir):
            print("  Created latest → merged/")
        else:
            print("  WARNING: Could not create latest link!")

        results[type_dir] = count

        # Check against targets
        if count >= config["target_count"]:
            print(f"  Status: ON TARGET ({count}/{config['target_count']})")
        elif count >= config["min_count"]:
            gap = config["target_count"] - count
            print(f"  Status: ACCEPTABLE ({count}/{config['target_count']}, gap: {gap})")
        else:
            gap = config["min_count"] - count
            print(f"  Status: BELOW MINIMUM! ({count}/{config['min_count']}, need {gap} more)")

    return results


def verify():
    """Verify all paths that prepare_diverse_training.py needs."""
    print("\n" + "=" * 60)
    print("PATH VERIFICATION")
    print("=" * 60)

    all_good = True
    total_sft = 0

    for label, rel_path in PREPARE_PATHS.items():
        full_path = DATASET_DIR / rel_path
        if full_path.exists():
            count = sum(1 for line in open(full_path, encoding="utf-8") if line.strip())
            print(f"  OK: {rel_path} ({count} lines)")
            if label != "dpo_pairs":
                total_sft += count
        else:
            print(f"  MISSING: {rel_path}")
            all_good = False

    print(f"\n  Total SFT examples: {total_sft}")
    print("  Target: ≥10,000")

    if total_sft >= 10000:
        print("  Status: ON TARGET")
    elif total_sft >= 8000:
        print("  Status: ACCEPTABLE (may want to top up)")
    else:
        print("  Status: BELOW TARGET - need more data!")
        all_good = False

    if all_good:
        print("\n  READY FOR TRAINING")
    else:
        print("\n  NOT READY - fix missing/insufficient data first")

    print("=" * 60)
    return all_good


def get_current_counts():
    """Return current merged counts per data type (for top-up calculations)."""
    counts = {}
    for type_dir, config in DATA_TYPES.items():
        merged_path = DATASET_DIR / type_dir / "merged" / config["filename"]
        if merged_path.exists():
            counts[type_dir] = sum(1 for line in open(merged_path, encoding="utf-8") if line.strip())
        else:
            # Try latest symlink
            latest_path = DATASET_DIR / type_dir / "latest" / config["filename"]
            if latest_path.exists():
                counts[type_dir] = sum(1 for line in open(latest_path, encoding="utf-8") if line.strip())
            else:
                counts[type_dir] = 0
    return counts


def main():
    parser = argparse.ArgumentParser(description="Consolidate Neo-Logos generated data")
    parser.add_argument("--verify", action="store_true", help="Verify only (no merge)")
    args = parser.parse_args()

    if args.verify:
        verify()
    else:
        consolidate()
        verify()


if __name__ == "__main__":
    main()
