#!/usr/bin/env python3
"""Build a sanitized Hugging Face Dataset upload folder."""

import argparse
import hashlib
import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

from neo_logos.config.settings import PROJECT_ROOT

DATA_FILES = ("train.jsonl", "eval.jsonl", "test.jsonl", "dpo_pairs.jsonl")
LOCAL_PATH_MARKERS = (
    "/mnt/",
    "/home/",
    "Users/User",
    "Documents/github",
    "C:\\",
    "\\Users\\",
)
DEFAULT_DATASET_REPO_ID = "aetheronhq/neo-logos-training-dataset"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _count_jsonl(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_metadata(path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if path.suffix == ".jsonl":
        metadata["records"] = _count_jsonl(path)
    return metadata


def _infer_identity_qa_loaded(manifest: dict[str, Any]) -> int | None:
    sources = manifest.get("sources", {})
    if sources.get("identity_qa", {}).get("loaded") is not None:
        return int(sources["identity_qa"]["loaded"])

    processing = manifest.get("processing", {})
    total_formatted = processing.get("total_formatted")
    known_counts = (
        sources.get("identity", {}).get("loaded"),
        sources.get("articles", {}).get("loaded"),
        sources.get("conversations", {}).get("loaded"),
    )
    if total_formatted is None or any(count is None for count in known_counts):
        return None

    inferred = int(total_formatted) - sum(int(count) for count in known_counts)
    return inferred if inferred >= 0 else None


def _assert_no_local_paths(value: Any, location: str = "manifest") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            _assert_no_local_paths(nested, f"{location}.{key}")
        return
    if isinstance(value, list):
        for index, nested in enumerate(value):
            _assert_no_local_paths(nested, f"{location}[{index}]")
        return
    if isinstance(value, str) and any(marker in value for marker in LOCAL_PATH_MARKERS):
        raise ValueError(f"Local path leaked into {location}: {value}")


def sanitize_manifest(
    manifest: dict[str, Any],
    file_metadata: dict[str, dict[str, Any]] | None = None,
    dataset_repo_id: str = DEFAULT_DATASET_REPO_ID,
) -> dict[str, Any]:
    """Return a public manifest with host-local paths removed."""
    sanitized = deepcopy(manifest)
    sources = sanitized.setdefault("sources", {})

    for source in sources.values():
        if isinstance(source, dict):
            source.pop("path", None)

    identity_qa_loaded = _infer_identity_qa_loaded(sanitized)
    if identity_qa_loaded is not None and "identity_qa" not in sources:
        sources["identity_qa"] = {
            "loaded": identity_qa_loaded,
            "inferred_from_legacy_manifest": True,
        }

    source_total = 0
    for source in sources.values():
        if isinstance(source, dict) and source.get("loaded") is not None:
            source_total += int(source["loaded"])

    processing = sanitized.setdefault("processing", {})
    if source_total:
        processing["total_loaded"] = source_total
    if processing.get("total_formatted") == source_total:
        processing["dropped_invalid"] = 0
    elif processing.get("total_formatted") is not None and source_total:
        processing["dropped_invalid"] = max(
            0,
            source_total - int(processing["total_formatted"]),
        )

    sanitized["dpo_pairs"] = "dpo_pairs.jsonl"
    sanitized["release"] = {
        "dataset_repo": dataset_repo_id,
        "note": (
            "Sanitized for Hugging Face dataset release. Host-local source "
            "paths were removed from the legacy prepared_diverse manifest."
        ),
        "files": file_metadata or {},
    }
    _assert_no_local_paths(sanitized)
    return sanitized


def build_package(
    source_dir: Path,
    output_dir: Path,
    dataset_card: Path,
    dataset_repo_id: str,
    force: bool = False,
) -> Path:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    dataset_card = dataset_card.resolve()

    if not source_dir.exists():
        raise FileNotFoundError(f"Source dataset directory not found: {source_dir}")
    if not dataset_card.exists():
        raise FileNotFoundError(f"Dataset card not found: {dataset_card}")

    required_files = [source_dir / name for name in (*DATA_FILES, "manifest.json")]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required release files: " + ", ".join(missing))

    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise FileExistsError(f"Output directory is not empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {}
    for filename in DATA_FILES:
        source_file = source_dir / filename
        shutil.copy2(source_file, output_dir / filename)
        metadata[filename] = _file_metadata(source_file)

    manifest = _load_json(source_dir / "manifest.json")
    sanitized_manifest = sanitize_manifest(manifest, metadata, dataset_repo_id)
    _write_json(output_dir / "manifest.json", sanitized_manifest)
    shutil.copy2(dataset_card, output_dir / "README.md")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a sanitized Hugging Face Dataset upload folder"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=PROJECT_ROOT / "dataset_outputs" / "prepared_diverse" / "latest",
        help="Prepared dataset directory containing train/eval/test/DPO files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "dataset_outputs" / "hf" / "neo-logos-training-dataset",
        help="Output folder to upload with `hf upload ... --repo-type dataset`",
    )
    parser.add_argument(
        "--dataset-card",
        type=Path,
        default=PROJECT_ROOT / "DATASET_CARD.md",
        help="Dataset card markdown to copy to README.md",
    )
    parser.add_argument(
        "--dataset-repo-id",
        default=DEFAULT_DATASET_REPO_ID,
        help="Target Hugging Face dataset repo id recorded in manifest metadata",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the output directory if it already contains files",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = build_package(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        dataset_card=args.dataset_card,
        dataset_repo_id=args.dataset_repo_id,
        force=args.force,
    )
    print(f"HF dataset package written to: {output_dir}")
    print()
    print("Upload after review with:")
    print(
        f"  hf upload {args.dataset_repo_id} {output_dir} . "
        "--repo-type dataset"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
