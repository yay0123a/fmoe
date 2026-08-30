"""Build a deterministic SemanticRT manifest from overlapping split files."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

DEFAULT_SPLITS = (
    "test_day",
    "test_night",
    "test_hard",
    "test_mc",
    "test_mo",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/semantic_rt"))
    parser.add_argument(
        "--mfif-root", type=Path, default=Path("data/mfif/semantic_rt")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    return parser


def load_ids(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"SemanticRT split does not exist: {path}")
    values = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not values:
        raise ValueError(f"SemanticRT split is empty: {path}")
    return values


def required_paths(dataset_root: Path, mfif_root: Path, sample_id: str) -> tuple[Path, ...]:
    return (
        dataset_root / "rgb" / f"{sample_id}.jpg",
        dataset_root / "thermal" / f"{sample_id}.jpg",
        dataset_root / "labels" / f"{sample_id}.png",
        mfif_root / "AiF" / f"{sample_id}.jpg",
        mfif_root / "quantized_depth" / f"{sample_id}.png",
        mfif_root / "dof_stack" / sample_id / "0.jpg",
        mfif_root / "dof_stack" / sample_id / "1.jpg",
    )


def main() -> None:
    args = build_parser().parse_args()
    split_ids = {
        name: load_ids(args.dataset_root / f"{name}.txt") for name in args.splits
    }
    candidates = set().union(*split_ids.values())
    complete = sorted(
        sample_id
        for sample_id in candidates
        if all(
            path.is_file()
            for path in required_paths(args.dataset_root, args.mfif_root, sample_id)
        )
    )
    if args.count <= 0 or args.count > len(complete):
        raise ValueError(
            f"Requested {args.count} IDs from {len(complete)} complete candidates"
        )

    selected = sorted(random.Random(args.seed).sample(complete, args.count))
    manifest_text = "\n".join(selected) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(manifest_text, encoding="utf-8")

    selected_set = set(selected)
    stats = {
        "source_manifests": [
            str(args.dataset_root / f"{name}.txt") for name in args.splits
        ],
        "sampling": "uniform_without_replacement_from_unique_union",
        "seed": args.seed,
        "union_candidates": len(candidates),
        "complete_candidates": len(complete),
        "incomplete_candidates": len(candidates) - len(complete),
        "selected": len(selected),
        "manifest_sha256": hashlib.sha256(manifest_text.encode("utf-8")).hexdigest(),
        "source_counts": {name: len(values) for name, values in split_ids.items()},
        "selected_membership_counts": {
            name: len(selected_set & values) for name, values in split_ids.items()
        },
    }
    stats_path = args.output.with_suffix(".stats.json")
    stats_path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(selected)} IDs to {args.output}")
    print(json.dumps(stats["selected_membership_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
