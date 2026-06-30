#!/usr/bin/env python3
"""Compute hypervolume for MolLEO multi-objective YAML result files.

This script expects result files with a structure like:

SMILES:
- total_score
- - 'jnk3_current: 0.6'
  - 'qed: 0.9059'
  - 'sa: 2.4877;'
- step_id

Example:
    python compute_hypervolume.py \
        --results-dir ./results \
        --max-obj jnk3 qed \
        --min-obj sa \
        --ref jnk3=0 qed=0 sa=10

Notes:
1. Hypervolume depends on the reference point. For cross-run comparisons, use
   the same reference point for every method/seed.
2. If `--ref` is omitted, the script derives an automatic reference point from
   the observed data. That is convenient, but not suitable for fair comparison
   across different experiments unless computed on the same pooled dataset.
"""

from __future__ import annotations

import argparse
import ast
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import yaml


DETAIL_RE = re.compile(r"^\s*([^:]+):\s*(.+?)\s*;?\s*$")
FILE_OBJECTIVE_RE = re.compile(r"_(\[[^\]]*\])_(\[[^\]]*\])(?:\d+)?\.ya?ml$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute hypervolume from MolLEO multi-objective YAML results."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directory containing YAML result files.",
    )
    parser.add_argument(
        "--pattern",
        default="*.yaml",
        help="Glob pattern used to select result files inside --results-dir.",
    )
    parser.add_argument(
        "--max-obj",
        nargs="*",
        default=None,
        help="Objectives to maximize, e.g. --max-obj jnk3 qed",
    )
    parser.add_argument(
        "--min-obj",
        nargs="*",
        default=None,
        help="Objectives to minimize, e.g. --min-obj sa",
    )
    parser.add_argument(
        "--ref",
        nargs="*",
        default=None,
        help="Reference point as name=value pairs, e.g. --ref jnk3=0 qed=0 sa=10",
    )
    parser.add_argument(
        "--auto-ref-margin",
        type=float,
        default=0.0,
        help=(
            "Margin used when auto-building the reference point from observed data. "
            "For maximization objectives, the auto reference is min - margin*span; "
            "for minimization objectives, it is max + margin*span."
        ),
    )
    parser.add_argument(
        "--print-pareto",
        action="store_true",
        help="Print the non-dominated points used in the merged hypervolume.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain the expected mapping structure.")
    return data


def parse_detail_items(detail_items: Iterable[object]) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for item in detail_items:
        if isinstance(item, dict):
            for key, value in item.items():
                values[str(key).strip()] = float(value)
            continue

        text = str(item).strip()
        match = DETAIL_RE.match(text)
        if not match:
            continue

        key, raw_value = match.groups()
        values[key.strip()] = float(raw_value)
    return values


def parse_result_file(path: Path) -> List[Dict[str, object]]:
    raw = load_yaml(path)
    rows: List[Dict[str, object]] = []

    for smiles, payload in raw.items():
        if not isinstance(payload, list) or len(payload) < 2:
            continue
        details = payload[1]
        if not isinstance(details, list):
            continue

        row = {
            "smiles": smiles,
            "total_score": payload[0] if len(payload) > 0 else None,
            "step": payload[2] if len(payload) > 2 else None,
            "details": parse_detail_items(details),
        }
        rows.append(row)

    return rows


def parse_objectives_from_filename(path: Path) -> Tuple[List[str], List[str]]:
    match = FILE_OBJECTIVE_RE.search(path.name)
    if not match:
        raise ValueError(
            f"Cannot infer objectives from filename: {path.name}. "
            "Please pass --max-obj/--min-obj explicitly."
        )

    max_obj = ast.literal_eval(match.group(1))
    min_obj = ast.literal_eval(match.group(2))
    if not isinstance(max_obj, list) or not isinstance(min_obj, list):
        raise ValueError(f"Objective lists parsed from {path.name} are invalid.")
    return [str(x) for x in max_obj], [str(x) for x in min_obj]


def normalize_key(name: str) -> str:
    return name.strip().lower()


def build_candidate_keys(name: str) -> List[str]:
    base = normalize_key(name)
    candidates = [base]

    if not base.endswith("_current"):
        candidates.append(f"{base}_current")
    if not base.endswith("_score"):
        candidates.append(f"{base}_score")
    if base.endswith("_current"):
        candidates.append(base[: -len("_current")])
    if base.endswith("_score"):
        candidates.append(base[: -len("_score")])

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(candidates))


def resolve_objective_name(requested: str, available_keys: Iterable[str]) -> str:
    available_map = {normalize_key(key): key for key in available_keys}
    for candidate in build_candidate_keys(requested):
        if candidate in available_map:
            return available_map[candidate]

    raise KeyError(
        f"Objective '{requested}' was not found in YAML details. "
        f"Available keys: {sorted(available_keys)}"
    )


def parse_reference_pairs(pairs: Sequence[str] | None) -> Dict[str, float]:
    if not pairs:
        return {}

    ref: Dict[str, float] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(
                f"Invalid --ref item '{pair}'. Expected the form name=value."
            )
        name, raw_value = pair.split("=", 1)
        ref[normalize_key(name)] = float(raw_value)
    return ref


def objective_config(
    files: Sequence[Path],
    cli_max_obj: Sequence[str] | None,
    cli_min_obj: Sequence[str] | None,
) -> Tuple[List[str], List[str]]:
    if cli_max_obj is not None or cli_min_obj is not None:
        max_obj = list(cli_max_obj or [])
        min_obj = list(cli_min_obj or [])
        if not max_obj and not min_obj:
            raise ValueError("At least one objective must be provided.")
        return max_obj, min_obj

    if not files:
        raise ValueError("No YAML files were found.")

    return parse_objectives_from_filename(files[0])


def extract_points(
    rows: Sequence[Dict[str, object]],
    requested_max_obj: Sequence[str],
    requested_min_obj: Sequence[str],
) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
    if not rows:
        return np.empty((0, 0), dtype=float), [], [], []

    available_keys = set()
    for row in rows:
        available_keys.update(row["details"].keys())  # type: ignore[arg-type]

    resolved_max = [
        resolve_objective_name(name, available_keys) for name in requested_max_obj
    ]
    resolved_min = [
        resolve_objective_name(name, available_keys) for name in requested_min_obj
    ]
    resolved_order = resolved_max + resolved_min

    points: List[List[float]] = []
    smiles: List[str] = []
    for row in rows:
        detail_map = row["details"]  # type: ignore[assignment]
        if not all(key in detail_map for key in resolved_order):
            continue

        point = [float(detail_map[key]) for key in resolved_order]
        points.append(point)
        smiles.append(str(row["smiles"]))

    if not points:
        return np.empty((0, len(resolved_order)), dtype=float), resolved_max, resolved_min, []

    return (
        np.asarray(points, dtype=float),
        resolved_max,
        resolved_min,
        smiles,
    )


def transform_to_maximization(
    points: np.ndarray,
    min_obj_count: int,
) -> np.ndarray:
    if points.size == 0:
        return points.copy()

    transformed = points.copy()
    if min_obj_count:
        transformed[:, -min_obj_count:] *= -1.0
    return transformed


def build_reference_point(
    points: np.ndarray,
    objective_order: Sequence[str],
    requested_max_obj: Sequence[str],
    requested_min_obj: Sequence[str],
    ref_pairs: Dict[str, float],
    margin: float,
) -> np.ndarray:
    if points.size == 0:
        raise ValueError("Cannot build a reference point from an empty dataset.")

    ref = np.empty(points.shape[1], dtype=float)
    max_set = {normalize_key(name) for name in requested_max_obj}
    min_set = {normalize_key(name) for name in requested_min_obj}

    for index, objective_name in enumerate(objective_order):
        objective_key = normalize_key(objective_name)
        candidate_names = build_candidate_keys(objective_name)

        explicit = None
        for candidate in candidate_names:
            if candidate in ref_pairs:
                explicit = ref_pairs[candidate]
                break

        if explicit is not None:
            ref[index] = explicit
            continue

        column = points[:, index]
        span = float(column.max() - column.min())
        if span == 0.0:
            span = 1.0

        if objective_key in max_set or any(c in max_set for c in candidate_names):
            ref[index] = float(column.min() - margin * span)
        elif objective_key in min_set or any(c in min_set for c in candidate_names):
            ref[index] = float(column.max() + margin * span)
        else:
            raise KeyError(
                f"Unable to determine optimization direction for objective '{objective_name}'."
            )

    return ref


def reference_mask(
    transformed_points: np.ndarray,
    transformed_ref: np.ndarray,
    ) -> np.ndarray:
    if transformed_points.size == 0:
        return np.zeros((0,), dtype=bool)

    return np.all(transformed_points > transformed_ref, axis=1)


def filter_points_above_reference(
    transformed_points: np.ndarray,
    transformed_ref: np.ndarray,
    smiles: Sequence[str],
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    mask = reference_mask(transformed_points, transformed_ref)
    kept_points = transformed_points[mask]
    kept_smiles = [smiles[idx] for idx, keep in enumerate(mask) if keep]
    return kept_points, kept_smiles, mask


def pareto_front_indices(points: np.ndarray) -> np.ndarray:
    """Return indices of non-dominated points for a maximization problem."""
    if len(points) == 0:
        return np.array([], dtype=int)

    unique_points, unique_indices = np.unique(points, axis=0, return_index=True)
    survivors = np.arange(unique_points.shape[0])
    cursor = 0

    while cursor < len(unique_points):
        current = unique_points[cursor]
        dominated_mask = np.all(unique_points <= current, axis=1) & np.any(
            unique_points < current, axis=1
        )
        keep_mask = ~dominated_mask
        keep_mask[cursor] = True

        survivors = survivors[keep_mask]
        unique_points = unique_points[keep_mask]
        cursor = int(np.sum(keep_mask[:cursor])) + 1

    return unique_indices[survivors]


def hypervolume_2d(boxes: np.ndarray) -> float:
    if len(boxes) == 0:
        return 0.0

    order = np.argsort(boxes[:, 1])
    ordered = boxes[order]

    area = 0.0
    previous_height = 0.0
    start = 0
    total = len(ordered)

    while start < total:
        height = float(ordered[start, 1])
        width = float(np.max(ordered[start:, 0]))
        area += width * (height - previous_height)
        previous_height = height

        while start < total and math.isclose(
            float(ordered[start, 1]), height, rel_tol=0.0, abs_tol=1e-15
        ):
            start += 1

    return float(area)


def hypervolume_recursive(boxes: np.ndarray) -> float:
    """Exact hypervolume for boxes anchored at the origin."""
    if len(boxes) == 0:
        return 0.0

    dimension = boxes.shape[1]
    if dimension == 1:
        return float(np.max(boxes[:, 0]))
    if dimension == 2:
        return hypervolume_2d(boxes)

    order = np.argsort(boxes[:, -1])
    sorted_boxes = boxes[order]

    volume = 0.0
    previous_height = 0.0
    start = 0
    total = len(sorted_boxes)

    while start < total:
        height = float(sorted_boxes[start, -1])
        active_projection = sorted_boxes[start:, :-1]
        base_area = hypervolume_recursive(active_projection)
        volume += base_area * (height - previous_height)
        previous_height = height

        while start < total and math.isclose(
            float(sorted_boxes[start, -1]), height, rel_tol=0.0, abs_tol=1e-15
        ):
            start += 1

    return float(volume)


def compute_hypervolume(
    transformed_points: np.ndarray,
    transformed_ref: np.ndarray,
) -> Tuple[float, np.ndarray]:
    if transformed_points.size == 0:
        return 0.0, np.array([], dtype=int)

    contributing = transformed_points - transformed_ref
    front_idx = pareto_front_indices(contributing)
    front = contributing[front_idx]
    return hypervolume_recursive(front), front_idx


def format_reference(
    objective_order: Sequence[str],
    ref: Sequence[float],
) -> str:
    parts = [f"{name}={value:.6f}" for name, value in zip(objective_order, ref)]
    return ", ".join(parts)


def main() -> None:
    args = parse_args()

    results_dir = args.results_dir.resolve()
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    files = sorted(results_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(
            f"No YAML files matched pattern '{args.pattern}' in {results_dir}"
        )

    requested_max_obj, requested_min_obj = objective_config(
        files=files,
        cli_max_obj=args.max_obj,
        cli_min_obj=args.min_obj,
    )

    file_rows: Dict[Path, List[Dict[str, object]]] = {
        path: parse_result_file(path) for path in files
    }

    all_rows: List[Dict[str, object]] = []
    for rows in file_rows.values():
        all_rows.extend(rows)

    all_points_raw, resolved_max_obj, resolved_min_obj, _ = extract_points(
        all_rows,
        requested_max_obj=requested_max_obj,
        requested_min_obj=requested_min_obj,
    )

    if all_points_raw.size == 0:
        raise ValueError("No valid objective vectors were extracted from the YAML files.")

    objective_order = resolved_max_obj + resolved_min_obj
    ref_pairs = parse_reference_pairs(args.ref)
    ref_raw = build_reference_point(
        points=all_points_raw,
        objective_order=objective_order,
        requested_max_obj=requested_max_obj,
        requested_min_obj=requested_min_obj,
        ref_pairs=ref_pairs,
        margin=args.auto_ref_margin,
    )
    transformed_ref = transform_to_maximization(
        ref_raw.reshape(1, -1), min_obj_count=len(resolved_min_obj)
    )[0]

    print(f"Results dir: {results_dir}")
    print(f"Files: {len(files)}")
    print(f"Max objectives: {requested_max_obj}")
    print(f"Min objectives: {requested_min_obj}")
    print(f"Matched objective fields: {objective_order}")
    print(f"Reference point: {format_reference(objective_order, ref_raw)}")
    print("")

    merged_transformed_all = []
    merged_raw_all = []
    merged_smiles_all = []
    per_file_hv: List[float] = []

    for path in files:
        raw_points, resolved_max, resolved_min, smiles = extract_points(
            file_rows[path],
            requested_max_obj=requested_max_obj,
            requested_min_obj=requested_min_obj,
        )
        if resolved_max != resolved_max_obj or resolved_min != resolved_min_obj:
            raise ValueError(
                f"Objective field mismatch in {path.name}. "
                f"Expected {objective_order}, got {resolved_max + resolved_min}."
            )

        transformed = transform_to_maximization(
            raw_points, min_obj_count=len(resolved_min_obj)
        )
        filtered_points, filtered_smiles, keep_mask = filter_points_above_reference(
            transformed, transformed_ref, smiles
        )
        filtered_raw_points = raw_points[keep_mask]
        hv, front_idx = compute_hypervolume(filtered_points, transformed_ref)
        front_size = len(front_idx)
        total_size = len(filtered_points)
        per_file_hv.append(hv)

        merged_transformed_all.append(filtered_points)
        merged_raw_all.append(filtered_raw_points)
        merged_smiles_all.extend(filtered_smiles)

        print(
            f"{path.name}: "
            f"total={len(raw_points)}, "
            f"above_ref={total_size}, "
            f"pareto={front_size}, "
            f"hv={hv:.10f}"
        )

    merged_points = (
        np.vstack(merged_transformed_all)
        if merged_transformed_all and any(len(x) for x in merged_transformed_all)
        else np.empty((0, len(objective_order)), dtype=float)
    )
    merged_raw_points = (
        np.vstack(merged_raw_all)
        if merged_raw_all and any(len(x) for x in merged_raw_all)
        else np.empty((0, len(objective_order)), dtype=float)
    )

    merged_hv, merged_front_idx = compute_hypervolume(merged_points, transformed_ref)

    print("")
    print(
        f"Merged: total={len(merged_points)}, "
        f"pareto={len(merged_front_idx)}, "
        f"hv={merged_hv:.10f}"
    )
    print(
        f"Per-file mean/std: mean={np.mean(per_file_hv):.10f}, "
        f"std={np.std(per_file_hv):.10f}"
    )

    if args.print_pareto and len(merged_front_idx):
        print("")
        print("Merged Pareto points (original objective values):")
        for idx in merged_front_idx:
            values = ", ".join(
                f"{name}={value:.6f}"
                for name, value in zip(objective_order, merged_raw_points[idx])
            )
            print(f"  {merged_smiles_all[idx]} | {values}")


if __name__ == "__main__":
    main()
