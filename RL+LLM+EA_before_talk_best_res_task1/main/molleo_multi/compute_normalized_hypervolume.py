#!/usr/bin/env python3
"""Compute normalized hypervolume (HV in [0, 1]) for MolLEO YAML result files.

This script is meant for result files with a structure like:

SMILES:
- total_score
- - 'jnk3_current: 0.64'
  - 'qed: 0.8046256767555618'
  - 'sa: 2.393713577977886;'
- step_id

Why a "normalized" HV script?
1. Raw hypervolume depends on the original objective scales, so it can be larger
   than 1.0.
2. If every objective is normalized to [0, 1], minimization objectives are
   flipped into maximization form, and the reference point is fixed at
   (0, ..., 0), then the hypervolume is guaranteed to be in [0, 1].

This script:
- reads all YAML/YML files from a directory
- infers max/min objectives from filenames such as
  results_GPT-4_['jnk3', 'qed']_['sa']1.yaml
  or accepts them from --max-obj / --min-obj
- normalizes each objective to [0, 1] using pooled bounds from the directory
  or explicit bounds passed via --bounds
- computes an exact hypervolume on the normalized Pareto front
- prints each file's HV, plus mean / variance across files

Example:
    python compute_normalized_hypervolume.py \
        --results-dir ./results

    python compute_normalized_hypervolume.py \
        --results-dir ./results \
        --max-obj jnk3 qed \
        --min-obj sa \
        --bounds jnk3=0,1 qed=0,1 sa=1,10

Important:
If you want to compare different methods/directories fairly, use the same
normalization bounds for all of them via --bounds.
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
        description="Compute normalized hypervolume from MolLEO YAML results."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directory containing YAML result files.",
    )
    parser.add_argument(
        "--pattern",
        default=None,
        help=(
            "Optional glob pattern used inside --results-dir. "
            "If omitted, both *.yaml and *.yml are loaded."
        ),
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
        "--bounds",
        nargs="*",
        default=None,
        help=(
            "Normalization bounds as name=low,high. "
            "Example: --bounds jnk3=0,1 qed=0,1 sa=1,10"
        ),
    )
    parser.add_argument(
        "--ddof",
        type=int,
        default=0,
        help=(
            "Delta degrees of freedom used by variance/std. "
            "Use 0 for population variance, 1 for sample variance."
        ),
    )
    parser.add_argument(
        "--print-pareto",
        action="store_true",
        help="Print the merged normalized Pareto front.",
    )
    return parser.parse_args()


def discover_files(results_dir: Path, pattern: str | None) -> List[Path]:
    if pattern:
        return sorted(
            (path for path in results_dir.glob(pattern) if path.is_file()),
            key=lambda path: str(path),
        )

    yaml_files = list(results_dir.glob("*.yaml"))
    yml_files = list(results_dir.glob("*.yml"))
    return sorted({*yaml_files, *yml_files}, key=lambda path: str(path))


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

        rows.append(
            {
                "smiles": smiles,
                "total_score": payload[0] if len(payload) > 0 else None,
                "step": payload[2] if len(payload) > 2 else None,
                "details": parse_detail_items(details),
            }
        )

    return rows


def parse_objectives_from_filename(path: Path) -> Tuple[List[str], List[str]]:
    match = FILE_OBJECTIVE_RE.search(path.name)
    if not match:
        raise ValueError(
            f"Cannot infer objective directions from filename: {path.name}. "
            "Please pass --max-obj/--min-obj explicitly."
        )

    max_obj = ast.literal_eval(match.group(1))
    min_obj = ast.literal_eval(match.group(2))
    if not isinstance(max_obj, list) or not isinstance(min_obj, list):
        raise ValueError(f"Objective lists parsed from {path.name} are invalid.")

    return [str(item) for item in max_obj], [str(item) for item in min_obj]


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


def collect_available_keys(rows: Sequence[Dict[str, object]]) -> List[str]:
    keys = set()
    for row in rows:
        details = row["details"]
        if isinstance(details, dict):
            keys.update(details.keys())
    return sorted(keys)


def resolve_objective_order(
    rows: Sequence[Dict[str, object]],
    requested_max_obj: Sequence[str],
    requested_min_obj: Sequence[str],
) -> Tuple[List[str], List[str]]:
    available_keys = collect_available_keys(rows)
    resolved_max = [
        resolve_objective_name(name, available_keys) for name in requested_max_obj
    ]
    resolved_min = [
        resolve_objective_name(name, available_keys) for name in requested_min_obj
    ]
    return resolved_max, resolved_min


def extract_points(
    rows: Sequence[Dict[str, object]],
    objective_order: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    if not rows:
        return np.empty((0, len(objective_order)), dtype=float), []

    points: List[List[float]] = []
    smiles: List[str] = []

    for row in rows:
        details = row["details"]
        if not isinstance(details, dict):
            continue

        if not all(name in details for name in objective_order):
            continue

        try:
            point = [float(details[name]) for name in objective_order]
        except (TypeError, ValueError):
            continue

        points.append(point)
        smiles.append(str(row["smiles"]))

    if not points:
        return np.empty((0, len(objective_order)), dtype=float), []

    return np.asarray(points, dtype=float), smiles


def parse_bounds(
    pairs: Sequence[str] | None,
) -> Dict[str, Tuple[float, float]]:
    if not pairs:
        return {}

    bounds: Dict[str, Tuple[float, float]] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(
                f"Invalid --bounds item '{pair}'. Expected name=low,high."
            )

        name, raw_range = pair.split("=", 1)
        if "," not in raw_range:
            raise ValueError(
                f"Invalid --bounds item '{pair}'. Expected name=low,high."
            )

        raw_low, raw_high = raw_range.split(",", 1)
        low = float(raw_low)
        high = float(raw_high)
        if high < low:
            raise ValueError(
                f"Invalid bounds for '{name}': high ({high}) must be >= low ({low})."
            )

        bounds[normalize_key(name)] = (low, high)

    return bounds


def build_bounds(
    all_points: np.ndarray,
    objective_order: Sequence[str],
    explicit_bounds: Dict[str, Tuple[float, float]],
) -> Tuple[List[Tuple[float, float]], List[str]]:
    if all_points.size == 0:
        raise ValueError("Cannot build normalization bounds from an empty dataset.")

    bounds: List[Tuple[float, float]] = []
    sources: List[str] = []

    for index, objective_name in enumerate(objective_order):
        chosen = None
        for candidate in build_candidate_keys(objective_name):
            if candidate in explicit_bounds:
                chosen = explicit_bounds[candidate]
                sources.append("explicit")
                break

        if chosen is None:
            column = all_points[:, index]
            chosen = (float(np.min(column)), float(np.max(column)))
            sources.append("inferred")

        bounds.append(chosen)

    return bounds, sources


def normalize_points(
    points: np.ndarray,
    maximize_mask: Sequence[bool],
    bounds: Sequence[Tuple[float, float]],
) -> np.ndarray:
    if points.size == 0:
        return points.copy()

    normalized = np.empty_like(points, dtype=float)
    for index, (maximize, (low, high)) in enumerate(zip(maximize_mask, bounds)):
        column = points[:, index]
        if math.isclose(high, low, rel_tol=0.0, abs_tol=1e-15):
            normalized[:, index] = 1.0
            continue

        span = high - low
        if maximize:
            values = (column - low) / span
        else:
            values = (high - column) / span

        normalized[:, index] = np.clip(values, 0.0, 1.0)

    return normalized


def filter_points_above_reference(
    normalized_points: np.ndarray,
    smiles: Sequence[str],
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    if normalized_points.size == 0:
        return normalized_points.copy(), [], np.zeros((0,), dtype=bool)

    mask = np.all(normalized_points > 0.0, axis=1)
    kept_points = normalized_points[mask]
    kept_smiles = [smiles[index] for index, keep in enumerate(mask) if keep]
    return kept_points, kept_smiles, mask


def pareto_front_indices(points: np.ndarray) -> np.ndarray:
    """Return indices of non-dominated points for a maximization problem."""
    if len(points) == 0:
        return np.array([], dtype=int)

    unique_points, unique_indices = np.unique(points, axis=0, return_index=True)
    keep = np.ones(unique_points.shape[0], dtype=bool)

    for index in range(unique_points.shape[0]):
        current = unique_points[index]
        dominates_current = np.all(unique_points >= current, axis=1) & np.any(
            unique_points > current, axis=1
        )
        if np.any(dominates_current):
            keep[index] = False

    return np.sort(unique_indices[keep])


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


def compute_hypervolume(normalized_points: np.ndarray) -> Tuple[float, np.ndarray]:
    if normalized_points.size == 0:
        return 0.0, np.array([], dtype=int)

    front_idx = pareto_front_indices(normalized_points)
    front = normalized_points[front_idx]
    return hypervolume_recursive(front), front_idx


def format_bounds(
    objective_order: Sequence[str],
    maximize_mask: Sequence[bool],
    bounds: Sequence[Tuple[float, float]],
    sources: Sequence[str],
) -> List[str]:
    lines = []
    for name, maximize, (low, high), source in zip(
        objective_order, maximize_mask, bounds, sources
    ):
        direction = "max" if maximize else "min"
        lines.append(
            f"  - {name} ({direction}): raw_low={low:.6f}, raw_high={high:.6f}, source={source}"
        )
    return lines


def main() -> None:
    args = parse_args()

    results_dir = args.results_dir.resolve()
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    files = discover_files(results_dir, args.pattern)
    if not files:
        raise FileNotFoundError(f"No YAML/YML files were found in {results_dir}")

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

    resolved_max_obj, resolved_min_obj = resolve_objective_order(
        rows=all_rows,
        requested_max_obj=requested_max_obj,
        requested_min_obj=requested_min_obj,
    )
    objective_order = resolved_max_obj + resolved_min_obj
    maximize_mask = [True] * len(resolved_max_obj) + [False] * len(resolved_min_obj)

    all_points_raw, _ = extract_points(all_rows, objective_order)
    if all_points_raw.size == 0:
        raise ValueError("No valid objective vectors were extracted from the YAML files.")

    explicit_bounds = parse_bounds(args.bounds)
    bounds, bound_sources = build_bounds(
        all_points=all_points_raw,
        objective_order=objective_order,
        explicit_bounds=explicit_bounds,
    )

    print(f"Results dir: {results_dir}")
    print(f"Files: {len(files)}")
    print(f"Requested max objectives: {requested_max_obj}")
    print(f"Requested min objectives: {requested_min_obj}")
    print(f"Matched objective fields: {objective_order}")
    print("Normalization bounds:")
    for line in format_bounds(objective_order, maximize_mask, bounds, bound_sources):
        print(line)
    print("Reference point after normalization: (0, ..., 0)")
    print("Normalized ideal point: (1, ..., 1)")
    print("")

    per_file_hv: List[float] = []
    merged_points: List[np.ndarray] = []
    merged_raw_points: List[np.ndarray] = []
    merged_smiles: List[str] = []

    for path in files:
        raw_points, smiles = extract_points(file_rows[path], objective_order)
        normalized_points = normalize_points(
            points=raw_points,
            maximize_mask=maximize_mask,
            bounds=bounds,
        )
        filtered_points, filtered_smiles, keep_mask = filter_points_above_reference(
            normalized_points,
            smiles,
        )
        filtered_raw_points = raw_points[keep_mask]

        hv, front_idx = compute_hypervolume(filtered_points)
        per_file_hv.append(hv)

        if len(filtered_points):
            merged_points.append(filtered_points)
            merged_raw_points.append(filtered_raw_points)
            merged_smiles.extend(filtered_smiles)

        print(
            f"{path.name}: total={len(raw_points)}, "
            f"above_ref={len(filtered_points)}, "
            f"pareto={len(front_idx)}, "
            f"normalized_hv={hv:.10f}"
        )

    hv_values = np.asarray(per_file_hv, dtype=float)
    if len(hv_values) <= args.ddof:
        variance = float("nan")
        std = float("nan")
    else:
        variance = float(np.var(hv_values, ddof=args.ddof))
        std = float(np.std(hv_values, ddof=args.ddof))

    if merged_points:
        merged_points_array = np.vstack(merged_points)
        merged_raw_array = np.vstack(merged_raw_points)
    else:
        merged_points_array = np.empty((0, len(objective_order)), dtype=float)
        merged_raw_array = np.empty((0, len(objective_order)), dtype=float)

    merged_hv, merged_front_idx = compute_hypervolume(merged_points_array)

    print("")
    print(
        f"Merged normalized_hv={merged_hv:.10f}, "
        f"merged_pareto={len(merged_front_idx)}"
    )
    print(f"Mean normalized_hv={float(np.mean(hv_values)):.10f}")
    print(f"Variance(ddof={args.ddof})={variance:.10f}")
    print(f"Std(ddof={args.ddof})={std:.10f}")

    if args.print_pareto and len(merged_front_idx):
        print("")
        print("Merged Pareto points:")
        for index in merged_front_idx:
            normalized_values = ", ".join(
                f"{name}={value:.6f}"
                for name, value in zip(objective_order, merged_points_array[index])
            )
            raw_values = ", ".join(
                f"{name}={value:.6f}"
                for name, value in zip(objective_order, merged_raw_array[index])
            )
            print(
                f"  {merged_smiles[index]} | normalized[{normalized_values}] | raw[{raw_values}]"
            )


if __name__ == "__main__":
    main()
