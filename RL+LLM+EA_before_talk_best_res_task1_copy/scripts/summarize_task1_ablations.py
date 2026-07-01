#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DETAIL_RE = re.compile(r"^\s*([^:]+):\s*(.+?)\s*;?\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Task 1 ablation YAML results into CSV and Markdown."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=ROOT / "main" / "molleo_multi" / "ablation_results_task1",
        help="Root created by scripts/run_task1_ablations.py.",
    )
    parser.add_argument(
        "--aggregate",
        default="sum",
        choices=["sum", "pareto"],
        help="Aggregate directory to summarize.",
    )
    parser.add_argument(
        "--mol-lm",
        default="BioT5",
        help="Backbone directory to summarize.",
    )
    parser.add_argument(
        "--max-oracle-calls",
        type=int,
        default=10000,
        help="Budget used for AUC normalization.",
    )
    parser.add_argument(
        "--freq-log",
        type=int,
        default=100,
        help="AUC integration frequency.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Top-N used for AUC and final score.",
    )
    parser.add_argument(
        "--bounds",
        nargs="*",
        default=["jnk3=0,1", "qed=0,1", "sa=1,10"],
        help="Normalized-HV bounds as objective=low,high.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="CSV output path. Defaults under results-root.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Markdown output path. Defaults under results-root.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if isinstance(data, dict) else {}


def parse_detail_items(detail_items: Iterable[object]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for item in detail_items:
        if isinstance(item, dict):
            for key, value in item.items():
                out[str(key).strip().lower()] = float(value)
            continue
        match = DETAIL_RE.match(str(item).strip())
        if not match:
            continue
        key, raw_value = match.groups()
        out[key.strip().lower()] = float(raw_value)
    return out


def objective_value(details: Dict[str, float], name: str) -> float | None:
    key = name.lower()
    candidates = [key, f"{key}_current", f"{key}_score"]
    if key.endswith("_current"):
        candidates.append(key[: -len("_current")])
    for candidate in candidates:
        if candidate in details:
            return details[candidate]
    return None


def yaml_rows(path: Path) -> List[dict]:
    raw = load_yaml(path)
    rows = []
    for smiles, payload in raw.items():
        if not isinstance(payload, list) or len(payload) < 3:
            continue
        details_raw = payload[1]
        if not isinstance(details_raw, list):
            details_raw = []
        try:
            total_score = float(payload[0])
            call_index = int(payload[2])
        except (TypeError, ValueError):
            continue
        rows.append(
            {
                "smiles": smiles,
                "total_score": total_score,
                "details": parse_detail_items(details_raw),
                "call_index": call_index,
            }
        )
    rows.sort(key=lambda row: row["call_index"])
    return rows


def top_auc(rows: Sequence[dict], top_n: int, freq_log: int, max_oracle_calls: int) -> float:
    if not rows:
        return float("nan")
    ordered = sorted(rows, key=lambda row: row["call_index"])
    score_sum = 0.0
    prev = 0.0
    called = 0
    max_count = min(len(ordered), max_oracle_calls)
    for idx in range(freq_log, max_count, freq_log):
        window = sorted(ordered[:idx], key=lambda row: row["total_score"], reverse=True)[:top_n]
        now = float(np.mean([row["total_score"] for row in window]))
        score_sum += freq_log * (now + prev) / 2.0
        prev = now
        called = idx
    window = sorted(ordered[:max_count], key=lambda row: row["total_score"], reverse=True)[:top_n]
    now = float(np.mean([row["total_score"] for row in window]))
    score_sum += (max_count - called) * (now + prev) / 2.0
    if max_count < max_oracle_calls:
        score_sum += (max_oracle_calls - max_count) * now
    return score_sum / float(max_oracle_calls)


def final_top(rows: Sequence[dict], top_n: int) -> float:
    if not rows:
        return float("nan")
    top_rows = sorted(rows, key=lambda row: row["total_score"], reverse=True)[:top_n]
    return float(np.mean([row["total_score"] for row in top_rows]))


def parse_bounds(pairs: Sequence[str]) -> Dict[str, Tuple[float, float]]:
    bounds = {}
    for pair in pairs:
        name, values = pair.split("=", 1)
        low, high = values.split(",", 1)
        bounds[name.lower()] = (float(low), float(high))
    return bounds


def normalize_points(rows: Sequence[dict], bounds: Dict[str, Tuple[float, float]]) -> np.ndarray:
    points = []
    for row in rows:
        jnk3 = objective_value(row["details"], "jnk3")
        qed = objective_value(row["details"], "qed")
        sa = objective_value(row["details"], "sa")
        if jnk3 is None or qed is None or sa is None:
            continue
        raw = {"jnk3": jnk3, "qed": qed, "sa": sa}
        values = []
        for name, maximize in [("jnk3", True), ("qed", True), ("sa", False)]:
            low, high = bounds[name]
            if math.isclose(high, low, rel_tol=0.0, abs_tol=1e-15):
                values.append(1.0)
                continue
            if maximize:
                val = (raw[name] - low) / (high - low)
            else:
                val = (high - raw[name]) / (high - low)
            values.append(float(np.clip(val, 0.0, 1.0)))
        points.append(values)
    if not points:
        return np.empty((0, 3), dtype=float)
    return np.asarray(points, dtype=float)


def pareto_front_indices(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.array([], dtype=int)
    unique_points, unique_indices = np.unique(points, axis=0, return_index=True)
    keep = np.ones(unique_points.shape[0], dtype=bool)
    for index, current in enumerate(unique_points):
        dominated = np.all(unique_points >= current, axis=1) & np.any(unique_points > current, axis=1)
        if np.any(dominated):
            keep[index] = False
    return np.sort(unique_indices[keep])


def hypervolume_2d(boxes: np.ndarray) -> float:
    if len(boxes) == 0:
        return 0.0
    ordered = boxes[np.argsort(boxes[:, 1])]
    area = 0.0
    previous_height = 0.0
    start = 0
    while start < len(ordered):
        height = float(ordered[start, 1])
        width = float(np.max(ordered[start:, 0]))
        area += width * (height - previous_height)
        previous_height = height
        while start < len(ordered) and math.isclose(float(ordered[start, 1]), height, abs_tol=1e-15):
            start += 1
    return float(area)


def hypervolume_recursive(boxes: np.ndarray) -> float:
    if len(boxes) == 0:
        return 0.0
    if boxes.shape[1] == 1:
        return float(np.max(boxes[:, 0]))
    if boxes.shape[1] == 2:
        return hypervolume_2d(boxes)
    ordered = boxes[np.argsort(boxes[:, -1])]
    volume = 0.0
    previous_height = 0.0
    start = 0
    while start < len(ordered):
        height = float(ordered[start, -1])
        volume += hypervolume_recursive(ordered[start:, :-1]) * (height - previous_height)
        previous_height = height
        while start < len(ordered) and math.isclose(float(ordered[start, -1]), height, abs_tol=1e-15):
            start += 1
    return float(volume)


def normalized_hv(rows: Sequence[dict], bounds: Dict[str, Tuple[float, float]]) -> float:
    points = normalize_points(rows, bounds)
    if points.size == 0:
        return 0.0
    points = points[np.all(points > 0.0, axis=1)]
    if points.size == 0:
        return 0.0
    front = points[pareto_front_indices(points)]
    return hypervolume_recursive(front)


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0


def discover_result_files(base: Path) -> List[Path]:
    return sorted(path for path in base.glob("seed*/results_*.yaml") if path.is_file())


def main() -> int:
    args = parse_args()
    base = args.results_root / args.aggregate / args.mol_lm
    bounds = parse_bounds(args.bounds)
    if args.output_csv is None:
        args.output_csv = args.results_root / f"summary_{args.aggregate}_{args.mol_lm}.csv"
    if args.output_md is None:
        args.output_md = args.results_root / f"summary_{args.aggregate}_{args.mol_lm}.md"

    rows_out = []
    for ablation_dir in sorted(path for path in base.iterdir() if path.is_dir()) if base.exists() else []:
        seed_files = discover_result_files(ablation_dir)
        aucs, finals, hvs = [], [], []
        seeds = []
        for path in seed_files:
            rows = yaml_rows(path)
            aucs.append(top_auc(rows, args.top_n, args.freq_log, args.max_oracle_calls))
            finals.append(final_top(rows, args.top_n))
            hvs.append(normalized_hv(rows, bounds))
            seed_name = path.parent.name
            seeds.append(seed_name.replace("seed", ""))
            rows_out.append(
                {
                    "ablation": ablation_dir.name,
                    "seed": seed_name.replace("seed", ""),
                    "path": str(path),
                    "auc_top10": aucs[-1],
                    "final_top10": finals[-1],
                    "normalized_hv": hvs[-1],
                }
            )
        if seed_files:
            auc_mean, auc_std = mean_std(aucs)
            final_mean, final_std = mean_std(finals)
            hv_mean, hv_std = mean_std(hvs)
            rows_out.append(
                {
                    "ablation": ablation_dir.name,
                    "seed": "mean",
                    "path": "",
                    "auc_top10": auc_mean,
                    "final_top10": final_mean,
                    "normalized_hv": hv_mean,
                    "auc_top10_std": auc_std,
                    "final_top10_std": final_std,
                    "normalized_hv_std": hv_std,
                    "n": len(seed_files),
                }
            )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ablation",
        "seed",
        "n",
        "auc_top10",
        "auc_top10_std",
        "final_top10",
        "final_top10_std",
        "normalized_hv",
        "normalized_hv_std",
        "path",
    ]
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_out:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    mean_rows = [row for row in rows_out if row.get("seed") == "mean"]
    mean_rows.sort(key=lambda row: row["auc_top10"], reverse=True)
    with args.output_md.open("w", encoding="utf-8") as handle:
        handle.write(f"# Task 1 Ablation Summary ({args.aggregate}, {args.mol_lm})\n\n")
        handle.write("| Ablation | n | Top-10 AUC | Final Top-10 | Normalized HV |\n")
        handle.write("|---|---:|---:|---:|---:|\n")
        for row in mean_rows:
            handle.write(
                "| {ablation} | {n} | {auc:.4f} +/- {auc_std:.4f} | "
                "{final:.4f} +/- {final_std:.4f} | {hv:.4f} +/- {hv_std:.4f} |\n".format(
                    ablation=row["ablation"],
                    n=row.get("n", ""),
                    auc=row["auc_top10"],
                    auc_std=row.get("auc_top10_std", float("nan")),
                    final=row["final_top10"],
                    final_std=row.get("final_top10_std", float("nan")),
                    hv=row["normalized_hv"],
                    hv_std=row.get("normalized_hv_std", float("nan")),
                )
            )

    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
