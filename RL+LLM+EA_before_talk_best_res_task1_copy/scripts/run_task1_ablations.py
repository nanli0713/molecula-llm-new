#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
DEFAULT_PYTHON = Path("/root/miniconda3/envs/temp/bin/python")
DEFAULT_ABLATIONS = [
    "full",
    "no_rl",
    "random_strategy",
    "no_dkb",
    "no_positive_memory",
    "no_negative_memory",
    "static_dkb",
    "no_incremental_retrain",
    "no_history_prompt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Task 1 MOL-E3 ablation experiments with isolated outputs."
    )
    parser.add_argument(
        "--ablations",
        nargs="+",
        default=DEFAULT_ABLATIONS,
        help="Ablation preset names passed to run.py --ablation.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Seeds to run. Each seed gets an isolated output/model-update directory.",
    )
    parser.add_argument(
        "--mol-lm",
        default="BioT5",
        choices=["BioT5", "GPT-4"],
        help="Backbone generator to use.",
    )
    parser.add_argument(
        "--aggregate",
        default="sum",
        choices=["sum", "pareto"],
        help="Use molleo_multi for Sum or molleo_multi_pareto for Pareto aggregation.",
    )
    parser.add_argument(
        "--max-oracle-calls",
        type=int,
        default=10000,
        help="Oracle budget passed through to run.py.",
    )
    parser.add_argument(
        "--freq-log",
        type=int,
        default=100,
        help="Logging interval passed through to run.py.",
    )
    parser.add_argument(
        "--bin-size",
        type=int,
        default=100,
        help="BioT5 supplement bin size passed through to run.py.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early-stopping patience passed through to run.py.",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable),
        help="Python executable used to run experiments.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "main" / "molleo_multi" / "ablation_results_task1",
        help="Root directory for result YAML files.",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=ROOT / "logs" / "ablations_task1",
        help="Root directory for stdout/stderr logs.",
    )
    parser.add_argument(
        "--model-cache-root",
        type=Path,
        default=ROOT / "main" / "molleo_multi" / "ablation_model_cache_task1",
        help="Root directory for per-run incremental model checkpoints.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs whose expected result YAML already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional arguments appended to run.py after '--'.",
    )
    return parser.parse_args()


def expected_result_path(output_dir: Path, mol_lm: str, seed: int) -> Path:
    suffix = f"results_{mol_lm}_['jnk3', 'qed']_['sa']{seed}.yaml"
    return output_dir / suffix


def copy_initial_checkpoints(model_dir: Path, data_dir: Path) -> tuple[Path, Path]:
    model_dir.mkdir(parents=True, exist_ok=True)
    sgatt_add = model_dir / "sgatt_add.pth"
    predictor_add = model_dir / "predictor_add.pth"
    if not sgatt_add.exists():
        shutil.copy2(data_dir / "sgatt_init.pth", sgatt_add)
    if not predictor_add.exists():
        shutil.copy2(data_dir / "predictor_init.pth", predictor_add)
    return sgatt_add, predictor_add


def build_env(model_dir: Path, data_dir: Path) -> dict[str, str]:
    sgatt_add, predictor_add = copy_initial_checkpoints(model_dir, data_dir)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        item for item in [str(ROOT), str(WORKSPACE), env.get("PYTHONPATH", "")] if item
    )
    env["HF_ENDPOINT"] = env.get("HF_ENDPOINT", "https://hf-mirror.com")
    env["CSV_PATH_URL"] = str(data_dir / "selected_molecules_task1.csv")
    env["MODEL_INIT_PATH_URL"] = str(data_dir / "sgatt_init.pth")
    env["MODEL_INIT_PREDICTOR_PATH_URL"] = str(data_dir / "predictor_init.pth")
    env["MODEL_ADD_PATH_URL"] = str(sgatt_add)
    env["MODEL_ADD_PREDICTOR_PATH_URL"] = str(predictor_add)
    env["KG_TRIPLES_EMB_PATH"] = str(data_dir / "total_kg_triples_emb.pkl")
    return env


def command_for(args: argparse.Namespace, ablation: str, seed: int, output_dir: Path) -> List[str]:
    method = "molleo_multi" if args.aggregate == "sum" else "molleo_multi_pareto"
    cmd = [
        str(args.python),
        "run.py",
        method,
        "--mol_lm",
        args.mol_lm,
        "--max_obj",
        "jnk3",
        "qed",
        "--min_obj",
        "sa",
        "--seed",
        str(seed),
        "--ablation",
        ablation,
        "--output_dir",
        str(output_dir),
        "--max_oracle_calls",
        str(args.max_oracle_calls),
        "--freq_log",
        str(args.freq_log),
        "--bin_size",
        str(args.bin_size),
        "--patience",
        str(args.patience),
    ]
    cmd.extend(args.extra_args)
    return cmd


def main() -> int:
    args = parse_args()
    data_dir = ROOT / "main" / "molleo_multi" / "datas_task1"
    if not data_dir.exists():
        raise FileNotFoundError(f"Task 1 data directory not found: {data_dir}")

    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)
        args.log_root.mkdir(parents=True, exist_ok=True)
        args.model_cache_root.mkdir(parents=True, exist_ok=True)

    for ablation in args.ablations:
        for seed in args.seeds:
            run_name = f"{args.aggregate}_{args.mol_lm}_{ablation}_seed{seed}"
            output_dir = args.output_root / args.aggregate / args.mol_lm / ablation / f"seed{seed}"
            model_dir = args.model_cache_root / args.aggregate / args.mol_lm / ablation / f"seed{seed}"
            log_path = args.log_root / f"{run_name}.log"

            result_path = expected_result_path(output_dir, args.mol_lm, seed)
            if args.resume and result_path.exists():
                print(f"[skip] {run_name}: {result_path}")
                continue

            cmd = command_for(args, ablation, seed, output_dir)
            print("[run]", " ".join(cmd))
            print("[log]", log_path)
            if args.dry_run:
                continue

            output_dir.mkdir(parents=True, exist_ok=True)
            env = build_env(model_dir=model_dir, data_dir=data_dir)
            with log_path.open("w", encoding="utf-8") as log_file:
                proc = subprocess.run(
                    cmd,
                    cwd=ROOT,
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            if proc.returncode != 0:
                print(f"[fail] {run_name}: return code {proc.returncode}; see {log_path}", file=sys.stderr)
                return proc.returncode
            print(f"[done] {run_name}: {result_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
