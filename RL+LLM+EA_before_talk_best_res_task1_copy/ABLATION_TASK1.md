# Task 1 Ablation Experiments

This directory now has a small ablation layer for the Task 1 multi-objective experiment in
`/root/autodl-tmp/RL+LLM+EA_before_talk_best_res_task1_copy`.

Task 1 follows the paper setting in `main.tex`:

- maximize `jnk3`
- maximize `qed`
- minimize `sa`

The main ablation implementation is in `main/ablation.py`, and the Task 1 runner is
`scripts/run_task1_ablations.py`.

## Ablation Presets

Recommended presets for the paper table:

| Preset | What It Tests |
|---|---|
| `full` | Complete MOL-E3: DKB + DQN strategy + history prompt + incremental retraining |
| `no_rl` | Replace learned DQN strategy selection with a fixed hybrid strategy |
| `random_strategy` | Replace learned DQN strategy selection with random action scheduling |
| `no_dkb` | Remove high/low fragment knowledge from prompts, state, reward memory, and updates |
| `no_positive_memory` | Remove high-scoring fragment guidance only |
| `no_negative_memory` | Remove low-scoring fragment avoidance only |
| `static_dkb` | Use initial DKB only; disable online DKB refresh and incremental retraining |
| `no_incremental_retrain` | Keep online memory updates but freeze the cross-attention scorer |
| `no_history_prompt` | Remove previous generated high/low molecule feedback from later prompts |

Optional sanity-check baseline:

| Preset | What It Tests |
|---|---|
| `molleo_like` | Memoryless LLM-editing baseline: no DKB, no DQN, no history prompt, no incremental retraining |

List all presets:

```bash
cd /root/autodl-tmp/RL+LLM+EA_before_talk_best_res_task1_copy
python run.py molleo_multi --list_ablations
```

## Dry Run

Print the commands before starting expensive runs:

```bash
cd /root/autodl-tmp/RL+LLM+EA_before_talk_best_res_task1_copy
python scripts/run_task1_ablations.py \
  --dry-run \
  --mol-lm BioT5 \
  --seeds 1 2 3
```

## Run The Recommended BioT5 Ablations

```bash
cd /root/autodl-tmp/RL+LLM+EA_before_talk_best_res_task1_copy
python scripts/run_task1_ablations.py \
  --mol-lm BioT5 \
  --seeds 1 2 3 \
  --max-oracle-calls 10000
```

If you want to force a specific conda/python environment, add for example:

```bash
python scripts/run_task1_ablations.py \
  --python /root/miniconda3/bin/python \
  --mol-lm BioT5 \
  --seeds 1 2 3
```

The runner automatically overrides these paths for the current process, so it uses the
`task1_copy` data and does not require editing `/root/autodl-tmp/.env`:

- `CSV_PATH_URL`
- `MODEL_INIT_PATH_URL`
- `MODEL_INIT_PREDICTOR_PATH_URL`
- `MODEL_ADD_PATH_URL`
- `MODEL_ADD_PREDICTOR_PATH_URL`
- `KG_TRIPLES_EMB_PATH`

Each ablation and seed gets its own incremental checkpoint cache under:

```text
main/molleo_multi/ablation_model_cache_task1/
```

This avoids cross-seed contamination from online LoRA retraining.

## Run A Smaller Smoke Test

Use this first if you only want to confirm the environment:

```bash
cd /root/autodl-tmp/RL+LLM+EA_before_talk_best_res_task1_copy
python scripts/run_task1_ablations.py \
  --mol-lm BioT5 \
  --ablations full no_rl \
  --seeds 1 \
  --max-oracle-calls 300 \
  --freq-log 100
```

## Run GPT-5.2/GPT-4-Style Ablations

In this codebase, the GPT backend is selected with `--mol-lm GPT-4`, while the actual
served model name comes from `MODEL` in `/root/autodl-tmp/.env` such as `gpt-5.2`.

```bash
cd /root/autodl-tmp/RL+LLM+EA_before_talk_best_res_task1_copy
python scripts/run_task1_ablations.py \
  --mol-lm GPT-4 \
  --ablations full no_rl no_dkb static_dkb \
  --seeds 1 2 3 \
  --max-oracle-calls 10000
```

## Output Layout

Result YAMLs:

```text
main/molleo_multi/ablation_results_task1/sum/<mol_lm>/<ablation>/seed<seed>/
```

Logs:

```text
logs/ablations_task1/
```

Each run writes files like:

```text
results_BioT5_['jnk3', 'qed']_['sa']1.yaml
```

## Summarize Results

After runs finish:

```bash
cd /root/autodl-tmp/RL+LLM+EA_before_talk_best_res_task1_copy
python scripts/summarize_task1_ablations.py \
  --aggregate sum \
  --mol-lm BioT5
```

Outputs:

```text
main/molleo_multi/ablation_results_task1/summary_sum_BioT5.csv
main/molleo_multi/ablation_results_task1/summary_sum_BioT5.md
```

The summary includes:

- `auc_top10`: the Task 1 top-10 AUC over total score
- `final_top10`: final top-10 mean total score
- `normalized_hv`: normalized hypervolume using default bounds `jnk3=0,1 qed=0,1 sa=1,10`

To use different HV bounds:

```bash
python scripts/summarize_task1_ablations.py \
  --aggregate sum \
  --mol-lm BioT5 \
  --bounds jnk3=0,1 qed=0,1 sa=1,10
```

## Direct Single Run

You can also run one ablation directly:

```bash
cd /root/autodl-tmp/RL+LLM+EA_before_talk_best_res_task1_copy
python run.py molleo_multi \
  --mol_lm BioT5 \
  --max_obj jnk3 qed \
  --min_obj sa \
  --seed 1 \
  --ablation no_dkb \
  --output_dir main/molleo_multi/ablation_results_task1/manual/no_dkb_seed1
```

For direct runs, prefer the batch runner unless you manually isolate
`MODEL_ADD_PATH_URL` and `MODEL_ADD_PREDICTOR_PATH_URL`, because incremental retraining writes
checkpoint files during optimization.

## Notes

- The full component ablations are wired for `molleo_multi`, the Sum aggregation version used in
  Table 1 of `main.tex`.
- `molleo_multi_pareto` can still be run with `--aggregate pareto`, but its current code does not
  train the DQN after storing replay transitions. Treat it as a Pareto aggregation comparison, not
  a fully matched RL ablation, unless you also synchronize the Pareto optimizer loop.
- The runner supports `--resume`, which skips a seed if the expected result YAML already exists.
