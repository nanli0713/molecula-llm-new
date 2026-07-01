from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class AblationConfig:
    name: str = "full"
    description: str = "Full MOL-E3: dynamic knowledge base, DQN strategy, history feedback, and incremental retraining."
    use_positive_memory: bool = True
    use_negative_memory: bool = True
    update_positive_memory: bool = True
    update_negative_memory: bool = True
    use_reward_memory: bool = True
    use_history_prompt: bool = True
    use_incremental_retrain: bool = True
    use_rl: bool = True
    strategy_mode: str = "dqn"
    fixed_action: int = 2
    dkb_high_k: int = 3
    dkb_low_k: int = 2

    @property
    def use_dkb(self) -> bool:
        return self.use_positive_memory or self.use_negative_memory


FULL = AblationConfig()

ABLATION_PRESETS: Dict[str, AblationConfig] = {
    "full": FULL,
    "no_rl": replace(
        FULL,
        name="no_rl",
        description="Remove DQN learning; always use the hybrid positive-negative editing action.",
        use_rl=False,
        strategy_mode="fixed",
        fixed_action=-1,
    ),
    "random_strategy": replace(
        FULL,
        name="random_strategy",
        description="Remove DQN learning; choose editing actions uniformly at random.",
        use_rl=False,
        strategy_mode="random",
    ),
    "no_dkb": replace(
        FULL,
        name="no_dkb",
        description="Remove dynamic knowledge base from prompts, RL state memory, reward fragment memory, and online updates.",
        use_positive_memory=False,
        use_negative_memory=False,
        update_positive_memory=False,
        update_negative_memory=False,
        use_reward_memory=False,
    ),
    "no_positive_memory": replace(
        FULL,
        name="no_positive_memory",
        description="Remove high-scoring fragment guidance while keeping negative memory.",
        use_positive_memory=False,
        update_positive_memory=False,
    ),
    "no_negative_memory": replace(
        FULL,
        name="no_negative_memory",
        description="Remove low-scoring fragment avoidance while keeping positive memory.",
        use_negative_memory=False,
        update_negative_memory=False,
    ),
    "static_dkb": replace(
        FULL,
        name="static_dkb",
        description="Use the initial knowledge base only; disable online DKB refresh and incremental cross-attention retraining.",
        update_positive_memory=False,
        update_negative_memory=False,
        use_incremental_retrain=False,
    ),
    "no_incremental_retrain": replace(
        FULL,
        name="no_incremental_retrain",
        description="Keep online molecule/fragment memory updates but disable LoRA incremental retraining of the cross-attention scorer.",
        use_incremental_retrain=False,
    ),
    "no_history_prompt": replace(
        FULL,
        name="no_history_prompt",
        description="Remove generated high/low molecule history from later LLM prompts.",
        use_history_prompt=False,
    ),
    "molleo_like": replace(
        FULL,
        name="molleo_like",
        description="Memoryless LLM editing baseline: no DKB, no DQN, no history prompt, no incremental retraining.",
        use_positive_memory=False,
        use_negative_memory=False,
        update_positive_memory=False,
        update_negative_memory=False,
        use_reward_memory=False,
        use_history_prompt=False,
        use_incremental_retrain=False,
        use_rl=False,
        strategy_mode="fixed",
        fixed_action=3,
    ),
}


def available_ablations() -> List[str]:
    return list(ABLATION_PRESETS.keys())


def describe_ablations(names: Iterable[str] | None = None) -> str:
    selected = list(names) if names is not None else available_ablations()
    lines = []
    for name in selected:
        cfg = ABLATION_PRESETS[name]
        lines.append(f"{name}: {cfg.description}")
    return "\n".join(lines)


def get_ablation_config(name: str | None) -> AblationConfig:
    key = (name or "full").strip()
    if key not in ABLATION_PRESETS:
        valid = ", ".join(available_ablations())
        raise ValueError(f"Unknown ablation '{key}'. Valid choices: {valid}")
    return ABLATION_PRESETS[key]
