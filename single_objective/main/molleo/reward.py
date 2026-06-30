from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterable, Any
import numpy as np
from utils import get_brics_fragments, count_frag_freq
from clustering import get_fp

class RunningStats:
    def __init__(self, momentum: float = 0.0):
        self.m = 0.0     # mean
        self.s = 0.0     # second moment accumulator
        self.n = 0
        self.momentum = momentum
        self.var = 1.0   # for EMA mode
        self.initialized = False

    def update(self, x: float):
        if self.momentum <= 0.0:
            self.n += 1
            if self.n == 1:
                self.m = x
                self.s = 0.0
            else:
                delta = x - self.m
                self.m += delta / self.n
                self.s += delta * (x - self.m)
        else:
            # EMA
            alpha = self.momentum
            if not self.initialized:
                self.m = x
                self.var = 1.0
                self.initialized = True
            else:
                prev_m = self.m
                self.m = (1 - alpha) * self.m + alpha * x
                self.var = (1 - alpha) * (self.var + alpha * (x - prev_m) ** 2)
        return self

    def mean(self) -> float:
        return self.m

    def std(self) -> float:
        if self.momentum <= 0.0:
            if self.n < 2:
                return 1.0
            return np.sqrt(max(self.s / (self.n - 1), 1e-12))
        else:
            return float(np.sqrt(max(self.var, 1e-12)))


class TermNormalizer:
    def __init__(self, momentum: float = 0.0, eps: float = 1e-8, clip: Optional[float] = 5.0):
        self.stats: Dict[str, RunningStats] = {}
        self.momentum = momentum
        self.eps = eps
        self.clip = clip

    def normalize(self, key: str, x: float, update: bool = True) -> float:
        if key not in self.stats:
            self.stats[key] = RunningStats(momentum=self.momentum)
        if update:
            self.stats[key].update(x)
        mu = self.stats[key].mean()
        sigma = self.stats[key].std()
        z = (x - mu) / (sigma + self.eps)
        if self.clip is not None:
            z = float(np.clip(z, -self.clip, self.clip))
        return z

@dataclass
class RewardConfig:
    target_main: float = 0.9
    target_tox: float = 0.0
    target_cluster_sim: float = 0.6 
    novelty_band: Tuple[float, float] = (0.3, 0.85) 

    cutoff_new_cluster: float = 0.825
    duplicate_weight: float = -1.0 

    w_main: float = 1.0
    w_tox: float = 1.0
    w_struct_pos: float = 0.7
    w_struct_neg: float = 1.0 
    w_novelty: float = 0.6
    w_new_cluster: float = 0.3
    w_relative: float = 0.8
    w_duplicate: float = 0.3

    penalty_bad_frag: float = 2.0
    penalty_novelty_hard: float = 3.0

    normalizer_momentum: float = 0.05 
    z_clip: float = 5.0

    final_tanh_scale: float = 3.0
    final_scale: float = 5.0 


def calc_frag_weights(high_frag_freq: Dict[str, int],
                      low_frag_freq: Dict[str, int],
                      epsilon: float = 1e-3) -> Dict[str, float]:
    all_frags = set(list(high_frag_freq.keys()) + list(low_frag_freq.keys()))
    weights = {}
    for frag in all_frags:
        fh = high_frag_freq.get(frag, 0)
        fl = low_frag_freq.get(frag, 0)
        weights[frag] = float(np.log((fh + epsilon) / (fl + epsilon)))  # >0:正片段；<0:负片段
    return weights

def structure_pos_neg_scores(smiles: Any, frag_weights: Dict[str, float],
                             neg_hard_threshold: float = -1.5) -> Tuple[float, float, bool]:

    frags = get_brics_fragments(smiles) or []
    if len(frags) == 0:
        return 0.0, 0.0, False

    vals = [frag_weights.get(f, 0.0) for f in frags]
    pos_vals = [max(v, 0.0) for v in vals]
    neg_vals = [-min(v, 0.0) for v in vals] 

    pos_raw = float(np.mean(pos_vals)) if pos_vals else 0.0
    neg_raw = float(np.mean(neg_vals)) if neg_vals else 0.0
    has_hard_bad = any(v <= neg_hard_threshold for v in vals)
    return pos_raw, neg_raw, has_hard_bad


def assign_new_mol(new_mol, cluster_centers_fps, fp_type="morgan", cutoff=0.825):
    if type(new_mol).__name__ != "Mol":
        print("clustering assign_new_mol mols:", new_mol)
    fp_new = get_fp(new_mol, fp_type)
    from rdkit import DataStructs
    sims = [DataStructs.TanimotoSimilarity(fp_new, fp_cent) for fp_cent in cluster_centers_fps]
    best_idx = int(np.argmax(sims)) if len(sims) > 0 else -1
    best_sim = float(sims[best_idx]) if best_idx >= 0 else 0.0
    is_new = best_sim < cutoff * 0.2
    return {
        "is_new_cluster": bool(is_new),
        "best_sim": best_sim,
        "to_cluster": None if is_new else best_idx
    }


@dataclass
class RewardEngine:
    cluster_centers_fps: List[Any]
    all_smiles: set
    high_mols: Optional[List[Any]] = None
    low_mols: Optional[List[Any]] = None
    config: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self):
        self.norm = TermNormalizer(
            momentum=self.config.normalizer_momentum,
            clip=self.config.z_clip
        )
        if self.high_mols is not None and self.low_mols is not None:
            high_ff = count_frag_freq(self.high_mols)
            low_ff = count_frag_freq(self.low_mols)
            self.frag_weights = calc_frag_weights(high_ff, low_ff, epsilon=1e-3)
        else:
            self.frag_weights = {}

    def _novelty_terms(self, best_sim: float, is_new: bool) -> Tuple[float, float, bool]:
        """
        novelty_raw = -|best_sim - target_sim|  
        new_cluster_raw = 1.0 if is_new else 0.0
        hard_penalty if best_sim outside band
        """
        t = self.config.target_cluster_sim
        novelty_raw = -(abs(best_sim - t))
        lo, hi = self.config.novelty_band
        hard = (best_sim < lo) or (best_sim > hi)
        new_cluster_raw = 1.0 if is_new else 0.0
        return novelty_raw, new_cluster_raw, hard

    def _relative_improvement(self,
                              curr: float,
                              baseline: Optional[float]) -> float:
        if baseline is None:
            return 0.0
        denom = max(abs(baseline), 1e-6)
        return float((curr - baseline) / denom)

    def get_reward(self,
                   mol_list: List[Any],
                   main_scores: List[float],
                   toxicity_scores: Optional[List[float]] = None,
                   parent_map: Optional[Dict[Any, Any]] = None,
                   parent_main_scores: Optional[Dict[Any, float]] = None,
                   topk_avg_main: Optional[float] = None,
                   fp_type: str = "morgan") -> List[float]:
        rewards = []
        for mol, main in zip(mol_list, main_scores):
            if mol is None:
                rewards.append(0.0)
                continue

            try:
                from rdkit import Chem
                smi = Chem.MolToSmiles(mol)
            except Exception:
                smi = str(mol)

            is_duplicate = smi in self.all_smiles

            assign_info = assign_new_mol(mol, self.cluster_centers_fps, fp_type=fp_type,
                                         cutoff=self.config.cutoff_new_cluster)
            best_sim = assign_info["best_sim"]
            is_new_cluster = assign_info["is_new_cluster"]

            pos_raw, neg_raw, has_bad_frag = structure_pos_neg_scores(
                smi, self.frag_weights, neg_hard_threshold=-1.5
            )

            main_raw = -(abs(main - self.config.target_main))
            tox_raw = 0.0
            if toxicity_scores is not None:
                tox_val = toxicity_scores[len(rewards)]
                tox_raw = -(abs(tox_val - self.config.target_tox))

            novelty_raw, new_cluster_raw, novelty_hard = self._novelty_terms(best_sim, is_new_cluster)

            rel_improve_parent = 0.0
            if parent_map is not None and parent_main_scores is not None:
                p1 = parent_map.get(smi, None)

                if p1 is None and mol in parent_map:
                    p1 = parent_map[mol]
                base = None
                if p1 is not None:
                    base = parent_main_scores.get(p1[0], None)
                rel_improve_parent = self._relative_improvement(main, base)

            rel_improve_topk = self._relative_improvement(main, topk_avg_main)

            relative_raw = max(rel_improve_parent, rel_improve_topk)

            duplicate_raw = 1.0 if is_duplicate else 0.0

            z = {}
            z["main"] = self.norm.normalize("main", main_raw, update=True)
            z["tox"] = self.norm.normalize("tox", tox_raw, update=True)
            z["struct_pos"] = self.norm.normalize("struct_pos", pos_raw, update=True)
            z["struct_neg"] = self.norm.normalize("struct_neg", neg_raw, update=True)
            z["novelty"] = self.norm.normalize("novelty", novelty_raw, update=True)
            z["new_cluster"] = self.norm.normalize("new_cluster", new_cluster_raw, update=True)
            z["relative"] = self.norm.normalize("relative", relative_raw, update=True)
            z["duplicate"] = self.norm.normalize("duplicate", duplicate_raw, update=True)

            cfg = self.config
            combined = (
                cfg.w_main * z["main"]
                + cfg.w_tox * z["tox"]
                + cfg.w_struct_pos * z["struct_pos"]
                - cfg.w_struct_neg * z["struct_neg"]
                + cfg.w_novelty * z["novelty"]
                + cfg.w_new_cluster * z["new_cluster"]
                + cfg.w_relative * z["relative"]
                - cfg.w_duplicate * z["duplicate"]
            )

            penalty = 0.0
            if has_bad_frag:
                penalty -= cfg.penalty_bad_frag
            if novelty_hard:
                penalty -= cfg.penalty_novelty_hard

            final = cfg.final_scale * np.tanh((combined + penalty) / cfg.final_tanh_scale)

            rewards.append(float(final))

        return rewards
    
from typing import List, Dict, Any, Tuple
import math

def _build_rdkit_catalogs():

    from rdkit.Chem import FilterCatalog as FC
    catalogs = {}
    for name, cat in [
        ("PAINS", FC.FilterCatalogParams.FilterCatalogs.PAINS),
        ("BRENK", FC.FilterCatalogParams.FilterCatalogs.BRENK),
        ("NIH",   FC.FilterCatalogParams.FilterCatalogs.NIH),
        ("ZINC",  FC.FilterCatalogParams.FilterCatalogs.ZINC),
    ]:
        p = FC.FilterCatalogParams()
        p.AddCatalog(cat)
        catalogs[name] = FC.FilterCatalog(p)
    return catalogs

class RDKitToxicityScorer:
    def __init__(self,
                 weights: Dict[str, float] = None,
                 lambda_alpha: float = math.log(2),
                 tox_threshold: float = 0.5):

        self.weights = weights or {
            "PAINS": 1.0,
            "BRENK": 0.8,
            "NIH":   0.6,
            "ZINC":  0.5,
        }
        self.lambda_alpha = float(lambda_alpha)
        self.tox_threshold = float(tox_threshold)
        self.catalogs = _build_rdkit_catalogs()

    def score_mol(self, mol) -> Dict[str, Any]:
        import math
        from rdkit import Chem

        if mol is None:
            return {"prob": 0.0, "is_toxic": False, "raw": 0.0, "alerts": {}}

        alerts_detail: Dict[str, List[Tuple[str, str]]] = {}
        raw = 0.0

        for name, catalog in self.catalogs.items():
            matches = catalog.GetMatches(mol)  # list of FilterCatalogEntry (new) or FilterMatch (old)
            if not matches:
                continue

            entries = []
            for m in matches:
                entry = m.GetFilter() if hasattr(m, "GetFilter") else m

                ename = ""
                edesc = ""
                try:
                    ename = entry.GetName()
                except Exception:
                    pass
                try:
                    edesc = entry.GetDescription()
                except Exception:
                    pass

                entries.append((ename, edesc))

            if entries:
                alerts_detail[name] = entries
                raw += self.weights.get(name, 0.0) * len(entries)  # 用 entries 长度更稳妥

        prob = 1.0 - math.exp(-self.lambda_alpha * raw)
        is_toxic = prob >= self.tox_threshold

        return {
            "prob": float(prob),
            "is_toxic": bool(is_toxic),
            "raw": float(raw),
            "alerts": alerts_detail
        }

    def score_mol_list(self, mol_list: List[Any]) -> Dict[str, Any]:
        probs, labels, details = [], [], []
        for mol in mol_list:
            r = self.score_mol(mol)
            probs.append(r["prob"])
            labels.append(r["is_toxic"])
            details.append({"raw": r["raw"], "alerts": r["alerts"]})
        return {"probs": probs, "labels": labels, "details": details}