from __future__ import print_function
from collections import defaultdict
import math
import random
from typing import List
from tqdm import tqdm
from crossAtt_implements import Graph_encoder
import pandas as pd
import joblib
import numpy as np
from rdkit import  rdBase
from rdkit.Chem.rdchem import Mol
rdBase.DisableLog('rdApp.error')
import torch
from rdkit import Chem, DataStructs
from settings import settings
import rdkit
from reward import RDKitToxicityScorer, RewardConfig, RewardEngine
import crossover as co, mutate as mu
from main.ablation import get_ablation_config
from main.optimizer import BaseOptimizer
from torch.utils.data import DataLoader
from crossAtt import MolMultiModalDataset
from biot5 import BioT5
from GPT4 import GPT4
from clustering import restore_clusters_from_df, get_fp, compute_center_mol
from utils import  get_brics_fragments, get_fp_scores
from trainCrossAtt import collate_fn
import torch.nn as nn
import torch.optim as optim
from rl import ActionSelector, DQNTrainer, EpisodeManager, ReplayBuffer, _replace_linear_with_lora, collect_lora_trainable_params, merge_lora_and_unwrap, select_fragment
from fragments_utils import build_brics_fragment_contexts
MINIMUM = 1e-10
def format_detail(detail):
    if detail is None:
        return ""
    if isinstance(detail, str):
        return detail
    if isinstance(detail, (list, tuple)):
        return " ".join(map(str, detail))
    return str(detail)

def nearest_cluster_id(current_mol_a, current_mol_b, clustering_data, fp_type="morgan", weights=(0.5, 0.5)):
    centers_fps = []
    cluster_ids = []
    for cid, cluster in enumerate(clustering_data):
        if cluster:
            _, center_fp = compute_center_mol(cluster, fp_type=fp_type)
            centers_fps.append(center_fp)
            cluster_ids.append(cid)
    if not centers_fps:
        return 0

    fp_a = get_fp(current_mol_a, fp_type)
    fp_b = get_fp(current_mol_b, fp_type)

    sims_a = np.array(DataStructs.BulkTanimotoSimilarity(fp_a, centers_fps))
    sims_b = np.array(DataStructs.BulkTanimotoSimilarity(fp_b, centers_fps))

    w0, w1 = weights
    s = w0 + w1
    if s <= 0:
        w0 = w1 = 0.5
    else:
        w0, w1 = w0 / s, w1 / s

    sims = w0 * sims_a + w1 * sims_b
    return int(cluster_ids[int(np.argmax(sims))])

def make_mating_pool(population_mol: List[Mol], population_scores, population_scores_detail, offspring_size: int):
    """
    Given a population of RDKit Mol and their scores, sample a list of the same size
    with replacement using the population_scores as weights
    Args:
        population_mol: list of RDKit Mol
        population_scores: list of un-normalised scores given by ScoringFunction
        offspring_size: number of molecules to return
    Returns: a list of RDKit Mol (probably not unique)
    """
    # scores -> probs
    all_tuples = list(zip(population_scores, population_mol, population_scores_detail))
    population_scores = [s + MINIMUM for s in population_scores]
    sum_scores = sum(population_scores)
    population_probs = [p / sum_scores for p in population_scores]
    mating_indices = np.random.choice(len(all_tuples), p=population_probs, size=offspring_size, replace=True)
    
    mating_tuples = [all_tuples[indice] for indice in mating_indices]
    
    return mating_tuples

def reproduce(mating_tuples, mutation_rate, mol_lm=None, net=None):
    """
    Args:
        mating_pool: list of RDKit Mol
        mutation_rate: rate of mutation
    Returns:
    """
    parent = []
    parent.append(random.choice(mating_tuples))
    parent.append(random.choice(mating_tuples))

    parent_mol = [t[1] for t in parent]
    new_child = co.crossover(parent_mol[0], parent_mol[1])
    new_child_mutation = None
    if new_child is not None:
        new_child_mutation = mu.mutate(new_child, mutation_rate, mol_lm)
    return new_child, new_child_mutation

def get_best_mol(population_scores, population_mol):
    top_mol = population_mol[np.argmax(population_scores)]
    top_smi = Chem.MolToSmiles(top_mol)
    return top_smi

def get_topn_fragment_bonus(high_mols, high_scores, topn=3, decay_weights=None):
    """
    返回每个片段来自topN分子的额外bonus
    """
    high_mols = _ensure_mol_list(high_mols)
    high_scores = _sanitize_scores(high_scores, len(high_mols))

    valid = []
    for m, s in zip(high_mols, high_scores):
        if s is not None:
            valid.append((m, s))

    if not valid:
        return {}

    valid = sorted(valid, key=lambda x: x[1], reverse=True)[:topn]

    if decay_weights is None:
        # 默认 top1 > top2 > top3
        decay_weights = [2.0, 1.0, 0.5]

    frag_bonus = defaultdict(float)

    for rank, (mol, score) in enumerate(valid):
        w = decay_weights[rank] if rank < len(decay_weights) else decay_weights[-1]
        frags = set(get_brics_fragments(mol))
        for f in frags:
            frag_bonus[f] += w

    return dict(frag_bonus)

def get_cluster_center_feature(current_mol, clustering_data, fp_type="morgan"):

    cluster_centers_fps = []
    for cluster in clustering_data:
        if cluster: 
            center_mol, center_fp = compute_center_mol(cluster, fp_type=fp_type)
            center_fp_np = np.array(center_fp)
            cluster_centers_fps.append(center_fp)

    if not cluster_centers_fps:
        return torch.zeros(4096, dtype=torch.float32)

    if type(current_mol) is not rdkit.Chem.rdchem.Mol:
        print("run get_cluster_center_feature mols:", current_mol)

    fp_current = get_fp(current_mol, fp_type) 
    sims = [DataStructs.TanimotoSimilarity(fp_current, fp_cent) for fp_cent in cluster_centers_fps]
    best_idx = int(np.argmax(sims)) 
    center_fp = cluster_centers_fps[best_idx] 
    center_fp_np = np.array(center_fp) 
    return torch.tensor(center_fp_np, dtype=torch.float32) 


def sanitize(mol_list):
    new_mol_list = [] 
    smiles_set = set() 
    for mol in mol_list:
        if mol is not None:
            try:
                smiles = Chem.MolToSmiles(mol) 
                if smiles is not None and smiles not in smiles_set: 
                    smiles_set.add(smiles)  
                    new_mol_list.append(mol)
            except ValueError:
                print('bad smiles') 
    return new_mol_list


def get_mol_pool_feature(mol_list, fp_type="morgan"):
    if not mol_list:
        return torch.zeros(4096, dtype=torch.float32)

    fps = []
    temp = [] 

    for mol in mol_list:
        if type(mol) is str:
            temp.append(Chem.MolFromSmiles(mol)) 
        else:
            temp.append(mol) 

    temp = sanitize(temp) 
    mol_list = temp

    for m in mol_list:
        fp = get_fp(m, fp_type) 
        fps.append(np.array(fp))

    avg_fp = np.mean(fps, axis=0) 
    return torch.tensor(avg_fp, dtype=torch.float32) 


def build_state_dict(parents, clustering_data, high_score_mols, low_score_mols, fp_type="morgan"):
    if type(parents[0]) is not rdkit.Chem.rdchem.Mol: 
        print("run build_state_dict mols:", parents[0])

    state = {
        "parent_fp": torch.cat([ 
            torch.tensor(np.array(get_fp(m, fp_type)), dtype=torch.float32)
            for m in parents
        ], dim=0), 

        "cluster_center": get_cluster_center_feature(parents[0], clustering_data, fp_type) + get_cluster_center_feature(parents[1], clustering_data, fp_type),
        "high_pool": get_mol_pool_feature(high_score_mols, fp_type),
        "low_pool": get_mol_pool_feature(low_score_mols, fp_type),
    }
    return state

def format_prompt_lines(ctx_list, desc='High score substractions: '):
    lines = []
    for c in ctx_list:
        lines.append(f"{desc}: [Mol FRAG] {c['fragment']} | This frag has {c['n_cuts']} attachment points = {c['labels']} | cut times={c['n_cuts']}")
        for idx, cut in enumerate(c["cuts"]):
            lines.append(f"  {idx} - {cut['text']}")
    return lines

def ensure_mol(x):
    if isinstance(x, Chem.Mol):
        return x
    if isinstance(x, str):
        m = Chem.MolFromSmiles(x)
        if m is None:
            raise ValueError(f"Invalid SMILES: {x}")
        return m
    raise TypeError(f"Expected RDKit Mol or SMILES str, got {type(x)}")

def _ensure_mol_list(mols):
    return [ensure_mol(m) for m in mols]

def _collect_brics_presence_and_index(mol_list):
    frag2count = defaultdict(int)
    frag2idxs = defaultdict(list)
    mol_sets = []
    for idx, m in enumerate(mol_list):
        frags = set(get_brics_fragments(m))
        mol_sets.append(frags)
        for f in frags:
            frag2count[f] += 1
            frag2idxs[f].append(idx)
    return frag2count, frag2idxs, mol_sets

def _log_risk_ratio(n_pos, N_pos, n_neg, N_neg, alpha=1.0):
    p_pos = (n_pos + alpha) / (N_pos + 2.0 * alpha)
    p_neg = (n_neg + alpha) / (N_neg + 2.0 * alpha)
    return float(np.log(p_pos) - np.log(p_neg))

def _as_float_or_none(x):
    try:
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None

def _sanitize_scores(scores, N_expected):
    if scores is None:
        return None
    try:
        L = len(scores)
    except TypeError:
        return None

    s = list(scores)
    if L < N_expected:
        s = s + [None] * (N_expected - L)
    elif L > N_expected:
        s = s[:N_expected]

    s = [_as_float_or_none(v) for v in s]

    if not any(v is not None for v in s):
        return None
    return s

def _pick_exemplar_index(frag, frag2idxs, scores=None):
    idxs = frag2idxs.get(frag, [])
    if not idxs:
        return None
    if scores is None:
        return idxs[0]

    valid_pairs = []
    n_scores = len(scores)
    for i in idxs:
        if 0 <= i < n_scores:
            v = scores[i]
            if v is not None:
                valid_pairs.append((i, v))
    if valid_pairs:
        return max(valid_pairs, key=lambda p: p[1])[0]
    return idxs[0]

def _compute_fragment_group_weight(frag, frag2idxs, scores, top_weight=0.5):
    idxs = frag2idxs.get(frag, [])
    if not idxs:
        return 0.0

    if scores is None:
        return float(len(idxs))

    valid_pairs = []
    for i in idxs:
        if 0 <= i < len(scores):
            s = scores[i]
            if s is not None:
                valid_pairs.append((i, s))

    if not valid_pairs:
        return 0.0

    valid_pairs = sorted(valid_pairs, key=lambda x: x[1], reverse=True)

    if len(valid_pairs) == 1:
        return float(valid_pairs[0][1])

    top_score = valid_pairs[0][1]
    rest_scores = [x[1] for x in valid_pairs[1:]]

    rest_part = float(np.mean(rest_scores)) if rest_scores else 0.0
    return float(top_weight * top_score + (1.0 - top_weight) * rest_part)

def format_generation_record(smiles, score, detail):
    return (
        "Smiles Mol: " + str(smiles) +
        ",Total score: " + str(score) +
        ", Score details by using Evaluation rules: " + format_detail(detail)
    )
    # return (
    #     "Smiles Mol: " + str(smiles)
    #     # ",Total score: " + str(score) +
    #     # ", Score details by using Evaluation rules: " + format_detail(detail)
    # )

def update_generation_history(history, record, topk=3, reverse=True):
    history.append(record)
    history.sort(key=extract_score, reverse=reverse)
    return history[:topk]

def _pick_exemplar_index_weighted(frag, frag2idxs, scores=None, top_weight=0.5):
    idxs = frag2idxs.get(frag, [])
    if not idxs:
        return None
    if scores is None:
        return idxs[0]

    valid_pairs = []
    n_scores = len(scores)
    for i in idxs:
        if 0 <= i < n_scores:
            v = scores[i]
            if v is not None:
                valid_pairs.append((i, v))

    if valid_pairs:
        valid_pairs = sorted(valid_pairs, key=lambda x: x[1], reverse=True)
        return valid_pairs[0][0]

    return idxs[0]

def _compute_fragment_group_weight(frag, frag2idxs, scores, top_weight=0.5):
    """
    计算某个片段在一个组内的加权分数。
    思路：
    - 如果没有 scores，就退化为出现次数
    - 如果有 scores，则强调该片段对应的最高分样本，同时兼顾其余样本均值
    """
    idxs = frag2idxs.get(frag, [])
    if not idxs:
        return 0.0

    if scores is None:
        return float(len(idxs))

    valid_pairs = []
    for i in idxs:
        if 0 <= i < len(scores):
            s = scores[i]
            if s is not None:
                valid_pairs.append((i, s))

    if not valid_pairs:
        return 0.0

    valid_pairs = sorted(valid_pairs, key=lambda x: x[1], reverse=True)

    if len(valid_pairs) == 1:
        return float(valid_pairs[0][1])

    top_score = valid_pairs[0][1]
    rest_scores = [x[1] for x in valid_pairs[1:]]
    rest_part = float(np.mean(rest_scores)) if rest_scores else 0.0

    return float(top_weight * top_score + (1.0 - top_weight) * rest_part)


def _pick_exemplar_index_weighted(frag, frag2idxs, scores=None, top_weight=0.5):
    """
    为某个片段选择代表分子索引。
    当前策略：优先选择该片段对应的最高分分子。
    """
    idxs = frag2idxs.get(frag, [])
    if not idxs:
        return None

    if scores is None:
        return idxs[0]

    valid_pairs = []
    n_scores = len(scores)
    for i in idxs:
        if 0 <= i < n_scores:
            v = scores[i]
            if v is not None:
                valid_pairs.append((i, v))

    if valid_pairs:
        valid_pairs = sorted(valid_pairs, key=lambda x: x[1], reverse=True)
        return valid_pairs[0][0]

    return idxs[0]


def get_topn_fragment_bonus(high_mols, high_scores, topn=3, decay_weights=None):
    """
    对高分组中 topN 分子的片段给予额外 bonus。
    默认：
        top1: 2.5
        top2: 1.2
        top3: 0.6

    返回:
        {frag: bonus_score}
    """
    high_mols = _ensure_mol_list(high_mols)
    high_scores = _sanitize_scores(high_scores, len(high_mols))

    valid = []
    for m, s in zip(high_mols, high_scores):
        if s is not None:
            valid.append((m, s))

    if not valid:
        return {}

    valid = sorted(valid, key=lambda x: x[1], reverse=True)[:topn]

    if decay_weights is None:
        decay_weights = [2.5, 1.2, 0.6]

    frag_bonus = defaultdict(float)

    for rank, (mol, _) in enumerate(valid):
        w = decay_weights[rank] if rank < len(decay_weights) else decay_weights[-1]
        frags = set(get_brics_fragments(mol))
        for f in frags:
            frag_bonus[f] += float(w)

    return dict(frag_bonus)


def get_top1_fragments(high_mols, high_scores):
    """
    取最高分分子(top1)的全部 BRICS 片段。
    """
    high_mols = _ensure_mol_list(high_mols)
    high_scores = _sanitize_scores(high_scores, len(high_mols))

    valid = []
    for m, s in zip(high_mols, high_scores):
        if s is not None:
            valid.append((m, s))

    if not valid:
        return []

    top1_mol = sorted(valid, key=lambda x: x[1], reverse=True)[0][0]
    return list(set(get_brics_fragments(top1_mol)))

def _frag_num_heavy_atoms(frag_smi):
    mol = Chem.MolFromSmiles(frag_smi)
    if mol is None:
        return 0
    return mol.GetNumHeavyAtoms()

def summarize_brics_by_groups(
    high_mols,
    low_mols,
    high_scores=None,
    low_scores=None,
    k_high=20,
    k_low=20,
    exploration_temp=None,
    alpha=1.0,
    replacement_threshold=5,
    min_count=1,
    min_high_count=2,
    top_weight=0.5,
    frag_history=None,
    confidence_c=5.0,
    temporal_weight=1.0,
    topn_bonus=3,
    topn_decay_weights=None,
    top1_force=False,
    top1_force_n=2,
    min_frag_heavy_atoms=4,
):
    """
    高低分分子 BRICS 片段汇总与打分。

    新增能力：
    1. topN 高分分子片段 bonus
    2. top1 分子片段强制进入 high 结果
    3. 保留原有统计项：区分性、频率、组内得分、历史稳定性、置信度
    """
    high_mols = _ensure_mol_list(high_mols)
    low_mols = _ensure_mol_list(low_mols)

    N_h = len(high_mols)
    N_l = len(low_mols)

    if frag_history is None:
        frag_history = []
    history_len = len(frag_history)

    high_scores = _sanitize_scores(high_scores, N_h)
    low_scores  = _sanitize_scores(low_scores, N_l)

    h_count, h_f2idxs, h_sets = _collect_brics_presence_and_index(high_mols)
    l_count, l_f2idxs, l_sets = _collect_brics_presence_and_index(low_mols)

    all_frags = set(h_count) | set(l_count)

    # 新增：topN 分子片段 bonus
    top_frag_bonus = get_topn_fragment_bonus(
        high_mols,
        high_scores,
        topn=topn_bonus,
        decay_weights=topn_decay_weights
    )

    frag_importance = {}
    frag_debug = {}

    for f in all_frags:
        nh = h_count.get(f, 0)
        nl = l_count.get(f, 0)

        if nh + nl < min_count:
            continue
        if nh < min_high_count:
            continue

        if _frag_num_heavy_atoms(f) < min_frag_heavy_atoms:
            continue

        # 1) 高低分组区分性
        base_score = _log_risk_ratio(nh, N_h, nl, N_l, alpha=alpha)

        # 2) 组内分数权重
        high_group_weight = _compute_fragment_group_weight(
            f, h_f2idxs, high_scores, top_weight=top_weight
        )
        low_group_weight = _compute_fragment_group_weight(
            f, l_f2idxs, low_scores, top_weight=top_weight
        )

        # 3) 相对频率
        high_freq = nh / max(N_h, 1)
        low_freq = nl / max(N_l, 1)

        # 4) 时间稳定性
        if history_len > 0:
            temporal_stability = sum(f in hist for hist in frag_history) / history_len
        else:
            temporal_stability = 0.0

        # 5) topN bonus
        top_bonus = top_frag_bonus.get(f, 0.0)

        # 综合原始得分
        raw_score = (
            1.0 * base_score
            + 1.5 * high_freq
            - 0.5 * low_freq
            + 0.2 * (high_group_weight - low_group_weight)
            + temporal_weight * temporal_stability
            + top_bonus
        )

        # 6) 支持度置信度
        support = nh + nl
        confidence = support / (support + confidence_c)

        score = confidence * raw_score
        frag_importance[f] = score

        frag_debug[f] = {
            "nh": nh,
            "nl": nl,
            "base_score": base_score,
            "high_freq": high_freq,
            "low_freq": low_freq,
            "high_group_weight": high_group_weight,
            "low_group_weight": low_group_weight,
            "temporal_stability": temporal_stability,
            "top_bonus": top_bonus,
            "support": support,
            "confidence": confidence,
            "final_score": score,
        }

    high_candidates = list(h_count.keys())
    low_candidates = list(l_count.keys())

    h_frag_top = select_fragment(
        high_candidates,
        frag_importance,
        k=k_high,
        exploration_temp=exploration_temp
    )

    # 新增：强制注入 top1 分子片段
    if top1_force:
        top1_frags = get_top1_fragments(high_mols, high_scores)

        # 根据 frag_importance 排序 top1 片段
        top1_frags = sorted(
            top1_frags,
            key=lambda f: frag_importance.get(f, float("-inf")),
            reverse=True
        )

        forced = []
        for f in top1_frags:
            if f in high_candidates and f not in forced:
                forced.append(f)
            if len(forced) >= top1_force_n:
                break

        h_frag_top = (forced + [f for f in h_frag_top if f not in forced])[:k_high]

    frag_importance_low = {f: -frag_importance.get(f, 0.0) for f in all_frags}
    l_frag_top = select_fragment(
        low_candidates,
        frag_importance_low,
        k=k_low,
        exploration_temp=exploration_temp
    )

    h_ctx_all, l_ctx_all = [], []

    for f in h_frag_top:
        ex_idx = _pick_exemplar_index_weighted(
            f, h_f2idxs, high_scores, top_weight=top_weight
        )
        if ex_idx is None:
            continue

        ctx_map = build_brics_fragment_contexts(
            high_mols[ex_idx],
            replacement_threshold=replacement_threshold
        )
        if f in ctx_map:
            h_ctx_all.append(ctx_map[f])

    for f in l_frag_top:
        ex_idx = _pick_exemplar_index_weighted(
            f, l_f2idxs, low_scores, top_weight=top_weight
        )
        if ex_idx is None:
            continue

        ctx_map = build_brics_fragment_contexts(
            low_mols[ex_idx],
            replacement_threshold=replacement_threshold
        )
        if f in ctx_map:
            l_ctx_all.append(ctx_map[f])

    h_prompt_lines = format_prompt_lines(h_ctx_all, desc='High score substractions ')
    l_prompt_lines = format_prompt_lines(l_ctx_all, desc='Low score substractions ')

    return {
        "high": {
            "ctx": h_ctx_all,
            "prompt_lines": h_prompt_lines,
            "chosen_frags": h_frag_top,
        },
        "low": {
            "ctx": l_ctx_all,
            "prompt_lines": l_prompt_lines,
            "chosen_frags": l_frag_top,
        },
        "frag_importance": frag_importance,
        "frag_debug": frag_debug,
        "top_frag_bonus": top_frag_bonus,
        "stats": {
            "N_high": N_h,
            "N_low": N_l,
            "n_frag_high": len(h_count),
            "n_frag_low": len(l_count),
            "k_high": len(h_ctx_all),
            "k_low": len(l_ctx_all),
            "alpha": alpha,
            "min_count": min_count,
            "min_high_count": min_high_count,
            "history_len": history_len,
            "confidence_c": confidence_c,
            "temporal_weight": temporal_weight,
            "topn_bonus": topn_bonus,
            "topn_decay_weights": topn_decay_weights if topn_decay_weights is not None else [2.5, 1.2, 0.6],
            "top1_force": top1_force,
            "top1_force_n": top1_force_n,
        }
    }


from typing import List, Iterable, Any, Callable, Optional

def dedup_mol_score_pairs(mols, scores):
    seen = set()
    out_mols, out_scores = [], []
    for mol, score in zip(mols, scores):
        if mol is None:
            continue
        try:
            smi = Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            continue
        if smi in seen:
            continue
        seen.add(smi)
        out_mols.append(mol)
        out_scores.append(score)
    return out_mols, out_scores

def update_high_pool(high_mols, high_scores, new_mols, new_scores, max_size):
    all_mols = list(high_mols) + list(new_mols)
    all_scores = list(high_scores) + list(new_scores)

    all_mols, all_scores = dedup_mol_score_pairs(all_mols, all_scores)

    pairs = sorted(zip(all_scores, all_mols), key=lambda x: x[0], reverse=True)[:max_size]
    new_high_scores = [p[0] for p in pairs]
    new_high_mols = [p[1] for p in pairs]
    return new_high_mols, new_high_scores

def update_low_pool(low_mols, low_scores, new_mols, new_scores, max_size):
    all_mols = list(low_mols) + list(new_mols)
    all_scores = list(low_scores) + list(new_scores)

    all_mols, all_scores = dedup_mol_score_pairs(all_mols, all_scores)

    pairs = sorted(zip(all_scores, all_mols), key=lambda x: x[0], reverse=False)[:max_size]
    new_low_scores = [p[0] for p in pairs]
    new_low_mols = [p[1] for p in pairs]
    return new_low_mols, new_low_scores

def dedup_nested_list(nested_list: List[List[Any]], key: Optional[Callable[[Iterable[Any]], Any]] = None) -> List[List[Any]]:

    if not nested_list:
        return []

    seen = set()
    result = []

    for sublist in nested_list:
        if key is None:
            k = tuple(sublist)  
        else:
            k = key(sublist)   
        
        if k not in seen:
            seen.add(k)
            result.append(sublist) 

    return result
import re
score_pattern = re.compile(r"Total score:\s*([+-]?\d+(?:\.\d+)?)")

def extract_score(s):
    m = score_pattern.search(s)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return float('-inf') 
    return float('-inf')
from openai import OpenAI

client = OpenAI(
    api_key=settings.api_key,
    base_url=settings.base_url
)
client2 = OpenAI(
    api_key=settings.api_key2,
    base_url=settings.base_url
)

def query_LLM(question, model=settings.model, temperature=0.5):
    message = [{"role": "system", "content": "You are a helpful agent who can extract data for specific format."}]
    prompt1 = question
    message.append({"role": "user", "content": prompt1})
    flag = 0
    for retry in range(3):
        try:
            response = client.chat.completions.create(
                model=model, messages=message, temperature=temperature, stream=False,
            ).choices[0].message.content
            message.append({"role": "assistant", "content": response})
            flag = 1
            break
        except Exception as e:
            print(f"{type(e).__name__} {e}")
    if not flag:
        for retry in range(3): 
            try:      
                response = client2.chat.completions.create(
                        model=model, messages=message, temperature=temperature, stream=False,
                    ).choices[0].message.content
                message.append({"role": "assistant", "content": response})
            except Exception as e:
                print(f"sencond times:{type(e).__name__} {e}")
    print("=>")
    return message, response

def canonicalize_smiles_list(mols):
    smis = []
    for mol in mols:
        if mol is None:
            continue
        try:
            smis.append(Chem.MolToSmiles(mol, canonical=True))
        except Exception:
            continue
    return smis

def build_score_map(smiles_list, score_list):
    return dict(zip(smiles_list, score_list))

def query_oracle_with_cache(oracle, smiles_list, score_cache, detail_cache=None):
    if detail_cache is None:
        detail_cache = {}

    need_query = [smi for smi in smiles_list if smi not in score_cache]

    if need_query:
        scores, details = oracle(need_query)
        for smi, s, d in zip(need_query, scores, details):
            score_cache[smi] = s
            detail_cache[smi] = d

    scores_out = [score_cache[smi] for smi in smiles_list]
    details_out = [detail_cache.get(smi, None) for smi in smiles_list]
    return scores_out, details_out

query_oracle_with_local_buffer = query_oracle_with_cache

def as_prompt_text(items):
    if items is None:
        return ""
    if isinstance(items, str):
        return items
    if isinstance(items, (list, tuple, set)):
        return "\n".join(str(item) for item in items if item is not None)
    return str(items)

def extract_fragment_lines(prompt_lines, limit):
    if not prompt_lines or limit <= 0:
        return ""
    fragments = []
    seen = set()
    for line in prompt_lines:
        text = str(line)
        if "FRAG]" not in text:
            continue

        fragment = text.split("FRAG]", 1)[1].split("|", 1)[0].strip()
        if not fragment or fragment in seen:
            continue

        fragments.append(fragment)
        seen.add(fragment)
        if len(fragments) >= limit:
            break
    return "\n".join(item for item in fragments if item)

def choose_ablation_action(ablation_cfg, selector, q_values, cluster_id, n_actions, mol_lm_name=None):
    if ablation_cfg.strategy_mode == "random":
        return random.randrange(n_actions)
    if ablation_cfg.strategy_mode == "fixed":
        if int(ablation_cfg.fixed_action) < 0:
            return 0 if mol_lm_name == "BioT5" else 2
        return int(ablation_cfg.fixed_action)
    return selector.pick(q_values, cluster_id=cluster_id)

class GB_GA_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "graph_ga"
        self.ablation = getattr(args, "ablation_config", get_ablation_config(getattr(args, "ablation", "full")))
        print(f"Using ablation config: {self.ablation.name}")

        self.mol_lm = None
        if args.mol_lm == "GPT-4":
            self.mol_lm = GPT4()
        elif args.mol_lm == "BioT5":
            self.mol_lm = BioT5()
        lm_name = "baseline"
        if args.mol_lm != None:
            lm_name = args.mol_lm
            self.mol_lm.task_mode = self.args.task_mode
        self.scores_list1 = 'QED, SA, JNK3'
        self.scores_list2 = 'QED, SA, GSK3b'
        self.scores_list3 = 'QED, SA, DRD2, JNK3, GSK3b'
        self.score_list = ''
        self.oracle_score_cache = {}
        self.oracle_detail_cache = {}
        if self.args.task_mode == "1":
            self.score_list = self.scores_list1
        elif self.args.task_mode == "2":
            self.score_list = self.scores_list2
        else:
            self.score_list = self.scores_list3
        # /home/yinan/Documents/fenzi/MOLLEO-main-server2/RL+LLM+EA_before_talk_best_res_task1
        self.task_desc1_high = f"Please read the following descriptions of high-scoring molecular fragments and summarize each fragment based on its structural characteristics (e.g., aromatic ring type, linkage, substituents, heteroatom distribution, etc.).\nExplanation not more than eight words, as short as you can. \n\n## For each fragment:\n\n- Describe its main structural features concisely.\n\n## Fragment content:\n\n"

        self.task_desc1_low = f"Please read the following descriptions of low-scoring molecular fragments and summarize each fragment based on its structural characteristics (e.g., aromatic ring type, linkage, substituents, heteroatom distribution, etc.).\nExplanation not more than eight words, as short as you can. \n\n## For each fragment:\n\n- Describe its main structural features concisely.\n\n## Fragment content:\n\n"

        self.task_desc2_high = """
## **Task**: \n\nPlease reiterate the detailed atomic connections for each high-scoring fragment and extract structural patterns.   

## Output in the format: 
First One is original fregments, and the following are the surrounding features.
{Molecule Frag1; disconnected position and surrounding features},
{Molecule Frag2; disconnected position and surrounding features},
...


Do not omit any important high-score details, especially attachment point information. Explanation should summarize the disconnected position and surrounding chemical features based on the attachment point information. Explanation not more than eight words, as short as you can. And no Line Break in this block.
## Fragment content:\n
        """

        self.task_desc2_low = """
## **Task**: \n\nPlease reiterate the detailed atomic connections for each low-scoring fragment and extract structural patterns.  

## Output in the format:  
First One is original fregments, and the following are the surrounding features.
{Molecule Frag1; where it was disconnected and nearby features},
{Molecule Frag2; where it was disconnected and nearby features}, 
...

Do not omit any important low-score details, especially attachment point information. Explanation should summarize where the fragment was disconnected and the nearby chemical features based on the attachment point information. Explanation not more than eight words, as short as you can. And no Line Break in this block.

## Fragment content:\n
        """
        self.smiles_score_clusters = pd.read_csv(settings.csv_path_url, encoding='gbk')
        cluster_idx, clusters, keep_idx = restore_clusters_from_df(self.smiles_score_clusters)
        self.clustering_data = (cluster_idx, clusters, keep_idx)
        self.high_score_mols = self.smiles_score_clusters.sort_values(by='scores', ascending=False).head(int(len(self.smiles_score_clusters)*0.1)).reset_index(drop=True)['smiles'].tolist()
        self.low_score_mols = self.smiles_score_clusters.sort_values(by='scores', ascending=False).tail(int(len(self.smiles_score_clusters)*0.1)).reset_index(drop=True)['smiles'].tolist()
        
        df_sorted = self.smiles_score_clusters.sort_values(by='scores', ascending=False).reset_index(drop=True)

        n = len(df_sorted)
        k = int(n * 0.1)

        mid_start = (n - k) // 2
        mid_end = mid_start + k

        self.mid_score_mols = df_sorted.iloc[mid_start:mid_end]['smiles'].tolist()
        self.mid_scores = df_sorted.iloc[mid_start:mid_end]['scores'].tolist()

        
        self.high_scores = self.smiles_score_clusters.sort_values(by='scores', ascending=False).head(int(len(self.smiles_score_clusters) * 0.1)).reset_index(drop=True)['scores'].tolist()
        self.low_scores = self.smiles_score_clusters.sort_values(by='scores', ascending=False).tail(int(len(self.smiles_score_clusters) * 0.1)).reset_index(drop=True)['scores'].tolist()
        self.init_crossatt_path = settings.model_init_path_url
        self.init_crossatt_path_pre = settings.model_init_predictor_path_url
        self.add_crossatt_path = settings.model_add_path_url
        self.add_crossatt_path_pre = getattr(settings, "model_add_predictor_path_url", None) or settings.model_init_predictor_path_url
        self.high_pool_size = len(self.high_score_mols)
        self.low_pool_size = len(self.low_score_mols)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cross_att= Graph_encoder(
            node_features_1=10,
            edge_features_1=5,
            message_size=100,
            message_passes=3,
            out_features=1
        ).to(device=self.device)
        self.state_dict = torch.load(self.init_crossatt_path, map_location=self.device)
        filtered_state_dict = {
            k: v for k, v in self.state_dict.items()
            if not k.startswith("node_emb") and not k.startswith("edge_emb")
        }
        self.cross_att.load_state_dict(filtered_state_dict, strict=False)
        self.cross_att_predictor = torch.nn.Linear(128, 1).to(device=self.device)
        self.cross_att_predictor.load_state_dict(torch.load(self.init_crossatt_path_pre, map_location=self.device))
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(list(self.cross_att.parameters()) + list(self.cross_att_predictor.parameters()), lr=1e-4)
        self.initial_data = list(zip(self.smiles_score_clusters['smiles'].tolist(), self.smiles_score_clusters['scores'].tolist()))
        self.vocab = list("CNOSPFBrClI=#0123456789()[]@+-%")
        self.cluster_centers = []
        self.cluster_centers_fps = []
        for cluster in self.clustering_data[1]:
            if len(cluster) == 0: continue
            center_mol, center_fp = compute_center_mol(cluster, fp_type="morgan")
            self.cluster_centers.append(center_mol)
            self.cluster_centers_fps.append(center_fp)
        self.buffer_size = 10000
        self.gamma = 0.99 
        self.epsilon_start, self.epsilon_end, self.epsilon_decay = 1.0, 0.05, 0.995
        self.epsilon = self.epsilon_start
        self.update_target_interval = 200
        self.batch_size = 32
        self.state_dim = 20480
        self.q_hidden = 256
        self.n_actions = 4
        self.reward_engine = RewardEngine(
            cluster_centers_fps=self.cluster_centers_fps,
            all_smiles=set(), 
            high_mols=self.high_score_mols,
            low_mols=self.low_score_mols,
            config=RewardConfig(
                target_main=0.9,
                target_tox=0.0,
                target_cluster_sim=0.6,
                novelty_band=(0.3, 0.85),
                normalizer_momentum=0.05,
            )
        )
        self.toxic_scorers = RDKitToxicityScorer(
            weights={"PAINS": 1.0, "BRENK": 1.0, "NIH": 0.7, "ZINC": 0.5},
            lambda_alpha=math.log(2), 
            tox_threshold=0.5
        )

        self.episode = EpisodeManager()
        self.selector = None
        self.replay = None
        self.trainer = None
        self.high_frag_history = []
        self.high_frag_history_maxlen = 10
        self.args = args


    def init_dqn_components(self):
        self.selector = ActionSelector(n_actions=self.n_actions)
        self.replay = ReplayBuffer(capacity=self.buffer_size)
        self.trainer = DQNTrainer(state_dim=self.state_dim, n_actions=self.n_actions, device=self.device)
        
    def surrogate_score(self, mols):
        """
        mols: List[rdkit.Chem.Mol]
        return: List[float]
        """
        self.cross_att.eval()
        self.cross_att_predictor.eval()

        # ✅ 1. Mol → SMILES
        smiles_list = []
        for m in mols:
            if m is None:
                continue
            try:
                s = Chem.MolToSmiles(m)
                if s:
                    smiles_list.append(s)
            except Exception:
                continue

        if len(smiles_list) == 0:
            return []

        # ✅ 2. labels 是 dummy，占位即可
        dummy_labels = [0.0] * len(smiles_list)

        dataset = MolMultiModalDataset(
            smiles_list,
            dummy_labels
        )

        loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=collate_fn
        )

        scores = []
        with torch.no_grad():
            for batch in loader:
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                emb = self.cross_att(
                    mol=batch["mol"],
                    adj_1=batch["adj_1"],
                    nd_1=batch["nd_1"],
                    ed_1=batch["ed_1"],
                    de_1=batch["de_1"],
                    mask_1=batch["mask_1"],
                    bg=batch["bg"],
                    entity_emb=batch["entity_emb"],
                    relation_emb=batch["relation_emb"],
                )

                pred = self.cross_att_predictor(emb).squeeze(-1)
                scores.extend(pred.cpu().tolist())

        return scores

    def _optimize(self, config):
        import os
        print("CWD:", os.getcwd())
        print("gsk3b.pkl exists:", os.path.exists("oracle/gsk3b.pkl"))
        print("gsk3b_current.pkl exists:", os.path.exists("oracle/gsk3b_current.pkl"))
        ablation = self.ablation
        print(f"Active ablation preset: {ablation.name}")

        self.oracle.assign_evaluator(self.args)
        # self.oracle_other.assign_evaluator(self.args)
        pool = joblib.Parallel(n_jobs=self.n_jobs)
        
        if self.smi_file is not None:
            # Exploitation run
            starting_population = self.all_smiles[:config["population_size"]]
        else:
            # Exploration run
            starting_population = np.random.choice(self.all_smiles, config["population_size"])
        self.init_dqn_components()
        if not ablation.use_rl:
            print(f"DQN disabled for ablation '{ablation.name}', strategy_mode={ablation.strategy_mode}.")

        # select initial population
        population_smiles = starting_population
        population_mol = [Chem.MolFromSmiles(s) for s in population_smiles]
        # population_scores, population_scores_detail = self.oracle([Chem.MolToSmiles(mol) for mol in population_mol])
        population_smiles = [Chem.MolToSmiles(mol, canonical=True) for mol in population_mol]
        population_scores, population_scores_detail = query_oracle_with_cache(
            self.oracle,
            population_smiles,
            self.oracle_score_cache,
            self.oracle_detail_cache
        )
        
        print(f"Length of population_mol:{len(population_mol)}")   
        patience = 0
        no_improve_rounds = 0
        flag = 1
        iteration = 0
        offspring_mol_temp = []
        parent = []
        parents_score_detail = []
        parent_map = {}
        total_flag = 0
        past_generation_total = []
        past_generation_total_low = []
        while True:
            if len(self.oracle) > 100:
                self.sort_buffer()
                old_score = np.mean([item[1][0] for item in list(self.mol_buffer.items())[:100]])
            else:
                old_score = 0

            # new_population
            mating_tuples = make_mating_pool(population_mol, population_scores, population_scores_detail, config["population_size"])

            iteration += 1
            print("High pool size:", len(self.high_score_mols))
            print("Low pool size:", len(self.low_score_mols))
            print("High score range:", min(self.high_scores), max(self.high_scores))
            print("Low score range:", min(self.low_scores), max(self.low_scores))
            print("High mean:", np.mean(self.high_scores))
            print("Low mean:", np.mean(self.low_scores))
            res = summarize_brics_by_groups(
                high_mols=self.high_score_mols,
                low_mols=self.low_score_mols,
                high_scores=self.high_scores,
                low_scores=self.low_scores,
                k_high=ablation.dkb_high_k if ablation.use_positive_memory else 0,
                k_low=ablation.dkb_low_k if ablation.use_negative_memory else 0,
                exploration_temp=None,
                alpha=1.0,
                replacement_threshold=5,
                min_count=1,
                min_high_count=1,
                frag_history=self.high_frag_history,
                confidence_c=5.0,
                temporal_weight=1.0,
            )
            print("Chosen high frags:", res["high"]["chosen_frags"])
            print("Chosen low frags:", res["low"]["chosen_frags"])
            print("Num high ctx:", len(res["high"]["ctx"]))
            print("Num low ctx:", len(res["low"]["ctx"]))
            print("High prompt lines:", res["high"]["prompt_lines"][:5])
            print("Low prompt lines:", res["low"]["prompt_lines"][:5])
            h_prompt_lines = res["high"]["prompt_lines"] if ablation.use_positive_memory else []
            l_prompt_lines = res["low"]["prompt_lines"] if ablation.use_negative_memory else []
            combine_prompt_high = h_prompt_lines
            combine_prompt_low = l_prompt_lines
            global CURRENT_HIGH_POOL_FEATURE, CURRENT_LOW_POOL_FEATURE
            state_high_mols = self.high_score_mols if ablation.use_positive_memory else []
            state_low_mols = self.low_score_mols if ablation.use_negative_memory else []
            CURRENT_HIGH_POOL_FEATURE = get_mol_pool_feature(state_high_mols, fp_type="morgan")
            CURRENT_LOW_POOL_FEATURE = get_mol_pool_feature(state_low_mols, fp_type="morgan")
            # if self.args.mol_lm == 'GPT-4':
            #     h_prompt = self.task_desc1_high + "\n" + "\n".join(h_prompt_lines) + "\n" + self.task_desc2_high
            #     _, r = query_LLM(h_prompt)
            #     if r is None:
            #         _, r = query_LLM(h_prompt)
            #     combine_prompt_high = re.findall(r'{(.*?)}', r)
            #     # combine_prompt_high = r
            #     if combine_prompt_high is None or not combine_prompt_high:
            #         combine_prompt_high = h_prompt_lines
            #     l_prompt = self.task_desc1_low + "\n" + "\n".join(l_prompt_lines) + "\n" + self.task_desc2_low
                
            #     _, r = query_LLM(l_prompt)
            #     if r is None:
            #         _, r = query_LLM(l_prompt)
            #     combine_prompt_low = re.findall(r'\{.*?\}', r, flags=re.S)
            #     # combine_prompt_low = r
            #     if combine_prompt_low is None or not combine_prompt_high:
            #         combine_prompt_low = l_prompt_lines
            #     combine_prompt_high  = "{High Score Molecule Frag; Features}\n" + "\n".join(combine_prompt_high) 
            #     combine_prompt_low  = "{Low Score Molecule Frag; Features} \n" + "\n".join(combine_prompt_low) 
            #     print("combine_prompt_high:", combine_prompt_high)
            if self.args.mol_lm == 'GPT-4':
                print(f"h_prompt_lines: {h_prompt_lines}")
                combine_prompt_high = extract_fragment_lines(h_prompt_lines, 10)
                combine_prompt_low = extract_fragment_lines(l_prompt_lines, 10)
            if self.args.mol_lm == 'BioT5':
                print(f"h_prompt_lines: {h_prompt_lines}")
                combine_prompt_high = extract_fragment_lines(h_prompt_lines, 10)
                combine_prompt_low = extract_fragment_lines(l_prompt_lines, 10)
            offspring_mol = []
            action_records = []
            parent_main_scores = {}
            last_state_tensor = None
            last_action = None
            last_next_parents = None
            if self.args.mol_lm != 'GPT-4':
                if flag == 1:
                    parent = [random.choice(mating_tuples) for _ in range(4)]
                else:
                    temp = [random.choice(mating_tuples) for _ in range(2)]
                    parent = parent[2:] + temp  # [c, d] + [new1, new2]
                parent_mol = [t[1] for t in parent]
                parents_mols = parent_mol[:2]
                parent_score = [t[0] for t in parent]
                parent_scores = parent_score[:2]
                parent_score_detail = [t[2] for t in parent]
                parent_scores_detail = parent_score_detail[:2]

                state = build_state_dict(
                    parents=parents_mols,
                    clustering_data=self.clustering_data[1],
                    high_score_mols=state_high_mols,
                    low_score_mols=state_low_mols,
                    fp_type="morgan"
                )
                s = torch.cat([state["parent_fp"], state["cluster_center"], state["high_pool"], state["low_pool"]], dim=0).to(self.device)

                cid = nearest_cluster_id(parents_mols[0], parents_mols[1], self.clustering_data[1], fp_type="morgan")
                with torch.no_grad():
                    q_vals = self.trainer.q(s.unsqueeze(0))
                action = choose_ablation_action(ablation, self.selector, q_vals, cid, self.n_actions, self.args.mol_lm)
                last_state_tensor = s
                last_action = action
                last_next_parents = parents_mols
            
            if self.args.mol_lm == 'GPT-4':
                for _ in range(config["offspring_size"]):
                    total_flag += 1
                    parent = [random.choice(mating_tuples) for _ in range(2)]
                    parents_mols = [t[1] for t in parent]
                    parent_scores = [t[0] for t in parent]
                    parent_scores_detail = [t[2] for t in parent]
                    for p_mol, p_score in zip(parents_mols, parent_scores):
                        parent_main_scores[p_mol] = p_score

                    state = build_state_dict(
                        parents=parents_mols,
                        clustering_data=self.clustering_data[1],
                        high_score_mols=state_high_mols,
                        low_score_mols=state_low_mols,
                        fp_type="morgan"
                    )
                    s = torch.cat([state["parent_fp"], state["cluster_center"], state["high_pool"], state["low_pool"]], dim=0).to(self.device)

                    cid = nearest_cluster_id(parents_mols[0], parents_mols[1], self.clustering_data[1], fp_type="morgan")
                    with torch.no_grad():
                        q_vals = self.trainer.q(s.unsqueeze(0))
                    action = choose_ablation_action(ablation, self.selector, q_vals, cid, self.n_actions, self.args.mol_lm)
                    last_state_tensor = s
                    last_action = action
                    last_next_parents = parents_mols

                    temp_mols = []
                    temp_smiles = []
                    temp_scores = []
                    temp_details = []
                    attempts = 0
                    max_retry = 2
                    accepted = False
                    cand_mols, cand_smiles, cand_scores, cand_score_detailis = [], [], [], []
                    past_generation = []
                    best_oracle_candidate = None
                    while attempts < max_retry and not accepted:
                        attempts += 1
                        off = None
                        past_generation_total = dedup_nested_list(past_generation_total)
                        off, _ = self.mol_lm.edit(
                            parents_mols, parent_scores, parent_scores_detail,
                            config["mutation_rate"],
                            as_prompt_text(combine_prompt_high), as_prompt_text(combine_prompt_low),
                            past_generation,
                            past_generation_total if ablation.use_history_prompt else [],
                            past_generation_total_low if ablation.use_history_prompt else [],
                            action, iteration
                        )
                        if off is None:
                            print("no valid candidates!")
                            continue
                        if isinstance(off, Mol):
                            off = [off]
                        if len(off) == 0:
                            print("no valid candidates!")
                            continue
                        cand_mols, cand_smiles = [], []
                        seen_smiles = set()
                        for item in off:
                            if item is None:
                                continue
                            smi = Chem.MolToSmiles(item, canonical=True)
                            if not smi or smi in seen_smiles:
                                continue
                            seen_smiles.add(smi)
                            cand_mols.append(item)
                            cand_smiles.append(smi)

                        if len(cand_mols) == 0:
                            print("no valid candidates after dedup!")
                            continue

                        off_scores = self.surrogate_score(cand_mols)
                        if len(off_scores) < len(cand_mols):
                            off_scores = list(off_scores) + [float("-inf")] * (len(cand_mols) - len(off_scores))
                        score_details = ["surrogate"] * len(cand_mols)
                        temp_mols += cand_mols
                        temp_smiles += cand_smiles
                        temp_scores += off_scores[:len(cand_mols)]
                        temp_details += score_details
                        

                        cand_pairs = list(zip(cand_mols, [float(s) for s in off_scores[:len(cand_mols)]], cand_smiles))
                        if cand_pairs:
                            selected_pairs = cand_pairs[:1]

                            topk_smiles = [item[2] for item in selected_pairs]
                            oracle_scores, oracle_details = query_oracle_with_local_buffer(
                                self.oracle,
                                topk_smiles,
                                self.oracle_score_cache,
                                self.oracle_detail_cache
                            )

                            oracle_ranked = [
                                (selected_pairs[i][0], oracle_scores[i], selected_pairs[i][2], oracle_details[i])
                                for i in range(len(selected_pairs))
                            ]
                            best_off, best_score, best_smi, cand_scor_detaili = max(oracle_ranked, key=lambda x: x[1])
                            print(f"best mol: {best_off, best_score, best_smi, cand_scor_detaili}, selected_for_oracle: {len(oracle_ranked)}, no_improve_rounds: {no_improve_rounds}")
                            if best_score <= min(parent_scores):
                                low_record = format_generation_record(best_smi, best_score, cand_scor_detaili)
                                past_generation_total_low = update_generation_history(
                                    past_generation_total_low, low_record, topk=3, reverse=False
                                )
                            if (best_oracle_candidate is None) or (best_score > best_oracle_candidate[1]):
                                best_oracle_candidate = (best_off, best_score, best_smi, cand_scor_detaili)
                        else:
                            print("no valid candidates after surrogate screening!")
                            continue
                        offspring_mol.append(best_off)
                        parent_map[best_smi] = parents_mols
                        action_records.append({
                            "state": s.detach().cpu(),
                            "action": action,
                            "reward": float(best_score - max(parent_scores)),
                            "parents": parents_mols,
                        })
                        print(f"accepted: off_score={best_score}, parent_best={max(parent_scores)}, smi={best_smi}")
                        accepted = True
                        past_generation = []

                        record = format_generation_record(best_smi, best_score, cand_scor_detaili)
                        if best_score > max(parent_scores):
                            no_improve_rounds = 0
                            past_generation_total = update_generation_history(
                                past_generation_total, record, topk=3, reverse=True
                            )
                        else:
                            no_improve_rounds += 1
                            past_generation_total_low = update_generation_history(
                                past_generation_total_low, record, topk=3, reverse=False
                            )
                        
                    if len(temp_mols) == 0:
                        no_improve_rounds += 1
                        continue

                if not accepted:
                    no_improve_rounds += 1
                    if best_oracle_candidate is None:
                        continue

                    best_off, best_score, best_smi, cand_scor_detaili = best_oracle_candidate
                    offspring_mol.append(best_off)
                    parent_map[best_smi] = parents_mols
                    action_records.append({
                        "state": s.detach().cpu(),
                        "action": action,
                        "reward": float(best_score - max(parent_scores)),
                        "parents": parents_mols,
                    })

                    record = format_generation_record(best_smi, best_score, cand_scor_detaili)

                    if best_score > max(parent_scores):
                        past_generation_total = update_generation_history(
                            past_generation_total, record, topk=3, reverse=True
                        )
                    else:
                        past_generation_total_low = update_generation_history(
                            past_generation_total_low, record, topk=3, reverse=False
                        )


            elif self.args.mol_lm == 'BioT5':
                # ===== 新增：初始化全局历史记录（保留Top 2）=====
                past_generation_total = []  # 与GPT-4相同的全局历史记录
                
                top_smi = get_best_mol(population_scores, population_mol) 
                offspring_mol = [reproduce(mating_tuples, config["mutation_rate"]) for _ in range(config["offspring_size"])]
                offspring_mol = [item[0] for item in offspring_mol]
                editted_smi = []
                biot5_parent_map = {}
                biot5_parent_scores = {}
                for m in offspring_mol:
                    if m != None:
                        smi = Chem.MolToSmiles(m, canonical=True)
                        editted_smi.append(smi)
                
                ii = 0
                idxs = np.argsort(population_scores)[::-1]
                # ===== 优化点1：补充分子时应用GPT-4的重试+历史机制 =====
                while len(editted_smi) < self.args.bin_size:
                    if ii == len(idxs):
                        print("exiting while loop before filling up bin..........")
                        break
                        
                    # 获取父代分子和分数（用于比较基准）
                    parent_mols = population_mol[idxs[ii]]
                    parent_scores = population_scores[idxs[ii]]
                    parent_smiles = Chem.MolToSmiles(parent_mols)
                    biot5_parent_scores[parent_mols] = parent_scores
                    
                    attempts = 0
                    max_retry = 2  # 与GPT-4相同的重试次数
                    accepted = False
                    best_candidate = None
                    
                    # ===== 优化点2：为每个补充分子创建独立历史上下文 =====
                    past_generation = []  # 临时历史（单次生成尝试）
                    
                    while attempts < max_retry and not accepted:
                        attempts += 1
                        # ===== 优化点3：与GPT-4完全一致的edit调用参数 =====
                        past_generation_total = dedup_nested_list(past_generation_total)  # 去重全局历史
                        edited_mols = self.mol_lm.edit(
                            [parent_mols],
                            as_prompt_text(combine_prompt_high),
                            as_prompt_text(combine_prompt_low),
                            past_generation,
                            past_generation_total if ablation.use_history_prompt else [],
                            action
                        )
                        
                        # ===== 优化点4：严格的空结果处理（与GPT-4一致）=====
                        if edited_mols is None or len(edited_mols) == 0:
                            print(f"Attempt {attempts}: No valid candidates from sedit!")
                            continue
                        edited_mols = edited_mols[0] # 这里是因为 传入的是5个分子的list的list
                        if not edited_mols:
                            print(f"No new mol generated!")
                            continue
                        print(f"Attempt {attempts}: Generated {len(edited_mols)} casndidates, {edited_mols}.")
                        # 评估候选分子
                        edited_smiles = [Chem.MolToSmiles(m) for m in edited_mols if m is not None]
                        # edited_scores, score_details = self.oracle_other(edited_smiles)
                        edited_scores = self.surrogate_score(edited_mols)
                        score_details = ["surrogate"] * len(edited_scores)

                        # 选择当前尝试中最好的候选
                        cand_pairs = list(zip(edited_mols, edited_scores, edited_smiles, score_details))
                        best_mol, best_score, best_smi, best_detail = max(cand_pairs, key=lambda x: x[1])
                        
                        # ===== 优化点5：GPT-4式选择策略 =====
                        if best_score > parent_scores:  # 严格优于父代
                            print(f"Accepted supplement: {best_score:.3f} > parent {parent_scores:.3f}, SMILES={best_smi}")
                            editted_smi.append(best_smi)
                            biot5_parent_map[Chem.MolToSmiles(best_mol, canonical=True)] = [parent_mols]
                            accepted = True
                            
                            # ===== 优化点6：更新全局历史（GPT-4格式）=====
                            past_generation_total.append(
                                f"Smiles Mol: {best_smi}, Total score: {best_score}, "
                                f"Score details: {' '.join(best_detail)}"
                            )
                            past_generation_total.sort(key=extract_score, reverse=True)
                            past_generation_total = past_generation_total[:2]  # 保留Top 2
                        else:
                            # 记录当前最佳用于降级接受
                            if best_candidate is None or best_score > best_candidate[1]:
                                best_candidate = (best_mol, best_score, best_smi, best_detail)
                            past_generation.append(f"{best_smi} {best_score} {' '.join(best_detail)}")
                    
                    # ===== 优化点7：降级接受机制（GPT-4逻辑）=====
                    if not accepted and best_candidate is not None:
                        best_mol, best_score, best_smi, best_detail = best_candidate
                        print(f"Degraded accept: {best_score:.3f} < parent {parent_scores:.3f}, SMILES={best_smi}")
                        editted_smi.append(best_smi)
                        biot5_parent_map[Chem.MolToSmiles(best_mol, canonical=True)] = [parent_mols]
                        
                        # 同样更新全局历史
                        past_generation_total.append(
                            f"Smiles Mol: {best_smi}, Total score: {best_score}, "
                            f"Score details: {' '.join(best_detail)}"
                        )
                        past_generation_total.sort(key=extract_score, reverse=True)
                        past_generation_total = past_generation_total[:2]
                    
                    ii += 1

                # ===== 保留BioT5核心逻辑：相似度筛选 =====
                sim = get_fp_scores(editted_smi, top_smi)
                print("fp_scores_to_top", sim)
                sorted_idx = np.argsort(np.squeeze(sim))[::-1][:config["offspring_size"]]
                print("top 70", sorted_idx)
                editted_smi = np.array(editted_smi)[sorted_idx].tolist()
                offspring_mol = [Chem.MolFromSmiles(s) for s in editted_smi]
                print("len offspring_mol", len(offspring_mol))
            if self.args.mol_lm == 'BioT5':
                parent_main_scores = biot5_parent_scores
                parent_map.update(biot5_parent_map)
            offspring_mol = list(set(offspring_mol))
            # add new_population
            population_mol += offspring_mol
            population_mol = self.sanitize(population_mol)

            # stats
            old_scores = population_scores

            population_smiles = [Chem.MolToSmiles(mol, canonical=True) for mol in population_mol]
            population_scores, population_scores_detail = query_oracle_with_cache(
                self.oracle,
                population_smiles,
                self.oracle_score_cache,
                self.oracle_detail_cache
            )
            population_score_map = build_score_map(population_smiles, population_scores)

            valid_offspring_mol = []
            valid_off_scores = []

            for mol in offspring_mol:
                if mol is None:
                    continue
                smi = Chem.MolToSmiles(mol, canonical=True)
                if smi in population_score_map:
                    score = population_score_map[smi]
                    valid_offspring_mol.append(mol)
                    valid_off_scores.append(score)
                    offspring_mol_temp.append((mol, score))

            toxic_out = self.toxic_scorers.score_mol_list(population_mol)
            top10_main = float(np.mean(sorted(population_scores, reverse=True)[:10]))
            if not ablation.use_reward_memory:
                self.reward_engine.frag_weights = {}
            reward_scores = self.reward_engine.get_reward(
                population_mol,
                population_scores,
                toxic_out["probs"],
                parent_map,
                parent_main_scores,
                top10_main
            )
            print(f"population_scores: {len(population_scores)}, offspring_mol:{len(offspring_mol)}, valid_offspring:{len(valid_offspring_mol)}")
            if ablation.use_incremental_retrain and iteration % 3 == 0:
                if len(offspring_mol_temp) == 0:
                    print("Skip LoRA retraining because offspring_mol_temp is empty.")
                    continue_retrain = False
                else:
                    continue_retrain = True
            else:
                continue_retrain = False
                if iteration % 3 == 0:
                    print(f"Skip LoRA retraining for ablation '{ablation.name}'.")
            if continue_retrain:
                if iteration != 1 and iteration != 2 and iteration != 3:
                    self.cross_att.load_state_dict(torch.load(self.add_crossatt_path, map_location=self.device))
                    self.cross_att_predictor.load_state_dict(torch.load(self.add_crossatt_path_pre, map_location=self.device))
                for p in self.cross_att.parameters():
                    p.requires_grad = False
                for p in self.cross_att_predictor.parameters():
                    p.requires_grad = True

                att_self_module = self.cross_att.cross_att.att_self
                lora_wrapped = _replace_linear_with_lora(att_self_module, r=4, alpha=8, dropout=0.05)
                lora_params = collect_lora_trainable_params(lora_wrapped)
                print('length of offspring_mol_temp:',len(offspring_mol_temp))
                print("Retrain the cross attention part with LoRA.....")
                recent_experience = offspring_mol_temp

                training_batch = recent_experience
                smiles_batch = []
                labels_batch = []
                for mol, score in training_batch:
                    try:
                        smiles_batch.append(Chem.MolToSmiles(mol, canonical=True))
                        labels_batch.append(score)
                    except Exception:
                        continue

                incremental_dataset = MolMultiModalDataset(smiles_batch, labels_batch)
                incremental_loader = DataLoader(incremental_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

                self.cross_att.train()
                self.cross_att_predictor.train()

                trainable_params = list(self.cross_att_predictor.parameters()) + lora_params
                inc_optimizer = torch.optim.AdamW(trainable_params, lr=2e-4, weight_decay=1e-4)

                for epoch in tqdm(range(5)):
                    total_loss = 0.0
                    n_batches = 0
                    for batch in incremental_loader:
                        batch = {k: v.to(device=self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                        output_emb = self.cross_att(
                            mol=batch["mol"],
                            adj_1=batch["adj_1"],
                            nd_1=batch["nd_1"],
                            ed_1=batch["ed_1"],
                            de_1=batch["de_1"],
                            mask_1=batch["mask_1"],
                            bg=batch["bg"],
                            entity_emb=batch["entity_emb"],
                            relation_emb=batch["relation_emb"]
                        ).to(self.device)

                        pred = self.cross_att_predictor(output_emb).squeeze(-1).to(self.device)
                        loss = self.criterion(pred, batch["label"])

                        inc_optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                        inc_optimizer.step()

                        total_loss += loss.item()
                        n_batches += 1

                    avg_loss = total_loss / max(1, n_batches)
                    print(f"Epoch {epoch + 1}/10, Average Loss: {avg_loss:.4f}")

                merge_lora_and_unwrap(lora_wrapped)

                torch.save(self.cross_att.state_dict(), self.add_crossatt_path)
                torch.save(self.cross_att_predictor.state_dict(), self.add_crossatt_path_pre)
                print(f"len of population right now: {len(offspring_mol_temp)}")
                offspring_mol_temp = []

            if ablation.use_rl and self.replay is not None and self.trainer is not None and len(self.replay) >= self.batch_size:
                train_loss = self.trainer.train_step(self.replay, self.batch_size)
                if train_loss is not None:
                    print(f"DQN train loss: {train_loss:.4f}")
                if iteration % self.update_target_interval == 0:
                    self.trainer.tar.load_state_dict(self.trainer.q.state_dict())
            if self.args.mol_lm == 'GPT-4':
                next_parents = last_next_parents
            else:
                candidates = [(parent_mol[2], parent_score[2]), (parent_mol[3], parent_score[3])]
                next_parents = [candidates[0][0], candidates[1][0]]
            
            if ablation.update_positive_memory:
                self.high_score_mols, self.high_scores = update_high_pool(
                    self.high_score_mols,
                    self.high_scores,
                    valid_offspring_mol,
                    valid_off_scores,
                    self.high_pool_size
                )
            current_high_frags = set()
            if ablation.update_positive_memory:
                for m in self.high_score_mols:
                    mol = ensure_mol(m)
                    try:
                        current_high_frags.update(get_brics_fragments(mol))
                    except Exception:
                        continue

                self.high_frag_history.append(current_high_frags)
                self.high_frag_history = self.high_frag_history[-self.high_frag_history_maxlen:]

            if ablation.update_negative_memory:
                self.low_score_mols, self.low_scores = update_low_pool(
                    self.low_score_mols,
                    self.low_scores,
                    valid_offspring_mol,
                    valid_off_scores,
                    self.low_pool_size
                )

            state_high_mols = self.high_score_mols if ablation.use_positive_memory else []
            state_low_mols = self.low_score_mols if ablation.use_negative_memory else []
            CURRENT_HIGH_POOL_FEATURE = get_mol_pool_feature(state_high_mols, fp_type="morgan")
            CURRENT_LOW_POOL_FEATURE = get_mol_pool_feature(state_low_mols, fp_type="morgan")

            if ablation.use_reward_memory:
                self.reward_engine.refresh_memory(
                    high_mols=state_high_mols,
                    low_mols=state_low_mols,
                    recompute_frag_weights=True
                )
            else:
                self.reward_engine.frag_weights = {}
            if len(valid_offspring_mol) > 0:
                offspring_toxic_out = self.toxic_scorers.score_mol_list(valid_offspring_mol)
                offspring_reward = self.reward_engine.get_reward(
                    valid_offspring_mol,
                    valid_off_scores,
                    offspring_toxic_out["probs"],
                    parent_map,
                    parent_main_scores,
                    top10_main
                )
            else:
                offspring_reward = []

            self.reward_engine.refresh_memory(
                all_smiles=[Chem.MolToSmiles(m, canonical=True) for m in population_mol if m is not None],
                recompute_frag_weights=False
            )
            s2 = None
            if next_parents is not None:
                next_state = build_state_dict(
                    parents=next_parents,
                    clustering_data=self.clustering_data[1],
                    high_score_mols=state_high_mols,
                    low_score_mols=state_low_mols,
                    fp_type="morgan"
                )
                s2 = torch.cat([next_state["parent_fp"], next_state["cluster_center"], next_state["high_pool"], next_state["low_pool"]], dim=0).to(self.device)
            if len(offspring_reward) == 0:
                offspring_reward = reward_scores[-config["offspring_size"]:] if len(reward_scores) > 0 else [0.0]
            done = self.episode.step(offspring_reward)
            next_replay_state = None if done or s2 is None else s2.detach().cpu()
            if ablation.use_rl and self.args.mol_lm == 'GPT-4' and action_records:
                for rec in action_records:
                    self.replay.push(
                        rec["state"],
                        rec["action"],
                        rec["reward"],
                        next_replay_state,
                        done
                    )
            elif ablation.use_rl and last_state_tensor is not None and last_action is not None:
                self.replay.push(
                    last_state_tensor.detach().cpu(),
                    last_action,
                    offspring_reward,
                    next_replay_state,
                    done
                )
            if done:
                self.episode.reset()
            high_tuples = list(zip(self.high_scores, self.high_score_mols))
            high_tuples = sorted(high_tuples, key=lambda x: x[0], reverse=True)[:len(self.high_score_mols)]
            self.high_score_mols = [t[1] for t in high_tuples]
            self.high_scores = [t[0] for t in high_tuples]

            low_tuples = list(zip(self.low_scores, self.low_score_mols))
            low_tuples = sorted(low_tuples, key=lambda x: x[0], reverse=False)[:len(self.low_score_mols)]
            self.low_score_mols = [t[1] for t in low_tuples]
            self.low_scores = [t[0] for t in low_tuples]
            
            population_tuples = list(zip(population_scores, population_mol))
            population_tuples = sorted(population_tuples, key=lambda x: x[0], reverse=True)[:config["population_size"]]
            population_mol = [t[1] for t in population_tuples]
            population_scores = [t[0] for t in population_tuples]

            flag = 0 if flag == 1 else 1

            ### early stopping
            if len(self.oracle) > 100:
                self.sort_buffer()
                new_score = np.mean([item[1][0] for item in list(self.mol_buffer.items())[:100]])
                # import ipdb; ipdb.set_trace()
                if (new_score - old_score) < 1e-3:
                    patience += 1
                    if patience >= self.args.patience:
                        self.log_intermediate(finish=True)
                        print('convergence criteria met, abort ...... ')
                        break
                else:
                    patience = 0

                old_score = new_score
                
            if self.finish:
                break
