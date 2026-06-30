import tdc
from tdc.generation import MolGen
import random
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.ML.Cluster import Butina
import datamol as dm
import operator
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.Draw import SimilarityMaps
from rdkit import DataStructs
import random, numpy as np
from rdkit.Chem import rdMolDescriptors
def restore_clusters_from_df(
    df, 
    cluster_col='cluster', 
    smiles_col='smiles'
):

    from collections import defaultdict
    from rdkit import Chem

    keep_idx = df[df[cluster_col].notna()].index.tolist()

    row_index_to_cleaned_pos = {row_idx: pos for pos, row_idx in enumerate(keep_idx)}

    clusters_raw = defaultdict(list)
    for idx in keep_idx:
        c = int(df.loc[idx, cluster_col])
        clusters_raw[c].append(idx)

    cluster_idx = tuple(
        tuple(row_index_to_cleaned_pos[idx] for idx in members)
        for c, members in sorted(clusters_raw.items())
    )

    cleaned_smiles = [df.loc[idx, smiles_col] for idx in keep_idx]
    cleaned_mols = [Chem.MolFromSmiles(smi) for smi in cleaned_smiles]
    clusters = [
        [cleaned_mols[pos] for pos in group]
        for group in cluster_idx
    ]

    return cluster_idx, clusters, keep_idx

def tanimoto_butina_clustering(smiles_or_mols, cutoff=0.7, fp_type="rdk"):

    if isinstance(smiles_or_mols[0], str):
        mols = [Chem.MolFromSmiles(s) for s in smiles_or_mols]
    else:
        mols = list(smiles_or_mols)

    keep_idx, mols_clean = zip(*[(i, m) for i, m in enumerate(mols) if m is not None])
    if len(mols_clean) < len(mols):
        print(f"[warn] 有 {len(mols)-len(mols_clean)} 条 SMILES 解析失败，已跳过")

    if fp_type == "morgan":
        fps = [Chem.GetMorganFingerprintAsBitVect(m, 3, nBits=4096) for m in mols_clean]
    else:  # "rdk"
        fps = [Chem.RDKFingerprint(m) for m in mols_clean]

    dists = []
    n = len(fps)
    for i in range(1, n):
        dists.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i],
                                                        returnDistance=True))

    cluster_indices = Butina.ClusterData(dists, n, cutoff, isDistData=True)
    cluster_mols = [operator.itemgetter(*cid)(mols_clean) for cid in cluster_indices]
    cluster_mols = [[m] if isinstance(m, Chem.Mol) else list(m) for m in cluster_mols]

    return cluster_indices, cluster_mols, list(keep_idx)  

import rdkit
def get_fp(mol, fp_type="morgan"):

    if fp_type == "morgan":
        return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 3, nBits=4096)
    elif fp_type == "rdk":
        return Chem.RDKFingerprint(mol)
    else:
        raise ValueError(f"Unknown fp_type: {fp_type}")

def compute_center_mol(mols, fp_type="morgan"):

    if type(mols) is rdkit.Chem.rdchem.Mol:
        mols = [mols]
    if type(mols[0]) is str:
        mols = [Chem.MolFromSmiles(m) for m in mols]
    fps = [get_fp(m, fp_type) for m in mols]
    n = len(fps)
    sims = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            sims[i, j] = sim
            sims[j, i] = sim
    avg_sims = sims.mean(axis=1)
    center_idx = np.argmax(avg_sims)
    return mols[center_idx], fps[center_idx]  

def assign_new_mol(new_mol, cluster_centers_fps, fp_type="morgan", cutoff=0.825):
    if type(new_mol) is not rdkit.Chem.rdchem.Mol:
        print("clustering assign_new_mol mols:", new_mol)
    fp_new = get_fp(new_mol, fp_type)
    sims = [DataStructs.TanimotoSimilarity(fp_new, fp_cent) for fp_cent in cluster_centers_fps]
    best_idx = int(np.argmax(sims))
    best_sim = sims[best_idx]
    if best_sim < cutoff * 0.2:  
        assign_info = {
            "is_new_cluster": True,
            "reward_bonus": True,
            "best_sim": best_sim,
            "to_cluster": None
        }
    else:
        assign_info = {
            "is_new_cluster": False,
            "reward_bonus": False,
            "best_sim": best_sim,
            "to_cluster": best_idx
        }
    return assign_info
  
import pandas as pd
if __name__ == "__main__":
    df = pd.read_csv("/home/yinan/Documents/fenzi/MOLLEO-main-server2/RL+LLM+EA_before_talk_best_res2/main/molleo_multi/datas_task1/selected_molecules_new.csv")
    picked_smiles = df['smiles'].tolist()

    cutoff = 0.85
    cluster_idx, clusters, keep_idx = tanimoto_butina_clustering(picked_smiles, cutoff=cutoff, fp_type="morgan")

    print(f"有效分子 {len(keep_idx)} 个，得到 {len(clusters)} 个簇")
    for i, cmols in enumerate(clusters[:5], 1):
        print(f"簇 #{i} (size={len(cmols)})")

    df['cluster'] = np.nan

    cluster_labels = [-1] * len(keep_idx)
    for cluster_id, members in enumerate(cluster_idx):
        for idx_in_cleaned in members:
            cluster_labels[idx_in_cleaned] = cluster_id
 
    for idx_in_cleaned, idx_in_df in enumerate(keep_idx):
        df.loc[idx_in_df, 'cluster'] = cluster_labels[idx_in_cleaned]

    df.to_csv('/home/yinan/Documents/fenzi/MOLLEO-main-server2/RL+LLM+EA_before_talk_best_res2/main/molleo_multi/datas_task1/selected_molecules_clustering.csv', index=False)
