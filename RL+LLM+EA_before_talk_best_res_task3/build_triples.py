from rdkit import Chem
from rdkit.Chem import BRICS
import pandas as pd
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
import random
def extract_fragment_bond_triples(smiles):
    triples = []
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return [], []
    frag_mol = BRICS.BreakBRICSBonds(mol)
    idx_tuple_list = Chem.GetMolFrags(frag_mol)
    frag_smiles = [Chem.MolFragmentToSmiles(frag_mol, idxs, isomericSmiles=True) for idxs in idx_tuple_list]

    frag_dummy = []
    for idxs in idx_tuple_list:
        dummies = set()
        for i in idxs:
            atom = frag_mol.GetAtomWithIdx(i)
            if atom.GetAtomicNum() == 0:
                dummies.add(atom.GetAtomMapNum())
        frag_dummy.append(dummies)

    dummy_map = {}
    for idx, dummy_set in enumerate(frag_dummy):
        for d in dummy_set:
            dummy_map.setdefault(d, []).append(idx)
    bonds = set()
    for d, frags in dummy_map.items():
        if len(frags) == 2:
            i, j = sorted(frags)
            bonds.add((i, j, d))
    for i, j, d in bonds:
        key = f"bond_{d}"
        triples.append((frag_smiles[i], key, frag_smiles[j]))
    return triples, frag_smiles

def build_kg_triples(smiles_list, score_list):
    triples = []
    for smi in smiles_list:
        bond_triples, frags = extract_fragment_bond_triples(smi)
        triples.extend(bond_triples)
        for frag in frags:
            triples.append([frag, 'part_of', smi])
    for smi, scor in zip(smiles_list, score_list):
        triples.append([smi, 'its_score', scor])
    return triples

def train_kg_embedding(triples_path, n_epochs=20):
    kg_factory = TriplesFactory.from_path(triples_path)
    result = pipeline(
        training=kg_factory,
        model='RotatE', 
        model_kwargs=dict(embedding_dim=128), 
        training_kwargs=dict(num_epochs=n_epochs),
        training_loop='sLCWA', 
        device='cuda'
    )
    entity_emb = result.model.entity_representations[0](torch.arange(len(kg_factory.entity_to_id))).detach().cpu().numpy()
    relation_emb = result.model.relation_representations[0](torch.arange(len(kg_factory.relation_to_id))).detach().cpu().numpy()
    return entity_emb, relation_emb, kg_factory.entity_to_id, kg_factory.relation_to_id
if __name__ == "__main__":
    kg_name = 'total_kg_triples'
    from tqdm import tqdm

    df = pd.read_csv('/home/yinan/Documents/fenzi/MOLLEO-main-server2/RL+LLM+EA_before_talk_best_res2/main/molleo_multi_pareto/datas_task3/selected_molecules_clustering.csv')
    # smiles_list = df.sort_values(by='scores', ascending=False).head(int(len(df))*0.2).reset_index(drop=True)['smiles']
    smiles_list = df.sort_values(by='scores', ascending=False).reset_index(drop=True)['smiles'].tolist()
    scores_list = df.sort_values(by='scores', ascending=False).reset_index(drop=True)['scores'].tolist()
    all_triples = []
    triples_path = f'/home/yinan/Documents/fenzi/MOLLEO-main-server2/RL+LLM+EA_before_talk_best_res2/main/molleo_multi_pareto/datas_task3/{kg_name}.tsv'
    for smi in tqdm(smiles_list):
        bond_triples, frags = extract_fragment_bond_triples(smi)
        all_triples.extend(bond_triples)
        for frag in frags:
            all_triples.append([frag, 'part_of', smi])

    for smile, score in zip(smiles_list, scores_list):
        all_triples.append([smile, 'its_score', score])

    with open(triples_path, 'w+') as f:
        for h, r, t in all_triples:
            f.write(f"{h}\t{r}\t{t}\n")

    test_size = int(len(all_triples) * 0.2)
    test_triples = random.sample(all_triples, test_size)
    test_path = f'/home/yinan/Documents/fenzi/MOLLEO-main-server2/RL+LLM+EA_before_talk_best_res2/main/molleo_multi_pareto/datas_task3/{kg_name}_test.tsv'
    with open(test_path, 'w+') as f:
        for h, r, t in test_triples:
            f.write(f"{h}\t{r}\t{t}\n")


    train_factory = TriplesFactory.from_path(triples_path)
    test_factory = TriplesFactory.from_path(test_path)
    result = pipeline(
        training=train_factory,
        testing=test_factory,
        model='RotatE',
        model_kwargs=dict(embedding_dim=128),
        training_kwargs=dict(num_epochs=100),
        training_loop='sLCWA',
        device='cuda', # or 'cpu'
        random_seed=2024
    )

    entity2id = train_factory.entity_to_id
    relation2id = train_factory.relation_to_id
    num_entities = len(entity2id)
    num_relations = len(relation2id)
    import torch, pickle
    entity_emb = result.model.entity_representations[0](torch.arange(num_entities)).detach().cpu().numpy()
    relation_emb = result.model.relation_representations[0](torch.arange(num_relations)).detach().cpu().numpy()
    with open(f'/home/yinan/Documents/fenzi/MOLLEO-main-server2/RL+LLM+EA_before_talk_best_res2/main/molleo_multi_pareto/datas_task3/{kg_name}_emb.pkl', 'wb') as f:
        pickle.dump({
            'entity_emb': entity_emb,
            'relation_emb': relation_emb,
            'entity2id': entity2id,
            'relation2id': relation2id
        }, f)