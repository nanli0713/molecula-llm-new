import os
import random
import pickle
import torch
import pandas as pd
from tqdm import tqdm
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

from build_triples import extract_fragment_bond_triples

# 你已有的方法
# from your_module import extract_fragment_bond_triples


DATA_DIR = '/home/lachesis/agent-chat/MOLLEO-main-server2/single_objective/data'
MODEL_DIR = '/home/lachesis/agent-chat/MOLLEO-main-server2/single_objective/models'
TSV_DIR = '/home/lachesis/agent-chat/MOLLEO-main-server2/single_objective/tsv_files'
os.makedirs(MODEL_DIR, exist_ok=True)

# random_seed = 2024
# random.seed(random_seed)
# torch.manual_seed(random_seed)


def process_one_csv(csv_path):
    kg_name = os.path.splitext(os.path.basename(csv_path))[0]
    print(f"\n===== Processing {kg_name} =====")

    df = pd.read_csv(csv_path)

    # 兼容不同列名
    if 'smiles' in df.columns:
        smiles_col = 'smiles'
    else:
        smiles_col = 'SMILES'

    if 'scores' in df.columns:
        score_col = 'scores'
    else:
        score_col = 'score'

    df = df.sort_values(by=score_col, ascending=False).reset_index(drop=True)

    smiles_list = df[smiles_col].tolist()
    scores_list = df[score_col].tolist()

    all_triples = []

    # -------- 1. 结构 & 片段三元组 --------
    for smi in tqdm(smiles_list, desc='Extracting triples'):
        bond_triples, frags = extract_fragment_bond_triples(smi)
        all_triples.extend(bond_triples)
        for frag in frags:
            all_triples.append([frag, 'part_of', smi])

    # -------- 2. score 三元组 --------
    for smi, score in zip(smiles_list, scores_list):
        all_triples.append([smi, 'its_score', str(score)])

    # -------- 3. 保存 triples --------
    triples_path = os.path.join(TSV_DIR, f'{kg_name}.tsv')
    with open(triples_path, 'w') as f:
        for h, r, t in all_triples:
            f.write(f"{h}\t{r}\t{t}\n")

    # -------- 4. train / test split --------
    test_size = int(len(all_triples) * 0.2)
    test_triples = random.sample(all_triples, test_size)

    test_path = os.path.join(TSV_DIR, f'{kg_name}_test.tsv')
    with open(test_path, 'w') as f:
        for h, r, t in test_triples:
            f.write(f"{h}\t{r}\t{t}\n")

    train_factory = TriplesFactory.from_path(triples_path)
    test_factory = TriplesFactory.from_path(test_path)

    # -------- 5. 训练 RotatE --------
    result = pipeline(
        training=train_factory,
        testing=test_factory,
        model='RotatE',
        model_kwargs=dict(embedding_dim=128),
        training_kwargs=dict(num_epochs=100),
        training_loop='sLCWA',
        device='cuda',
        random_seed=2024
    )

    # -------- 6. 导出 embedding --------
    entity2id = train_factory.entity_to_id
    relation2id = train_factory.relation_to_id

    num_entities = len(entity2id)
    num_relations = len(relation2id)

    entity_emb = result.model.entity_representations[0](
        torch.arange(num_entities)
    ).detach().cpu().numpy()

    relation_emb = result.model.relation_representations[0](
        torch.arange(num_relations)
    ).detach().cpu().numpy()

    emb_path = os.path.join(MODEL_DIR, f'{kg_name}_emb.pkl')
    with open(emb_path, 'wb') as f:
        pickle.dump({
            'entity_emb': entity_emb,
            'relation_emb': relation_emb,
            'entity2id': entity2id,
            'relation2id': relation2id
        }, f)

    print(f"✅ Saved embedding to {emb_path}")


def main():
    csv_files = [
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.endswith('.csv')
    ]

    print(f"Found {len(csv_files)} csv files")

    for csv_path in csv_files:
        process_one_csv(csv_path)

    print("\n🎉 All models finished!")


if __name__ == "__main__":
    main()
