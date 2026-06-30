import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader

from crossAtt import MolMultiModalDataset
from crossAtt_implements2 import Graph_encoder
from trainCrossAtt import collate_fn

# 你已有的模块
# from your_module import (
#     MolMultiModalDataset,
#     collate_fn,
#     Graph_encoder,
# )

# =========================
# 路径配置
# =========================
DATA_DIR = '/home/lachesis/agent-chat/MOLLEO-main-server2/single_objective/data'
PKL_DIR = '/home/lachesis/agent-chat/MOLLEO-main-server2/single_objective/models'
MODEL_DIR = '/home/lachesis/agent-chat/MOLLEO-main-server2/single_objective/weights'

# os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def train_one_task(csv_path):
    task_name = os.path.basename(csv_path).replace('.csv', '').replace('selected_molecules_', '')
    print(f"\n===== Training Cross-Attention for {task_name} =====")

    df = pd.read_csv(csv_path)

    # 兼容列名
    smiles_col = 'smiles' if 'smiles' in df.columns else 'SMILES'
    score_col = 'scores' if 'scores' in df.columns else 'score'

    smiles_all = df[smiles_col].tolist()
    labels_all = df[score_col].astype(float).tolist()

    # Dataset & Loader
    dataset = MolMultiModalDataset(smiles_all, labels_all)
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=8,
        collate_fn=collate_fn
    )
    filename = os.path.basename(csv_path)
    # 模型
    model = Graph_encoder(
        node_features_1=10,
        edge_features_1=5,
        message_size=100,
        message_passes=3,
        out_features=1,
        kg_path=os.path.join(PKL_DIR, filename.replace('.csv', '_emb.pkl'))
    ).to(device)

    predictor = nn.Linear(128, 1).to(device)

    optimizer = optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=1e-4
    )
    criterion = nn.MSELoss()

    # =========================
    # 训练
    # =========================
    for epoch in tqdm(range(4), desc=f"Epochs ({task_name})"):
        model.train()
        predictor.train()
        total_loss = 0

        for batch in loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            output_emb = model(
                mol=batch["mol"],
                adj_1=batch["adj_1"],
                nd_1=batch["nd_1"],
                ed_1=batch["ed_1"],
                de_1=batch["de_1"],
                mask_1=batch["mask_1"],
                bg=batch["bg"],
                entity_emb=batch["entity_emb"],
                relation_emb=batch["relation_emb"]
            )

            pred = predictor(output_emb).squeeze(-1)
            loss = criterion(pred, batch["label"])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Avg Loss: {total_loss / len(loader):.4f}")

    # =========================
    # 简单验证（训练集）
    # =========================
    model.eval()
    predictor.eval()

    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            emb = model(
                mol=batch["mol"],
                adj_1=batch["adj_1"],
                nd_1=batch["nd_1"],
                ed_1=batch["ed_1"],
                de_1=batch["de_1"],
                mask_1=batch["mask_1"],
                bg=batch["bg"],
                entity_emb=batch["entity_emb"],
                relation_emb=batch["relation_emb"]
            )
            pred = predictor(emb).squeeze(-1)
            preds.extend(pred.cpu().numpy())
            labels.extend(batch["label"].cpu().numpy())

    mse = np.mean((np.array(preds) - np.array(labels)) ** 2)
    print(f"Final Train MSE ({task_name}): {mse:.4f}")

    # =========================
    # 保存模型
    # =========================
    model_path = os.path.join(MODEL_DIR, f'{task_name}_graph_encoder.pth')
    pred_path = os.path.join(MODEL_DIR, f'{task_name}_predictor.pth')

    torch.save(model.state_dict(), model_path)
    torch.save(predictor.state_dict(), pred_path)

    print(f"✅ Saved:")
    print(f"  {model_path}")
    print(f"  {pred_path}")


def main():
    csv_files = [
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.endswith('_scores.csv')
    ]

    print(f"Found {len(csv_files)} tasks")

    for csv_path in csv_files:
        train_one_task(csv_path)

    print("\n🎉 All cross-attention models trained!")


if __name__ == "__main__":
    main()
