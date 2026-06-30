
import numpy as np
import pandas as pd
from tqdm import tqdm
from crossAtt import MolMultiModalDataset, compute_mol_features  
from crossAtt_implements import Graph_encoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np


import torch
import dgl
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    collated = {}
    FIXED_DIM = 10 
    keys = batch[0].keys()
    
    for key in keys:
        values = [d[key] for d in batch]
        if key == 'mol':
            collated[key] = values
        elif key == 'bg':
            collated[key] = dgl.batch(values)
        elif key == 'label':
            collated[key] = torch.tensor(values, dtype=torch.float)
        elif key in ['entity_emb', 'relation_emb']:
            values = [v.squeeze() if isinstance(v, torch.Tensor) else v for v in values]
            if isinstance(values, torch.Tensor):
                values = [values]
            collated[key] = torch.stack(values)
        elif key in ['de_1', 'mask_1']:
            values = [v.squeeze() if isinstance(v, torch.Tensor) else torch.tensor(v) for v in values]  
            if isinstance(values, torch.Tensor):
                values = [values]
            collated[key] = torch.stack(values)
        else:
            if not all(isinstance(v, torch.Tensor) for v in values):
                collated[key] = values  
                continue
            values = [v.squeeze() for v in values]

            fix_dim = 10
            num_dims = len(values[0].shape)
            if num_dims == 2:  
                max_N = max(v.shape[0] for v in values)      

                padded = []
                for v in values:                              # v: (N, D)
                    N, D = v.shape

                    if D > FIXED_DIM:                         
                        v = v[:, :FIXED_DIM]
                    elif D < FIXED_DIM:                       
                        pad_feat = torch.zeros(N, FIXED_DIM - D,
                                                device=v.device, dtype=v.dtype)
                        v = torch.cat([v, pad_feat], dim=1)

                    if N < max_N:
                        pad_row = torch.zeros(max_N - N, FIXED_DIM,
                                            device=v.device, dtype=v.dtype)
                        v = torch.cat([v, pad_row], dim=0)

                    padded.append(v)                          # (max_N, 10)

                collated[key] = torch.stack(padded)           # (B, max_N, 10)
            elif num_dims == 3:  
                max_dim1 = max(v.shape[0] for v in values)  # max N
                max_dim2 = max(v.shape[1] for v in values)  # max N
                max_dim3 = max(v.shape[2] for v in values)  # max E 
                padded = []
                for v in values:
                    pad = (0, max_dim3 - v.shape[2], 0, max_dim2 - v.shape[1], 0, max_dim1 - v.shape[0])
                    padded.append(torch.nn.functional.pad(v, pad, value=0))
            elif num_dims == 1:  
                padded = pad_sequence(values, batch_first=True, padding_value=0)
            else:
                raise ValueError(f"Unexpected shape for key '{key}': {values[0].shape}")
            
            collated[key] = torch.stack(padded) 
    
    return collated
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv('/home/yinan/Documents/fenzi/MOLLEO-main-server2/RL+LLM+EA_before_talk_best_res2/main/molleo_multi/datas_task3/selected_molecules_clustering.csv')
    smiles_all = df['smiles'].tolist()
    labels_all = df['scores'].tolist()  

    init_dataset = MolMultiModalDataset(smiles_all, labels_all)
    init_loader = DataLoader(init_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=collate_fn)

    test_smiles = smiles_all[0]
    mol, adj_1, nd_1, ed_1, de_1, mask_1, bg, entity_emb, relation_emb = compute_mol_features(test_smiles)

    model = Graph_encoder(
        node_features_1=10,
        edge_features_1=5,
        message_size=100,
        message_passes=3,
        out_features=1
    ).to(device)

    predictor = nn.Linear(128, 1).to(device)  
    optimizer = optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=1e-4)
    criterion = nn.MSELoss()

    for epoch in tqdm(range(8)):  
        model.train()
        predictor.train()
        total_loss = 0
        for batch in init_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(init_loader)
        print(f"Epoch {epoch + 1}/8, Average Loss: {avg_loss:.4f}")
    model.eval()
    predictor.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in init_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
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
            all_preds.extend(pred.cpu().numpy())  
            all_labels.extend(batch["label"].cpu().numpy())
    mse = np.mean((np.array(all_preds) - np.array(all_labels))**2)
    print(f"Initial validation MSE: {mse:.4f}")

    torch.save(model.state_dict(), "/home/yinan/Documents/fenzi/MOLLEO-main-server2/RL+LLM+EA_before_talk_best_res2/main/molleo_multi/datas_task3/sgatt_init.pth")
    torch.save(predictor.state_dict(), "/home/yinan/Documents/fenzi/MOLLEO-main-server2/RL+LLM+EA_before_talk_best_res2/main/molleo_multi/datas_task3/predictor_init.pth")