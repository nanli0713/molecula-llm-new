import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
from crossAtt import SGATT, MolMultiModalDataset

initial_smiles = ["CCO"] * 30000  
initial_labels = np.random.rand(30000).tolist()
memory = {
    "smiles": initial_smiles.copy(),
    "labels": initial_labels.copy(),
}
max_memory_size = 50000  

def update_memory(memory, new_smiles, new_labels, max_size):
    total = len(memory["smiles"]) + len(new_smiles)
    if total > max_size:
        keep = max_size - len(new_smiles)
        memory["smiles"] = memory["smiles"][-keep:]
        memory["labels"] = memory["labels"][-keep:]
    memory["smiles"].extend(new_smiles)
    memory["labels"].extend(new_labels)
    return memory


def incremental_train(model, predictor, memory, new_smiles, new_labels, epochs=2, batch_size=64, old_sample_ratio=1.0):
    memory = update_memory(memory, new_smiles, new_labels, max_memory_size)
    n_new = len(new_smiles)
    n_old = int(n_new * old_sample_ratio)
    old_indices = random.sample(range(len(memory["smiles"]) - n_new), min(n_old, len(memory["smiles"]) - n_new))
    sel_smiles = [memory["smiles"][i] for i in old_indices] + new_smiles
    sel_labels = [memory["labels"][i] for i in old_indices] + new_labels
    dataset = MolMultiModalDataset(sel_smiles, sel_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: {
        k: torch.cat([d[k] for d in batch]) if '1' in k else [d[k] for d in batch] for k in batch[0]
    })
    optimizer = optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=5e-5)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        predictor.train()
        total_loss = 0
        for batch in loader:
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
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"Incremental Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
    return memory

if __name__ == "__main__":
    model = SGATT()
    predictor = nn.Linear(128, 1)
    model.load_state_dict(torch.load("sgatt_init.pth"))
    predictor.load_state_dict(torch.load("predictor_init.pth"))

    for iter in range(100):  
        new_smiles = ea_llm_generate_batch(num=1000)
        new_dataset = MolMultiModalDataset(new_smiles, [0.0] * len(new_smiles))  
        new_loader = DataLoader(new_dataset, batch_size=128, collate_fn=lambda batch: {
            k: torch.cat([d[k] for d in batch]) if '1' in k else [d[k] for d in batch] for k in batch[0]
        })
        new_preds = []
        with torch.no_grad():
            model.eval()
            predictor.eval()
            for batch in new_loader:
                new_emb = model(
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
                pred = predictor(new_emb).squeeze(-1).cpu().numpy()
                new_preds.extend(pred)
        memory = incremental_train(model, predictor, memory, new_smiles, new_preds, epochs=2, batch_size=64, old_sample_ratio=1.0)
        torch.save(model.state_dict(), f"sgatt_iter_{iter}.pth")
        torch.save(predictor.state_dict(), f"predictor_iter_{iter}.pth")
        print(f"Iteration {iter + 1} completed.")