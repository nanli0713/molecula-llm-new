import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math
import copy
import pickle
import dgl
from os import fchdir
import torch as th
from torch.nn import init
from dgl import function as fn
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair
from dgl.readout import sum_nodes, broadcast_nodes, softmax_nodes
from rdkit.Chem import rdMolDescriptors, DataStructs, AllChem
from rdkit import Chem
from torch.utils.data import Dataset, DataLoader
from settings import settings
class SGATT(nn.Module):


    def __init__(self, node_fts_1, edge_fts_1,
                 message_size, message_passes, out_fts):
        super(SGATT, self).__init__()
        self.node_fts_1   = node_fts_1
        self.edge_fts_1   = edge_fts_1
        self.message_size = message_size
        self.message_passes = message_passes
        self.out_fts      = out_fts

        self.max_d      = 50
        self.input_dim_drug = 30000
        self.n_layer    = 2
        self.emb_size   = 384
        self.dropout_rate = 0
        self.n_heads    = 1
        self.hid_dim    = 128

        self.hidden_size             = 384
        self.intermediate_size       = 1536
        self.num_attention_heads     = 8
        self.attention_probs_dropout_prob = 0.1
        self.hidden_dropout_prob          = 0.1

        self.emb = Embeddings(self.input_dim_drug, self.emb_size,
                              self.max_d, self.dropout_rate)
        self.d_encoder = Encoder_MultipleLayers(
            self.n_layer, self.hidden_size, self.intermediate_size,
            self.num_attention_heads, self.attention_probs_dropout_prob,
            self.hidden_dropout_prob)
        self.p_encoder = Encoder_MultipleLayers(
            self.n_layer, self.hidden_size, self.intermediate_size,
            self.num_attention_heads, self.attention_probs_dropout_prob,
            self.hidden_dropout_prob)

        self.cross_att = CrossAttentionBlock(
            hid_dim=self.hid_dim, n_heads=self.n_heads,
            dropout=self.dropout_rate)

        self.decoder_trans_mpnn_cat = nn.Sequential(
            nn.Linear(406, 64), nn.ReLU(True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32), nn.ReLU(True),
            nn.Linear(32, 1)                          
        )

        self.decoder_trans_mpnn_sum = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(True),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1)
        )

        self.decoder_1 = nn.Sequential( 
            nn.Linear(50*384, 512), nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128)
        )

        self.decoder_1_ln = nn.Sequential( 
            nn.Linear(50*384, 512), nn.ReLU(True),
            nn.LayerNorm(512),  # ← 只改这一行
            nn.Linear(512, 128)
        )
        self.project_node_feats = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU()
        )
        self.gru = nn.GRU(64, 64)

        attn_fc = nn.Linear(2 * 64, 1, bias=False)
        edge_network1 = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 64 * 64)
        )
        edge_network2 = copy.deepcopy(edge_network1)

        self.gnn_layer = gnn(
            in_feats=64, out_feats=64, attn_fc=attn_fc,
            edge_func1=edge_network1,
            edge_func2=edge_network2,
            aggregator_type='sum')

        self.lstm = th.nn.LSTM(128, 64, 3)

        loaded_dict = pickle.load(
            open(settings.kg_triples_emb_path, 'rb'))
        entity_emb, relation_emb = loaded_dict['entity_emb'], loaded_dict['relation_emb']
        entity_emb = torch.from_numpy(entity_emb).float()
        relation_emb = torch.from_numpy(relation_emb).float()

        atom_emb = torch.randn((118, 128))
        node_emb = torch.cat((atom_emb, entity_emb), 0)
        bond_emb = torch.randn((4, 64))               
        bond_proj = nn.Linear(64, 128) 
        bond_emb = bond_proj(bond_emb)

        edge_emb = torch.cat((bond_emb, relation_emb), 0)

        self.node_emb = nn.Embedding.from_pretrained(node_emb,  freeze=False)
        self.edge_emb = nn.Embedding.from_pretrained(edge_emb,  freeze=False)

        self.morgan_dim = 4096
        self.morgan_proj = nn.Sequential(
            nn.Linear(self.morgan_dim, self.hid_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hid_dim)
        )

    def aggregate_message_1(self, nodes, node_neighbours, edges, mask):
        raise NotImplementedError

    def update_1(self, nodes, messages):
        raise NotImplementedError

    def readout_1(self, hidden_nodes, input_nodes, node_mask):
        raise NotImplementedError

    def readout(self, input_nodes, node_mask):
        raise NotImplementedError

    def final_layer(self, out):
        raise NotImplementedError
    
    def update_kg_embeddings(self, kg_emb_dict):
        entity_emb = torch.from_numpy(kg_emb_dict['entity_emb']).float()
        relation_emb = torch.from_numpy(kg_emb_dict['relation_emb']).float()
        atom_emb = self.node_emb.weight[:118]
        node_emb = torch.cat([atom_emb, entity_emb], dim=0)
        bond_emb = self.edge_emb.weight[:4]  
        bond_emb = self.bond_proj(bond_emb) 
        edge_emb = torch.cat([bond_emb, relation_emb], dim=0)
        
        self.node_emb = nn.Embedding.from_pretrained(node_emb, freeze=False)
        self.edge_emb = nn.Embedding.from_pretrained(edge_emb, freeze=False)

    def KMPNN(self, g, entity_emb, relation_emb):
        try:
            node_feats = self.node_emb(g.ndata['h'])            # (N,128)
            edge_feats = self.edge_emb(g.edata['e'])            # (E,64)
            node_feats = self.project_node_feats(node_feats)    # �?N,64)
            hidden_feats = node_feats.unsqueeze(0)              # (1,N,64)

            for _ in range(6):
                node_feats = F.relu(self.gnn_layer(g, node_feats, edge_feats))
                node_feats, hidden_feats = self.gru(node_feats.unsqueeze(0), hidden_feats)
                node_feats = node_feats.squeeze(0)
            return node_feats                                    # (N,64)
        except:
            return None

    def Set_readout(self, graph, feat):
        try:
            with graph.local_scope():
                batch_size = graph.batch_size
                h = (feat.new_zeros((3, batch_size, 64)),
                     feat.new_zeros((3, batch_size, 64)))
                q_star = feat.new_zeros(batch_size, 128)

                for _ in range(6):
                    q, h = self.lstm(q_star.unsqueeze(0), h)    # (1,B,128)
                    q = q.view(batch_size, 64)
                    e = (feat * broadcast_nodes(graph, q)).sum(dim=-1, keepdim=True)
                    graph.ndata['e'] = e
                    alpha = softmax_nodes(graph, 'e')
                    graph.ndata['r'] = feat * alpha
                    readout = sum_nodes(graph, 'r')
                    q_star = th.cat([q, readout], dim=-1)
                return q_star                                    # (B,128)
        except:
            return None
        
    def smile2fp(self, mol, nBits=4096):
        arr = np.zeros((nBits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 3, nBits), arr)
        return arr

    def forward(self, mol, adj_1, nd_1, ed_1, de_1, mask_1,
                bg, entity_emb, relation_emb):
        device = adj_1.device
        edge_batch_batch_indices_1, edge_batch_node_indices_1, edge_batch_neighbour_indices_1 = \
            adj_1.nonzero().unbind(-1)
        node_batch_batch_indices_1, node_batch_node_indices_1 = \
            adj_1.sum(-1).nonzero().unbind(-1)
        node_batch_adj_1 = adj_1[node_batch_batch_indices_1, node_batch_node_indices_1, :]
        node_degrees_1 = node_batch_adj_1.sum(-1).long()
        max_node_degree_1 = node_degrees_1.max()

        node_batch_node_neighbours_1  = torch.zeros(
            node_degrees_1.shape[0], max_node_degree_1, self.node_fts_1, device=device)
        node_batch_edges_1            = torch.zeros(
            node_degrees_1.shape[0], max_node_degree_1, self.edge_fts_1, device=device)
        
        node_batch_node_neighbour_mask_1 = torch.zeros(
            node_degrees_1.shape[0], max_node_degree_1, device=device)

        node_batch_neighbour_neighbour_indices_1 = torch.cat(
            [torch.arange(i) for i in node_degrees_1])
        edge_batch_node_batch_indices_1 = torch.cat(
            [i * torch.ones(degree) for i, degree in enumerate(node_degrees_1)]
        ).long()


        B, N, _ = adj_1.shape[:3]
        num_active_nodes = node_batch_batch_indices_1.shape[0]
        active_lookup = -1 * torch.ones(B, N, dtype=torch.long, device=device)
        active_lookup[node_batch_batch_indices_1, node_batch_node_indices_1] = torch.arange(num_active_nodes, device=device)

        edge_active_ids = active_lookup[edge_batch_batch_indices_1, edge_batch_node_indices_1]  # (E,)

        if torch.any(edge_active_ids < 0):
            raise RuntimeError("Found an edge whose source node is not active.")

        sort_key = edge_active_ids * (N + 1) + edge_batch_neighbour_indices_1
        sorted_idx = torch.argsort(sort_key)
        edge_active_ids_sorted = edge_active_ids[sorted_idx]

        _, counts = torch.unique_consecutive(edge_active_ids_sorted, return_counts=True)
        slot_indices_sorted = torch.cat([torch.arange(c, device=device) for c in counts])

        slot_indices = torch.empty_like(slot_indices_sorted)
        slot_indices[sorted_idx] = slot_indices_sorted

        node_batch_edges_1 = torch.zeros(
            num_active_nodes, max_node_degree_1, self.edge_fts_1, device=device
        )
        node_batch_node_neighbour_mask_1 = torch.zeros(
            num_active_nodes, max_node_degree_1, device=device
        )

        edge_features = ed_1[edge_batch_batch_indices_1, edge_batch_node_indices_1, edge_batch_neighbour_indices_1, :]  # (E, edge_fts)

        node_batch_edges_1[edge_active_ids, slot_indices, :] = edge_features
        node_batch_node_neighbour_mask_1[edge_active_ids, slot_indices] = 1
        hidden_nodes_1 = nd_1.clone()

        for _ in range(self.message_passes):
            feature_dim = hidden_nodes_1.shape[-1]
            if node_batch_node_neighbours_1.shape[-1] != feature_dim:
                shape = list(node_batch_node_neighbours_1.shape)
                shape[-1] = feature_dim
                node_batch_node_neighbours_1 = torch.zeros(*shape, device=hidden_nodes_1.device)
            node_batch_nodes_1 = hidden_nodes_1[
                node_batch_batch_indices_1, node_batch_node_indices_1, :]
            node_batch_node_neighbours_1[edge_active_ids, slot_indices, :] = hidden_nodes_1[edge_batch_batch_indices_1, edge_batch_neighbour_indices_1, :]

            messages_1 = self.aggregate_message_1(
                node_batch_nodes_1,
                node_batch_node_neighbours_1.clone(),
                node_batch_edges_1,
                node_batch_node_neighbour_mask_1)

            hidden_nodes_1[node_batch_batch_indices_1, node_batch_node_indices_1, :] = self.update_1(
                node_batch_nodes_1, messages_1)

        node_mask_1 = (adj_1.sum(-1) != 0)
        output_1 = self.readout_1(hidden_nodes_1, nd_1, node_mask_1)   # (B,128)

        kg_batch = self.KMPNN(bg, entity_emb, relation_emb)            # (N_tot,64)
        kg_out   = self.Set_readout(bg, kg_batch)                      # (B,128)

        batch_size = nd_1.size(0)
        ex_d_mask = de_1.unsqueeze(1).unsqueeze(2)
        ex_d_mask = (1.0 - ex_d_mask) * -10000.0                       

        d_emb = self.emb(de_1)                                         # (B,50,384)
        d_encoded_layers = self.d_encoder(d_emb.float(), ex_d_mask.float())
        d1_trans_fts = d_encoded_layers.view(batch_size, -1)           # (B,50*384)
        if batch_size == 1:
            d1_trans_fts_layer1 = self.decoder_1_ln(d1_trans_fts)             # (B,128)
        else:
            d1_trans_fts_layer1 = self.decoder_1(d1_trans_fts)             # (B,128)
        morgan_fp = torch.tensor([self.smile2fp(m) for m in mol], device=device)  # batch处理
        morgan_emb = self.morgan_proj(morgan_fp)

        final_emb = self.cross_att(output_1, kg_out, d1_trans_fts_layer1, morgan_emb)

        return final_emb                                              # 返回融合后的特征


def compute_mol_features(smiles, max_seq_len=50, kg_pickle_path=settings.kg_triples_emb_path):
    if type(smiles) is list:
        if type(smiles[0]) is str:
            mol = Chem.MolFromSmiles(smiles[0]) 
        else:
            mol = smiles[0]
    else:
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}") 

    adj_1 = torch.tensor(AllChem.GetAdjacencyMatrix(mol, useBO=True), dtype=torch.float32).unsqueeze(0) 

    N = mol.GetNumAtoms()
    nd_1 = torch.zeros(1, N, 10) 
    for i, atom in enumerate(mol.GetAtoms()):
        nd_1[0, i, 0] = atom.GetAtomicNum()             
        nd_1[0, i, 1] = atom.GetTotalValence()          
        nd_1[0, i, 2] = atom.GetDegree()                
        nd_1[0, i, 3] = atom.GetHybridization()         
        nd_1[0, i, 4] = atom.GetFormalCharge()          
        nd_1[0, i, 5] = atom.GetIsAromatic()            
        nd_1[0, i, 6] = atom.GetNumRadicalElectrons()   
        nd_1[0, i, 7] = atom.GetImplicitValence()       
        nd_1[0, i, 8] = atom.GetTotalNumHs()            
        nd_1[0, i, 9] = atom.GetChiralTag()             

    ed_1 = torch.zeros(1, N, N, 5)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = bond.GetBondTypeAsDouble()  
        ed_1[0, i, j, 0] = bond_type
        ed_1[0, j, i, 0] = bond_type  
        ed_1[0, i, j, 1] = bond.GetIsAromatic()     
        ed_1[0, j, i, 1] = bond.GetIsAromatic()
        ed_1[0, i, j, 2] = bond.GetIsConjugated()   
        ed_1[0, j, i, 2] = bond.GetIsConjugated()
        ed_1[0, i, j, 3] = bond.GetStereo()         
        ed_1[0, j, i, 3] = bond.GetStereo()
        ed_1[0, i, j, 4] = 1 if bond.IsInRing() else 0  
        ed_1[0, j, i, 4] = ed_1[0, i, j, 4]

    if type(smiles[-1]) is not str:
        smiles = [Chem.MolToSmiles(smile) for smile in smiles]
    tokens = [ord(c) % 30000 for c in smiles] 
    seq_len = min(len(tokens), max_seq_len)
    de_1 = torch.zeros(1, max_seq_len, dtype=torch.long)
    de_1[0, :seq_len] = torch.tensor(tokens[:seq_len])

    mask_1 = torch.zeros(1, max_seq_len)
    mask_1[0, :seq_len] = 1.0

    src, dst = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        src.extend([i, j])
        dst.extend([j, i])  
    bg = dgl.graph((torch.tensor(src), torch.tensor(dst)), num_nodes=N)
    bg.ndata['feat'] = nd_1[0]  
    bg.edata['feat'] = torch.tensor([bond.GetBondTypeAsDouble() for bond in mol.GetBonds()] * 2)  

    with open(kg_pickle_path, 'rb') as f:
        loaded_dict = pickle.load(f)
    entity_emb = loaded_dict['entity_emb']  # (num_entities, 128)
    relation_emb = loaded_dict['relation_emb']  # (num_relations, 64)

    return mol, adj_1, nd_1, ed_1, de_1, mask_1, bg, entity_emb, relation_emb
def _to_single_smiles(x):
    if isinstance(x, Chem.Mol):
        return Chem.MolToSmiles(x)
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            raise ValueError("Empty smiles list encountered.")
        first = x[0]
        return _to_single_smiles(first)
    return str(x)

class MolMultiModalDataset(Dataset):
    def __init__(self, smiles_list, labels):
        self.smiles = smiles_list
        self.labels = labels

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        # print(f"smiles of batch:{smiles}")
        smiles = _to_single_smiles(smiles)  
        assert isinstance(smiles, str) and len(smiles) > 0, f"Bad smiles at idx={idx}: {type(smiles)} {smiles!r}"
        mol, adj_1, nd_1, ed_1, de_1, mask_1, bg, entity_emb, relation_emb = compute_mol_features(smiles)
        adj_1 = torch.as_tensor(adj_1) if not isinstance(adj_1, torch.Tensor) else adj_1
        nd_1 = torch.as_tensor(nd_1) if not isinstance(nd_1, torch.Tensor) else nd_1
        ed_1 = torch.as_tensor(ed_1) if not isinstance(ed_1, torch.Tensor) else ed_1
        de_1 = torch.as_tensor(de_1) if not isinstance(de_1, torch.Tensor) else de_1
        mask_1 = torch.as_tensor(mask_1) if not isinstance(mask_1, torch.Tensor) else mask_1
        entity_emb = torch.as_tensor(entity_emb) if not isinstance(entity_emb, torch.Tensor) else entity_emb
        relation_emb = torch.as_tensor(relation_emb) if not isinstance(relation_emb, torch.Tensor) else relation_emb


        return {
            "mol": mol,
            "adj_1": adj_1,
            "nd_1": nd_1,
            "ed_1": ed_1,
            "de_1": de_1,
            "mask_1": mask_1,
            "bg": bg,
            "entity_emb": entity_emb,
            "relation_emb": relation_emb,
            "label": torch.tensor(self.labels[idx], dtype=torch.float32)
        }

class gnn(nn.Module):

    def __init__(self,
                 in_feats,                        
                 out_feats,                       
                 attn_fc,                         
                 edge_func1, edge_func2,          
                 aggregator_type='mean',          # 'sum' | 'mean' | 'max'
                 residual=False, bias=True):
        super(gnn, self).__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats

        # ------ Attention、Edge MLP -------- #
        self.attn_fc   = attn_fc
        self.edge_func1 = edge_func1
        self.edge_func2 = edge_func2

        if aggregator_type == 'sum':
            self.reducer = fn.sum
        elif aggregator_type == 'mean':
            self.reducer = fn.mean
        elif aggregator_type == 'max':
            self.reducer = fn.max
        else:
            raise KeyError(f'Aggregator type {aggregator_type} not recognized')
        self._aggre_type = aggregator_type

        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(self._in_dst_feats, out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)

        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_buffer('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        gain = init.calculate_gain('relu')
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def edge_attention(self, edges):
        z2 = th.cat([edges.src['z'], edges.dst['z']], dim=1)   
        a  = self.attn_fc(z2)                                  # (E,1)
        return {'attn_e': F.leaky_relu(a)}

    def message_func1(self, edges):
        return {'m1': edges.src['h'] * edges.data['w1'],       
                'attn_e1': edges.data['attn_e'],
                'z1': edges.src['z']}

    def message_func2(self, edges):
        return {'m2': edges.src['h'] * edges.data['w2'],
                'attn_e2': edges.data['attn_e'],
                'z2': edges.src['z']}

    def reduce_func1(self, nodes):
        alpha = F.softmax(nodes.mailbox['attn_e1'], dim=1).unsqueeze(-1)
        h = th.sum(alpha * nodes.mailbox['m1'], dim=1)
        return {'neigh1': h}

    def reduce_func2(self, nodes):
        alpha = F.softmax(nodes.mailbox['attn_e2'], dim=1).unsqueeze(-1)
        h = th.sum(alpha * nodes.mailbox['m2'], dim=1)
        return {'neigh2': h}

    def forward(self, graph, feat, efeat):
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, graph)
            graph.srcdata['h'] = feat_src.unsqueeze(-1)                      # (N,d,1)

            graph.edata['w1'] = self.edge_func1(efeat).view(-1, self._in_src_feats, self._out_feats)
            graph.edata['w2'] = self.edge_func2(efeat).view(-1, self._in_src_feats, self._out_feats)

            graph.ndata['z'] = feat_src
            graph.apply_edges(self.edge_attention)

            edges1 = th.nonzero(graph.edata['etype'] == 0).squeeze(1).int()
            edges2 = th.nonzero(graph.edata['etype'] == 1).squeeze(1).int()

            graph.send_and_recv(edges1, self.message_func1, self.reduce_func1)
            graph.send_and_recv(edges2, self.message_func2, self.reduce_func2)

            rst1 = graph.dstdata['neigh1'].sum(dim=1)
            rst2 = graph.dstdata['neigh2'].sum(dim=1)
            rst  = rst1 + rst2                                      # (N,d_out)

            if self.res_fc is not None:
                rst = rst + self.res_fc(feat_dst)
            # bias
            if self.bias is not None:
                rst = rst + self.bias
            return rst


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta  = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class Embeddings(nn.Module):

    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate):
        super(Embeddings, self).__init__()
        self.word_embeddings     = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout   = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        input_ids = input_ids.type(torch.long)
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long,
                                    device=input_ids.device).unsqueeze(0).expand_as(input_ids)

        words_embeddings     = self.word_embeddings(input_ids)
        position_embeddings  = self.position_embeddings(position_ids)
        embeddings = self.LayerNorm(words_embeddings + position_embeddings)
        return self.dropout(embeddings)

class SelfAttention(nn.Module):

    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        self.num_attention_heads  = num_attention_heads
        self.attention_head_size  = hidden_size // num_attention_heads
        self.all_head_size        = hidden_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key   = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        # B,L,H*D �?B,H,L,D
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(*new_shape).permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        Q = self.transpose_for_scores(self.query(hidden_states))
        K = self.transpose_for_scores(self.key(hidden_states))
        V = self.transpose_for_scores(self.value(hidden_states))

        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        scores = scores + attention_mask
        probs  = self.dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(probs, V)                            # (B,H,L,D)
        context = context.permute(0, 2, 1, 3).contiguous()          # (B,L,H,D)
        new_shape = context.size()[:-2] + (self.all_head_size,)
        return context.view(*new_shape)                             # (B,L,hidden)


class AttentionBlock(nn.Module):

    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()
        self.hid_dim, self.n_heads = hid_dim, n_heads
        assert hid_dim % n_heads == 0

        self.f_q = nn.Linear(hid_dim, hid_dim)
        self.f_k = nn.Linear(hid_dim, hid_dim)
        self.f_v = nn.Linear(hid_dim, hid_dim)
        self.fc  = nn.Linear(hid_dim, hid_dim)
        self.do  = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).cuda()

    def forward(self, query, key, value, mask=None):
        B = query.shape[0]
        Q = self.f_q(query).view(B, self.n_heads, -1).unsqueeze(3)
        K = self.f_k(key  ).view(B, self.n_heads, -1).unsqueeze(3).transpose(2,3)
        V = self.f_v(value).view(B, self.n_heads, -1).unsqueeze(3)

        energy = torch.matmul(Q, K) / self.scale.cuda()
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attn = self.do(torch.softmax(energy, dim=-1))

        out = torch.matmul(attn, V).permute(0,2,1,3).contiguous().view(B, self.hid_dim)
        return self.do(self.fc(out))


class CrossAttentionBlock(nn.Module):

    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()
        self.proj_seq = nn.Linear(100, 128)        
        self.att_seq2graph = AttentionBlock(hid_dim, n_heads, dropout)  
        self.att_fp2graph = AttentionBlock(hid_dim, n_heads, dropout)   
        self.att_self = AttentionBlock(hid_dim, n_heads, dropout)       

    def forward(self, graph_feature, kg, sequence_feature, morgan_feature):
        if kg is not None:
            g_out = graph_feature + kg
        else:
            g_out = graph_feature
        g_out = self.proj_seq(g_out)     # [16, 100]
        g_out = g_out + self.att_seq2graph(sequence_feature, g_out, g_out)
        g_out = g_out + self.att_fp2graph(morgan_feature, g_out, g_out)
        output = self.att_self(g_out, g_out, g_out)
        return output


class SelfOutput(nn.Module):  
    def __init__(self, hidden_size, hidden_dropout_prob):  
        super().__init__()  
        self.dense = nn.Linear(hidden_size, hidden_size)  
        self.LayerNorm = LayerNorm(hidden_size)  
        self.dropout = nn.Dropout(hidden_dropout_prob)  

    def forward(self, hidden_states, input_tensor):  
        hidden_states = self.dense(hidden_states)  
        hidden_states = self.dropout(hidden_states)  
        return self.LayerNorm(hidden_states + input_tensor)  


class Attention(nn.Module):  
    def __init__(self, hidden_size, num_heads, attn_p, hid_p):  
        super().__init__()  
        self.self  = SelfAttention(hidden_size, num_heads, attn_p)  
        self.output = SelfOutput(hidden_size, hid_p)  

    def forward(self, x, mask):  
        return self.output(self.self(x, mask), x)  
  
class Intermediate(nn.Module):  
    def __init__(self, hidden_size, intermediate_size):  
        super().__init__()  
        self.dense = nn.Linear(hidden_size, intermediate_size)  

    def forward(self, x):  
        return F.relu(self.dense(x))  


class Output(nn.Module):  
    def __init__(self, inter_size, hidden_size, dropout_p):  
        super().__init__()  
        self.dense = nn.Linear(inter_size, hidden_size)  
        self.LayerNorm = LayerNorm(hidden_size)  
        self.dropout = nn.Dropout(dropout_p)  

    def forward(self, x, residual):  
        return self.LayerNorm(self.dropout(self.dense(x)) + residual)  

class Encoder(nn.Module):  
    def __init__(self, hidden_size, inter_size, n_heads, attn_p, hid_p):  
        super().__init__()  
        self.attention   = Attention(hidden_size, n_heads, attn_p, hid_p)  
        self.intermediate = Intermediate(hidden_size, inter_size)  
        self.output       = Output(inter_size, hidden_size, hid_p)  

    def forward(self, x, mask):  
        att_out = self.attention(x, mask)  
        inter   = self.intermediate(att_out)  
        return self.output(inter, att_out)  
 
class Encoder_MultipleLayers(nn.Module):  
    def __init__(self, n_layer, hidden_size, inter_size, n_heads, attn_p, hid_p):  
        super().__init__()  
        layer = Encoder(hidden_size, inter_size, n_heads, attn_p, hid_p)  
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layer)])  

    def forward(self, x, mask):  
        for l in self.layer:  
            x = l(x, mask)  
        return x  
    
