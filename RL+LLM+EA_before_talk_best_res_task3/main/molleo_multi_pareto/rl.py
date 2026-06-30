import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import re
import numpy as np

def select_fragment(frags, frag_importance, k=5, exploration_temp=None):
    uniq = list(set(frags))                                               
    if not uniq:                                                          
        return []                                                         

    scores = np.array([frag_importance.get(f, 0.0) for f in uniq], dtype=float) 

    if exploration_temp is not None and exploration_temp > 0:             
        logits = scores / exploration_temp                                
        logits -= logits.max()                                            
        probs = np.exp(logits)                                            
        probs /= probs.sum()                                              
        idx = np.random.choice(len(uniq), size=min(k, len(uniq)),        
                               replace=False, p=probs)
        chosen = [uniq[i] for i in idx]                                   
        chosen = sorted(chosen, key=lambda f: frag_importance.get(f, 0.0), reverse=True) 
        return chosen                                                     

    ordered = sorted(uniq, key=lambda f: frag_importance.get(f, 0.0), reverse=True)  
    print(f"{ordered[:k]}")                                               
    return ordered[:k]                                                     

import math
class LoRALinear(nn.Module):

    def __init__(self, base_linear: nn.Linear, r: int = 4, alpha: int = 8, dropout: float = 0.0):
        super().__init__()
        self.in_features  = base_linear.in_features   
        self.out_features = base_linear.out_features  
        self.r = r                                     
        self.alpha = alpha                            
        self.scaling = alpha / float(r)                
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        self.weight = nn.Parameter(base_linear.weight.detach().clone(), requires_grad=False)  
        if base_linear.bias is not None:
            self.bias = nn.Parameter(base_linear.bias.detach().clone(), requires_grad=False) 
        else:
            self.bias = None

        device = self.weight.device  

        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features, device=device))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, device=device))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  
        nn.init.zeros_(self.lora_B)                            

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(self.dropout(x), self.weight, self.bias)
        lora_update = F.linear(self.dropout(x), self.lora_A)        # (B, r) = x @ A^T
        lora_update = F.linear(lora_update, self.lora_B)           # (B, out) = (x A^T) @ B^T
        return base + self.scaling * lora_update                   # y = x W0^T + s * x (A^T B^T) + b

def _replace_linear_with_lora(module: nn.Module, r=4, alpha=8, dropout=0.05):
    wrapped = []
    for name, child in list(module.named_children()):  
        if isinstance(child, nn.Linear):               
            lora = LoRALinear(child, r=r, alpha=alpha, dropout=dropout)
            setattr(module, name, lora) 
            wrapped.append((module, name, lora))
        else:
            wrapped.extend(_replace_linear_with_lora(child, r=r, alpha=alpha, dropout=dropout))
    return wrapped

def collect_lora_trainable_params(wrapped):
    params = []
    for _, _, lora in wrapped:
        params += [lora.lora_A, lora.lora_B]
    return params

def merge_lora_and_unwrap(wrapped):
    for parent, name, lora in wrapped:
        device = lora.weight.device
        dtype  = lora.weight.dtype
        with torch.no_grad():
            merged = nn.Linear(lora.in_features, lora.out_features, bias=(lora.bias is not None))
            merged.to(device=device, dtype=dtype) 

            lora_A = lora.lora_A.to(device=device)
            lora_B = lora.lora_B.to(device=device)

            scaling = lora.scaling.to(device) if isinstance(lora.scaling, torch.Tensor) else lora.scaling

            merged.weight.copy_(lora.weight + (lora_B @ lora_A) * scaling)

            if lora.bias is not None:
                merged.bias.copy_(lora.bias)

        setattr(parent, name, merged)

    
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rdkit import Chem, DataStructs
from clustering import compute_center_mol, get_fp

def nearest_cluster_id(current_mol, clustering_data, fp_type="morgan"):
    centers_fps = []
    for cluster in clustering_data:
        if cluster:
            _, center_fp = compute_center_mol(cluster, fp_type=fp_type)
            centers_fps.append(center_fp)
    if not centers_fps:
        return 0
    fp_current = get_fp(current_mol, fp_type)
    sims = [DataStructs.TanimotoSimilarity(fp_current, fp_c) for fp_c in centers_fps]
    return int(torch.tensor(sims).argmax().item())

import random 
import torch

class ReplayBuffer:
    def __init__(self, capacity=200_000, pad_value=0.0):

        self.buf = []                 
        self.capacity = capacity      
        self.idx = 0                  
        self.pad_value = float(pad_value)  

    @staticmethod
    def _to_reward_tensor(reward):
        if isinstance(reward, torch.Tensor):                
            r = reward.detach().cpu().float().reshape(-1)   
        elif isinstance(reward, (list, tuple)):             
            r = torch.tensor(reward, dtype=torch.float32).reshape(-1) 
        else:
            r = torch.tensor([float(reward)], dtype=torch.float32) 
        return r 

    def push(self, state, action, reward, next_state, done):
        item = (
            state.clone().float().cpu(),                        
            int(action),                                         
            self._to_reward_tensor(reward),                      
            None if next_state is None else next_state.clone().float().cpu(),  
            bool(done)                                          
        )
        if len(self.buf) < self.capacity:  
            self.buf.append(item)          
        else:                              
            self.buf[self.idx] = item      
        self.idx = (self.idx + 1) % self.capacity  

    def __len__(self):
        return len(self.buf)  

    def sample(self, batch_size, device):
        batch = random.sample(self.buf, batch_size)  
        s, a, r_list, s2, d = zip(*batch)           

        s = torch.stack([item.float().cpu() for item in s]).to(device)                
        a = torch.tensor(a, dtype=torch.long, device=device).unsqueeze(1)             
        d = torch.tensor(d, dtype=torch.float32, device=device).unsqueeze(1)          
        if any(item is None for item in s2):
            zero_state = torch.zeros_like(s[0].detach().cpu())
            s2 = torch.stack([zero_state if item is None else item.float().cpu() for item in s2]).to(device)
        else:
            s2 = torch.stack([item.float().cpu() for item in s2]).to(device)

        lens = [ri.numel() for ri in r_list]                                          
        max_len = max(lens) if len(lens) > 0 else 1                                   

        r = torch.full((batch_size, max_len), fill_value=self.pad_value,
                       dtype=torch.float32, device=device)                            
        r_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)  

        for i, ri in enumerate(r_list):                                               
            n = ri.numel()                                                            
            if n > 0:                                                                 
                r[i, :n] = ri.to(device).reshape(-1)                                  
                r_mask[i, :n] = True                                                  

        return s, a, r, s2, d, r_mask                                                 

class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim, n_actions=3, hidden=512, dropout=0.1):
        super().__init__()  
        self.fc1 = nn.Linear(state_dim, hidden)  
        self.ln1 = nn.LayerNorm(hidden)  
        self.fc2 = nn.Linear(hidden, hidden)  
        self.ln2 = nn.LayerNorm(hidden)  
        self.dropout = nn.Dropout(dropout) 
        self.V = nn.Linear(hidden, 1)  
        self.A = nn.Linear(hidden, n_actions)  
        self._reset()  

    def _reset(self):
        for m in [self.fc1, self.fc2]:  
            nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu')) 
            nn.init.zeros_(m.bias) 
        nn.init.orthogonal_(self.V.weight, gain=0.01); nn.init.zeros_(self.V.bias) 
        nn.init.orthogonal_(self.A.weight, gain=0.01); nn.init.zeros_(self.A.bias)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)  
        x = F.relu(self.ln1(self.fc1(x)))  
        x = self.dropout(x) 
        x = F.relu(self.ln2(self.fc2(x)))  
        x = self.dropout(x)  
        V = self.V(x) 
        A = self.A(x) 
        return V + A - A.mean(dim=1, keepdim=True)  


class DQNTrainer:
    def __init__(self, state_dim, n_actions, device, gamma=0.98, lr=1e-3, wd=1e-6, tau=0.01):
        self.device = device  
        self.gamma = gamma 
        self.tau = tau  
        self.q = DuelingQNetwork(state_dim, n_actions=n_actions).to(device)  
        self.tar = DuelingQNetwork(state_dim, n_actions=n_actions).to(device)  
        self.tar.load_state_dict(self.q.state_dict())  
        self.optim = optim.AdamW(self.q.parameters(), lr=lr, weight_decay=wd)  
        self.sched = optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=200_000, eta_min=1e-5)  

    @torch.no_grad()
    def _soft_update(self):
        for p, tp in zip(self.q.parameters(), self.tar.parameters()):  
            tp.data.mul_(1 - self.tau).add_(self.tau * p.data)  

    def train_step(self, replay: ReplayBuffer, batch_size):
        if len(replay) < batch_size:  
            return None  
        s, a, r, s2, d, r_mask = replay.sample(batch_size, self.device)  
        q_sa = self.q(s).gather(1, a) 

        r_scalar = []
        for i in range(r.shape[0]):
            valid = r[i][r_mask[i]]
            if valid.numel() == 0:
                r_scalar.append(torch.tensor(0.0, device=self.device))
                continue
            k = min(5, valid.numel())
            topk_vals, _ = torch.topk(valid, k=k, largest=True, sorted=False)
            r_scalar.append(topk_vals.mean())
        r = torch.stack(r_scalar).unsqueeze(1)

        with torch.no_grad(): 
            a2 = self.q(s2).argmax(dim=1, keepdim=True)
            q2 = self.tar(s2).gather(1, a2) 
            target = r + (1.0 - d) * self.gamma * q2 
        loss = F.smooth_l1_loss(q_sa, target)
        self.optim.zero_grad() 
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=5.0)
        self.optim.step() 
        self.sched.step()
        self._soft_update()
        return float(loss.item())

class ActionSelector:
    def __init__(self, n_actions=3, eps_start=1.0, eps_end=0.05, eps_decay_steps=50_000,
                 use_ucb=True, ucb_c=0.5, use_softmax=False, tau=1.0, min_trials_per_cluster=5,
                 eps_floor=0.05, force_random_order=True, gumbel_noise=1e-6, softmax_eps=0.02):
        self.n_actions = n_actions
        self.eps_start, self.eps_end = eps_start, eps_end
        self.eps_decay_steps = eps_decay_steps
        self.use_ucb, self.ucb_c = use_ucb, ucb_c
        self.use_softmax, self.tau = use_softmax, tau
        self.min_trials_per_cluster = min_trials_per_cluster

        self.eps_floor = eps_floor              
        self.force_random_order = force_random_order  
        self.gumbel_noise = gumbel_noise        
        self.softmax_eps = softmax_eps          

        self.global_step = 0
        self.counts = {}
        self.totals = {}

    def _epsilon(self):
        t = min(self.global_step, self.eps_decay_steps)
        frac = 1.0 - t / self.eps_decay_steps
        eps = self.eps_end + (self.eps_start - self.eps_end) * frac
        return max(eps, self.eps_floor)

    def _bump(self, cid, a):
        self.counts[(cid, a)] = self.counts.get((cid, a), 0) + 1
        self.totals[cid] = self.totals.get(cid, 0) + 1

    def pick(self, q_values, cluster_id):
        self.global_step += 1
        eps = self._epsilon()
        cid = int(cluster_id)
        _ = self.totals.get(cid, 0)

        actions = list(range(self.n_actions))   
        if self.force_random_order:
            random.shuffle(actions)             
        for a in actions:
            if self.counts.get((cid, a), 0) < self.min_trials_per_cluster:
                self._bump(cid, a)             
                return a                       

        if random.random() < eps:
            a = random.randrange(self.n_actions)  
            self._bump(cid, a)
            return a

        q = q_values.squeeze(0).detach().cpu()    
        if self.use_ucb:
            N = max(1, self.totals.get(cid, 0))   
            n_vec = torch.tensor([max(1, self.counts.get((cid, a), 0)) for a in range(self.n_actions)],
                                  dtype=q.dtype)  
            bonus = self.ucb_c * torch.sqrt(torch.log(torch.tensor(float(N) + 1.0, dtype=q.dtype)) / n_vec)
            scores = q + bonus                    

            if self.gumbel_noise > 0:
                g = -torch.log(-torch.log(torch.rand_like(scores)))
                scores = scores + self.gumbel_noise * g

            a = int(torch.argmax(scores).item())  

        elif self.use_softmax:
            probs = torch.softmax(q / max(1e-6, self.tau), dim=0)
            if self.softmax_eps > 0:
                probs = (1 - self.softmax_eps) * probs + self.softmax_eps / self.n_actions
            a = int(torch.multinomial(probs, 1).item())  

        else:
            scores = q
            if self.gumbel_noise > 0:
                g = -torch.log(-torch.log(torch.rand_like(scores)))
                scores = scores + self.gumbel_noise * g
            a = int(torch.argmax(scores).item())

        self._bump(cid, a)
        return a


import math 
from typing import Iterable, Optional, Union, Callable, Tuple, Dict, Any 
import torch 

Number = Union[float, int]
ArrayLike = Union[Iterable[Number], torch.Tensor]

class EpisodeManager:
    def __init__(
        self,
        max_steps: int = 50, 
        patience: int = 10,  
        min_delta: float = 0.0,  
        target_reward: Optional[float] = None,  
        aggregate: Union[str, Callable[[torch.Tensor], torch.Tensor]] = "topk_mean",  
        topk: int = 5,  
        quantile: float = 0.9, 
        target_mode: str = "any",  
        min_delta_mode: str = "absolute",  
        ema_alpha: Optional[float] = None,  
        ignore_nan: bool = True,  
    ):
        self.max_steps = int(max_steps)  
        self.patience = int(patience)  
        self.min_delta = float(min_delta)  
        self.target_reward = None if target_reward is None else float(target_reward)  

        self.aggregate = aggregate  
        self.topk = int(topk)  
        self.quantile = float(quantile) 
        self.target_mode = str(target_mode) 
        assert self.target_mode in ("any", "agg")  
        self.min_delta_mode = str(min_delta_mode)  
        assert self.min_delta_mode in ("absolute", "relative")  
        self.ema_alpha = None if ema_alpha is None else float(ema_alpha)  
        if self.ema_alpha is not None:  
            assert 0.0 < self.ema_alpha <= 1.0
        self.ignore_nan = bool(ignore_nan) 

        self.step_idx = 0 
        self.best_score = float("-inf")  
        self.no_improve_steps = 0  
        self.ema_value = None  
        self._last_info: Dict[str, Any] = {}  

    def reset(self):
        self.step_idx = 0 
        self.best_score = float("-inf") 
        self.no_improve_steps = 0  
        self.ema_value = None  
        self._last_info = {} 

    @staticmethod
    def _to_1d_float_tensor(x: ArrayLike) -> torch.Tensor:
        if isinstance(x, torch.Tensor):  
            t = x.detach().flatten().to(torch.float32).cpu() 
        else:
            t = torch.tensor(list(x), dtype=torch.float32).flatten() 
        return t  

    def _aggregate(self, rewards: torch.Tensor) -> Tuple[float, Dict[str, float]]:
        t = rewards  
        if self.ignore_nan:  
            t = t[~torch.isnan(t)]  
        n = t.numel()  

        stats = {
            "n": float(n),
            "max": float(t.max().item()) if n > 0 else float("-inf"),
            "mean": float(torch.nanmean(t).item()) if n > 0 else float("nan"),
            "median": float(torch.nanmedian(t).item()) if n > 0 else float("nan"),
        }

        if n > 0:
            k = max(1, min(self.topk, n))
            topk_vals, _ = torch.topk(t, k=k, largest=True, sorted=False)
            stats["topk_mean"] = float(torch.nanmean(topk_vals).item())
            q = min(max(self.quantile, 0.0), 1.0) 
            stats["quantile"] = float(torch.nanquantile(t, q=q).item())
        else:
            stats["topk_mean"] = float("nan") 
            stats["quantile"] = float("nan") 

        agg_name = None 
        if callable(self.aggregate):
            agg_value = float(self.aggregate(t).item()) if n > 0 else float("-inf")
            agg_name = "custom"
        else:
            agg = self.aggregate.lower()  
            agg_name = agg 
            if agg == "max":
                agg_value = stats["max"]  
            elif agg == "mean":
                agg_value = stats["mean"]  
            elif agg == "median":
                agg_value = stats["median"]  
            elif agg == "topk_mean":
                agg_value = stats["topk_mean"] 
            elif agg == "quantile":
                agg_value = stats["quantile"]  
            else:
                raise ValueError(f"Unknown aggregate: {self.aggregate}")  

        stats["agg"] = float(agg_value)  
        stats["agg_name"] = agg_name  
        return stats["agg"], stats  

    def _apply_ema(self, value: float) -> float:
        if self.ema_alpha is None: 
            return value 
        if self.ema_value is None or not math.isfinite(self.ema_value): 
            self.ema_value = value  
        else:
            self.ema_value = self.ema_alpha * value + (1.0 - self.ema_alpha) * self.ema_value  
        return self.ema_value  

    def _improved(self, score: float) -> bool:
        if not math.isfinite(self.best_score):  
            return True  
        if self.min_delta_mode == "absolute":  
            return score > self.best_score + self.min_delta  
        else:  
            return score > self.best_score * (1.0 + self.min_delta) 

    def step(self, rewards: ArrayLike, return_info: bool = False) -> Union[bool, Tuple[bool, Dict[str, Any]]]:
        r = self._to_1d_float_tensor(rewards) 
        self.step_idx += 1  

        agg_raw, stats = self._aggregate(r) 
        agg_for_patience = self._apply_ema(agg_raw)
        stats["agg_after_ema"] = float(agg_for_patience) if self.ema_alpha is not None else stats["agg"]

        target_hit = False  
        if self.target_reward is not None and r.numel() > 0: 
            if self.target_mode == "any": 
                target_hit = bool(torch.nanmax(r).item() >= self.target_reward)  
            else:  
                target_hit = bool(agg_for_patience >= self.target_reward)  
            stats["frac_ge_target"] = float(torch.mean((r >= self.target_reward).float()).item()) if self.target_mode == "any" else float("nan") 
        else:
            stats["frac_ge_target"] = float("nan")

        if self._improved(agg_for_patience):  
            self.best_score = float(agg_for_patience)  
            self.no_improve_steps = 0 
            improved = True  
        else:
            self.no_improve_steps += 1 
            improved = False  

        hit_max_steps = self.step_idx >= self.max_steps  
        hit_patience = self.no_improve_steps >= self.patience 

        done = target_hit or hit_patience or hit_max_steps 

        info = {
            "step_idx": self.step_idx,  
            "done": done,  
            "improved": improved,  
            "no_improve_steps": self.no_improve_steps, 
            "best_score": self.best_score,  
            "hit_max_steps": hit_max_steps,
            "hit_patience": hit_patience, 
            "target_hit": target_hit, 
            **stats, 
        }
        self._last_info = info 

        if return_info:  
            return done, info  
        return done

    def last_info(self) -> Dict[str, Any]:
        return dict(self._last_info)
