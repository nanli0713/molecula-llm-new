import os
import sys
import logging
from types import SimpleNamespace

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from transformers import AutoModel, AutoTokenizer

MOLECULESTM_ROOT = os.environ.get("MOLECULESTM_ROOT", "/root/autodl-tmp/MoleculeSTM-main")
if MOLECULESTM_ROOT not in sys.path:
    sys.path.append(MOLECULESTM_ROOT)

from MoleculeSTM.models import MLP
from MoleculeSTM.models.mega_molbart.mega_mol_bart import MegaMolBART
from MoleculeSTM.utils import prepare_text_tokens

RDLogger.DisableLog("rdApp.*")
logging.getLogger("MoleculeSTM.models.mega_molbart.mega_mol_bart").setLevel(logging.ERROR)


def _scheduled_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1.0, (1.0 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
    lr_ramp = lr_ramp * min(1.0, t / rampup)
    return initial_lr * lr_ramp


def _freeze(module):
    module.eval()
    for param in module.parameters():
        param.requires_grad = False
    return module


def _canonical_mol(smiles):
    if not smiles or not isinstance(smiles, str):
        return None, None
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        return None, None
    smiles = Chem.MolToSmiles(mol, canonical=True)
    return Chem.MolFromSmiles(smiles), smiles


def _as_smiles(mol_or_smiles):
    if isinstance(mol_or_smiles, str):
        _, smiles = _canonical_mol(mol_or_smiles)
        return smiles
    if isinstance(mol_or_smiles, Chem.Mol):
        return Chem.MolToSmiles(mol_or_smiles, canonical=True)
    return None


def _mean_pooling(token_embeddings, pad_mask):
    keep_mask = (~pad_mask).unsqueeze(-1).expand_as(token_embeddings).float()
    summed = torch.sum(token_embeddings * keep_mask, dim=0)
    denom = torch.clamp(keep_mask.sum(dim=0), min=1e-9)
    return summed / denom


class MoleculeSTMEditor(nn.Module):
    """Text-guided MoleculeSTM/MegaMolBART editor with a BioT5-like API."""

    def __init__(self, args=None):
        super().__init__()
        args = args or SimpleNamespace()

        root = getattr(args, "moleculestm_root", MOLECULESTM_ROOT)
        data_dir = os.path.join(root, "data")
        default_model_dir = os.path.join(
            data_dir,
            "pretrained_MoleculeSTM",
            "SciBERT-MegaMolBART-3e-5-1-1e-4-1-InfoNCE-0.1-32-32",
        )
        default_edit_dir = os.path.join(
            default_model_dir,
            "downstream_language_edit",
            "MegaMolBART_ZINC250K_1e-2_RR_no_normalize",
        )

        self.args = SimpleNamespace(
            seed=getattr(args, "seed", [42])[0] if isinstance(getattr(args, "seed", 42), list) else getattr(args, "seed", 42),
            device=getattr(args, "moleculestm_device", 0),
            dataspace_path=getattr(args, "moleculestm_dataspace_path", data_dir),
            model_dir=getattr(args, "moleculestm_model_dir", default_model_dir),
            language_edit_model_dir=getattr(args, "moleculestm_language_edit_model_dir", default_edit_dir),
            megamolbart_dir=getattr(
                args,
                "moleculestm_megamolbart_dir",
                os.path.join(data_dir, "pretrained_MegaMolBART", "checkpoints"),
            ),
            vocab_path=getattr(args, "moleculestm_vocab_path", os.path.join(root, "MoleculeSTM", "bart_vocab.txt")),
            ssl_emb_dim=getattr(args, "moleculestm_ssl_emb_dim", 256),
            max_seq_len=getattr(args, "moleculestm_max_seq_len", 512),
            lr=getattr(args, "moleculestm_lr", 0.1),
            epochs=getattr(args, "moleculestm_epochs", 15),
            l2_lambdas=getattr(args, "moleculestm_l2_lambdas", "10,1,0.1"),
            beam_size=getattr(args, "moleculestm_beam_size", 3),
            max_fragments=getattr(args, "moleculestm_max_fragments", 5),
            normalize=True,
        )

        self.device = torch.device(
            "cuda:" + str(self.args.device) if torch.cuda.is_available() else "cpu"
        )
        self.task = None
        self.text_cache = {}
        self.task2description = self._build_task_descriptions()

        self._set_seed()
        self._load_modules()

    def _set_seed(self):
        np.random.seed(int(self.args.seed))
        torch.manual_seed(int(self.args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(self.args.seed))

    @staticmethod
    def _build_task_descriptions():
        return {
            "qed": "This molecule has high drug-likeness.",
            "jnk3": "This molecule is a strong inhibitor of JNK3.",
            "gsk3b": "This molecule is a strong inhibitor of GSK3B.",
            "drd2": "This molecule is a strong inhibitor of DRD2.",
            "isomers_c9h10n2o2pf2cl": "This molecule has the molecular formula C9H10N2O2PF2Cl.",
            "isomers_C9H10N2O2PF2Cl": "This molecule has the molecular formula C9H10N2O2PF2Cl.",
            "isomers_c7h8n2o2": "This molecule has the molecular formula C7H8N2O2.",
            "perindopril_mpo": "This molecule has high multiproperty similarity to Perindopril.",
            "sitagliptin_mpo": "This molecule has high multiproperty similarity to Sitagliptin.",
            "ranolazine_mpo": "This molecule has high multiproperty similarity to Ranolazine.",
            "thiothixene_rediscovery": "This molecule looks like Thiothixene.",
            "mestranol_similarity": "This molecule looks like Mestranol.",
            "albuterol_similarity": "This molecule looks like Albuterol.",
            "celecoxib_rediscovery": "This molecule looks like Celecoxib.",
            "troglitazone_rediscovery": "This molecule looks like Troglitazone.",
            "amlodipine_mpo": "This molecule has high multiproperty similarity to Amlodipine.",
            "fexofenadine_mpo": "This molecule has high multiproperty similarity to Fexofenadine.",
            "osimertinib_mpo": "This molecule has high multiproperty similarity to Osimertinib.",
            "zaleplon_mpo": "This molecule has high multiproperty similarity to Zaleplon.",
            "median1": "This molecule has balanced drug-like molecular properties.",
            "median2": "This molecule has balanced pharmacokinetic molecular properties.",
            "deco_hop": "This molecule preserves the molecular core while improving substituents.",
            "scaffold_hop": "This molecule preserves biological relevance while changing the molecular scaffold.",
            "Valsartan_SMARTS": "This molecule satisfies structural requirements related to Valsartan.",
        }

    def _load_modules(self):
        scibert_cache = os.path.join(self.args.dataspace_path, "pretrained_SciBERT")
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            "allenai/scibert_scivocab_uncased",
            cache_dir=scibert_cache,
        )
        self.text_model = AutoModel.from_pretrained(
            "allenai/scibert_scivocab_uncased",
            cache_dir=scibert_cache,
        )
        text_model_path = os.path.join(self.args.model_dir, "text_model.pth")
        if os.path.exists(text_model_path):
            print("Loading MoleculeSTM text model from {}...".format(text_model_path))
            self.text_model.load_state_dict(torch.load(text_model_path, map_location="cpu"))

        self.MegaMolBART_wrapper = MegaMolBART(
            vocab_path=self.args.vocab_path,
            input_dir=self.args.megamolbart_dir,
            output_dir=None,
            grad_enabled=True,
        )
        self.molecule_model = self.MegaMolBART_wrapper.model

        self.text2latent = nn.Linear(768, self.args.ssl_emb_dim)
        self._load_state(self.text2latent, os.path.join(self.args.model_dir, "text2latent_model.pth"))

        self.generation2MoleculeSTM = MLP(256, [self.args.ssl_emb_dim, self.args.ssl_emb_dim])
        self._load_state(
            self.generation2MoleculeSTM,
            os.path.join(self.args.language_edit_model_dir, "generation2foundation_model.pth"),
        )

        self.MoleculeSTM2generation = MLP(self.args.ssl_emb_dim, [256, 256])
        self._load_state(
            self.MoleculeSTM2generation,
            os.path.join(self.args.language_edit_model_dir, "foundation2generation_model.pth"),
        )

        self.text_model = _freeze(self.text_model.to(self.device))
        self.molecule_model = _freeze(self.molecule_model.to(self.device))
        self.text2latent = _freeze(self.text2latent.to(self.device))
        self.generation2MoleculeSTM = _freeze(self.generation2MoleculeSTM.to(self.device))
        self.MoleculeSTM2generation = _freeze(self.MoleculeSTM2generation.to(self.device))

    @staticmethod
    def _load_state(module, path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        print("Loading {}...".format(path))
        module.load_state_dict(torch.load(path, map_location="cpu"))

    def _task_description(self):
        if not self.task:
            return "This molecule has improved molecular properties."
        task_name = self.task[0] if isinstance(self.task, (list, tuple)) else self.task
        return self.task2description.get(task_name, self.task2description.get(str(task_name).lower(), "This molecule has a high {} score.".format(task_name)))

    @staticmethod
    def _format_fragments(fragments, max_fragments):
        if fragments is None:
            return []
        items = []
        if isinstance(fragments, str):
            raw_items = fragments.splitlines()
        elif isinstance(fragments, (list, tuple)):
            raw_items = fragments
        else:
            raw_items = [fragments]

        for item in raw_items:
            if isinstance(item, dict):
                text = item.get("fragment", "")
            else:
                text = str(item)
                if "FRAG]" in text:
                    text = text.split("FRAG]", 1)[1]
                if "|" in text:
                    text = text.split("|", 1)[0]
            text = text.strip()
            if text and text not in items:
                items.append(text)
            if len(items) >= max_fragments:
                break
        return items

    def _condition_text(self, h_fragments=None, l_fragments=None, action=3):
        base = self._task_description()
        high = self._format_fragments(h_fragments, self.args.max_fragments)
        low = self._format_fragments(l_fragments, self.args.max_fragments)

        text = base + " Optimize the molecule toward this objective."
        if action == 0 and high:
            text += " Encourage beneficial high-scoring fragment patterns similar to: {}.".format(", ".join(high))
        elif action == 1 and low:
            text += " Avoid unfavorable low-scoring fragment patterns similar to: {}.".format(", ".join(low))
        elif action == 2:
            if high:
                text += " Encourage beneficial high-scoring fragment patterns similar to: {}.".format(", ".join(high))
            if low:
                text += " Avoid unfavorable low-scoring fragment patterns similar to: {}.".format(", ".join(low))
        return text

    def _get_text_repr(self, text):
        if text in self.text_cache:
            return self.text_cache[text]
        with torch.no_grad():
            token_ids, masks = prepare_text_tokens(
                device=self.device,
                description=[text],
                tokenizer=self.text_tokenizer,
                max_seq_len=self.args.max_seq_len,
            )
            output = self.text_model(input_ids=token_ids, attention_mask=masks)
            text_repr = self.text2latent(output["pooler_output"]).detach()
        self.text_cache[text] = text_repr
        return text_repr

    @staticmethod
    def _clip_loss(molecule_repr, text_repr):
        molecule_repr = F.normalize(molecule_repr, dim=-1)
        text_repr = F.normalize(text_repr, dim=-1)
        return -torch.mm(molecule_repr, text_repr.transpose(0, 1))[0]

    def _parse_l2_lambdas(self):
        if isinstance(self.args.l2_lambdas, str):
            return [float(x) for x in self.args.l2_lambdas.split(",") if x.strip()]
        if isinstance(self.args.l2_lambdas, (list, tuple)):
            return [float(x) for x in self.args.l2_lambdas]
        return [float(self.args.l2_lambdas)]

    def _decode_latent(self, latent, pad_mask, parent_smiles=None, l2_lambda=None):
        smiles_list = self.MegaMolBART_wrapper.inverse_transform(
            [latent.detach()],
            pad_mask.bool().cuda(),
            k=int(self.args.beam_size),
            sanitize=False,
        )
        mols = []
        seen = set()
        accepted_smiles = []
        for smiles in smiles_list:
            mol, canonical = _canonical_mol(smiles)
            if mol is None or canonical in seen:
                continue
            seen.add(canonical)
            mols.append(mol)
            accepted_smiles.append(canonical)

        prefix = "[MoleculeSTM decode]"
        context = []
        if parent_smiles is not None:
            context.append("parent={}".format(parent_smiles))
        if l2_lambda is not None:
            context.append("l2={}".format(l2_lambda))
        context.append("raw={}".format(smiles_list))
        context.append("valid={}".format(accepted_smiles))
        print("{} {}".format(prefix, " | ".join(context)))
        return mols

    def _edit_one(self, smiles, text):
        text_repr = self._get_text_repr(text)
        with torch.no_grad():
            latent_init, pad_mask = self.MegaMolBART_wrapper.smileslist2embedding([smiles])
        latent_init = latent_init.detach()
        pad_mask = pad_mask.detach()

        edited_mols = []
        seen = {smiles}
        for l2_lambda in self._parse_l2_lambdas():
            latent = latent_init.clone().detach().requires_grad_(True)
            optimizer = torch.optim.Adam([latent], lr=float(self.args.lr))

            with torch.enable_grad():
                for epoch in range(int(self.args.epochs)):
                    t = epoch / max(1, int(self.args.epochs))
                    optimizer.param_groups[0]["lr"] = _scheduled_lr(t, float(self.args.lr))
                    molecule_repr_generation = _mean_pooling(latent, pad_mask)
                    if self.args.normalize:
                        molecule_repr_generation = F.normalize(molecule_repr_generation, dim=-1)
                    molecule_repr = self.generation2MoleculeSTM(molecule_repr_generation)
                    loss = self._clip_loss(molecule_repr, text_repr)
                    loss = loss + float(l2_lambda) * ((latent - latent_init) ** 2).mean()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            for mol in self._decode_latent(
                latent,
                pad_mask,
                parent_smiles=smiles,
                l2_lambda=l2_lambda,
            ):
                canonical = Chem.MolToSmiles(mol, canonical=True)
                if canonical in seen:
                    continue
                seen.add(canonical)
                edited_mols.append(mol)

        final_smiles = [Chem.MolToSmiles(mol, canonical=True) for mol in edited_mols]
        print(
            "[MoleculeSTM edit] parent={} | total_valid={} | candidates={}".format(
                smiles,
                len(final_smiles),
                final_smiles,
            )
        )
        return edited_mols

    def edit(
        self,
        smiles_list,
        h_fragments=None,
        l_fragments=None,
        past_generation=None,
        past_generation_total=None,
        action=3,
    ):
        text = self._condition_text(h_fragments=h_fragments, l_fragments=l_fragments, action=action)
        print("MoleculeSTM action {} condition: {}".format(action, text))

        edited_batches = []
        for mol_or_smiles in smiles_list:
            smiles = _as_smiles(mol_or_smiles)
            if smiles is None:
                edited_batches.append([])
                continue
            try:
                edited_batches.append(self._edit_one(smiles, text))
            except Exception as exc:
                print("MoleculeSTM edit failed for {}: {} {}".format(smiles, type(exc).__name__, exc))
                edited_batches.append([])
        return edited_batches
