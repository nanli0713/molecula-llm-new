from rdkit.Chem import rdchem
from rdkit import Chem
from rdkit.Chem import BRICS
from collections import defaultdict

def _bt_to_str(bt):
    if bt == rdchem.BondType.SINGLE:
        return "single"
    if bt == rdchem.BondType.DOUBLE:
        return "double"
    if bt == rdchem.BondType.TRIPLE:
        return "triple"
    if bt == rdchem.BondType.AROMATIC:
        return "aromatic"
    return str(bt)

def _atom_summary(a):
    hyb = str(a.GetHybridization()).split(".")[-1]
    return {
        "symbol": a.GetSymbol(),
        "is_aromatic": a.GetIsAromatic(),
        "in_ring": a.IsInRing(),
        "hybridization": hyb,
        "formal_charge": a.GetFormalCharge(),
        "degree": a.GetDegree(),
        "implicit_h": a.GetNumImplicitHs(),
        "explicit_h": a.GetTotalNumHs()
    }

def _neighbor_env(mol, center_idx, exclude_idx=None):
    a = mol.GetAtomWithIdx(center_idx)
    env = []
    for b in a.GetBonds():
        j = b.GetOtherAtomIdx(center_idx)
        if exclude_idx is not None and j == exclude_idx:
            continue
        bj = {
            "neighbor_idx": j,
            "neighbor_symbol": mol.GetAtomWithIdx(j).GetSymbol(),
            "bond_type": _bt_to_str(b.GetBondType()),
            "bond_in_ring": b.IsInRing(),
            "bond_aromatic": (b.GetBondType()==rdchem.BondType.AROMATIC) or b.GetIsAromatic()
        }
        env.append(bj)
    return env

def _format_cut_text(label, bond_info, local_atom_info, partner_atom_info, local_neighbors):
    # Helpers kept local to avoid changing external structure
    def _article_for(token: str) -> str:
        t = (token or "").strip().lower()
        # "sp", "sp2", "sp3" should use "an"
        if t.startswith(("a", "e", "i", "o", "u", "sp")):
            return "an"
        return "a"

    def _hyb_text(hyb: str) -> str:
        if not hyb:
            return "unknown hybridization"
        # normalize: SP2 -> sp2
        h = hyb.lower()
        # RDKit sometimes gives "UNSPECIFIED"
        if h in ("unspecified", "other"):
            return "unknown hybridization"
        return h

    def _aromaticity_text(is_aromatic: bool) -> str:
        return "aromatic" if is_aromatic else "aliphatic"

    def _ring_text(in_ring: bool) -> str:
        return "in a ring" if in_ring else "not in a ring"

    def _atom_phrase(info: dict) -> str:
        if not info:
            return "an unknown atom"
        sym = info.get("symbol", "?")
        hyb = _hyb_text(info.get("hybridization"))
        arom = _aromaticity_text(bool(info.get("is_aromatic")))
        ring = _ring_text(bool(info.get("in_ring")))
        art = _article_for(hyb)
        return f"{art} {hyb} {sym} that is {arom} and {ring}"

    def _bond_phrase(bi: dict) -> str:
        if not bi:
            return "a bond"
        is_arom = bool(bi.get("is_aromatic"))
        btype = (bi.get("type") or "unknown").lower()
        is_conj = bool(bi.get("is_conjugated"))
        # Prefer "aromatic bond" over single/double naming
        if is_arom:
            base = "an aromatic bond"
        else:
            if btype in ("single", "double", "triple"):
                base = f"{_article_for(btype)} {btype} bond"
            else:
                base = "a bond"
        if is_conj and not is_arom:
            base = f"{base} (conjugated)"
        return base

    def _neighbors_phrase(neighs: list) -> str:
        if not neighs:
            return "no other neighbors"
        items = []
        for n in neighs:
            bt = (n.get("bond_type") or "unknown").lower().replace("_", " ")
            sym = n.get("neighbor_symbol") or "atom"
            ar_tag = "aromatic " if n.get("bond_aromatic") else ""
            ring_tag = " in a ring" if n.get("bond_in_ring") else ""
            items.append(f"{bt}-bonded {ar_tag}{sym}{ring_tag}".strip())
        if len(items) == 1:
            return items[0]
        return ", ".join(items[:-1]) + ", and " + items[-1]

    # Compose the sentence
    label_txt = f"[{label}*]" if label is not None else "[*]"
    ring_note = (
        "This break occurs within a ring."
        if bond_info and bond_info.get("in_ring")
        else "This break occurs in a side chain."
    )

    text = (
        f"Attachment point {label_txt}: "
        f"The local atom is {_atom_phrase(local_atom_info)}. "
        f"It is connected by {_bond_phrase(bond_info)} to {_atom_phrase(partner_atom_info)}. "
        f"{ring_note} "
        f"The local atom's other neighbors are: {_neighbors_phrase(local_neighbors)}."
    )
    return text

def build_brics_fragment_contexts(mol, replacement_threshold=5):
    """
    返回：dict[str -> dict], key 为片段 SMILES（带 [n*]），
    value 为包含 labels、cuts（每个断点的结构化信息与可读摘要）、以及 fragment_type 的上下文。
    """
    m = Chem.Mol(mol)
    for a in m.GetAtoms():
        a.SetIntProp("_OrigIdx", a.GetIdx())

    brics_bonds = BRICS.FindBRICSBonds(m)

    adj = defaultdict(list)
    for (i, j), (li, lj) in brics_bonds:
        b = m.GetBondBetweenAtoms(i, j)
        bt = b.GetBondType()
        info_ij = {
            "self_idx": i,
            "partner_idx": j,
            "self_label": int(li),
            "partner_label": int(lj),
            "bond_type": bt,
            "bond_type_str": _bt_to_str(bt),
            "in_ring": b.IsInRing(),
            "is_aromatic": (bt==rdchem.BondType.AROMATIC) or b.GetIsAromatic(),
            "is_conjugated": b.GetIsConjugated(),
        }
        info_ji = {
            "self_idx": j,
            "partner_idx": i,
            "self_label": int(lj),
            "partner_label": int(li),
            "bond_type": bt,
            "bond_type_str": _bt_to_str(bt),
            "in_ring": b.IsInRing(),
            "is_aromatic": (bt==rdchem.BondType.AROMATIC) or b.GetIsAromatic(),
            "is_conjugated": b.GetIsConjugated(),
        }
        adj[i].append(info_ij)
        adj[j].append(info_ji)

    broken = BRICS.BreakBRICSBonds(Chem.Mol(m))
    frags = Chem.GetMolFrags(broken, asMols=True, sanitizeFrags=True)

    frag_map = {}
    for fm in frags:
        s = Chem.MolToSmiles(fm, isomericSmiles=True)
        frag_map[s] = fm

    ctx = {}
    for s, fm in frag_map.items():
        labels = []
        cuts = []
        used_pairs = set()

        for a in fm.GetAtoms():
            if a.GetAtomicNum() != 0:
                continue
            label = int(a.GetIsotope()) if a.GetIsotope() else None
            if label is not None:
                labels.append(label)

            nbrs = [n for n in a.GetNeighbors() if n.GetAtomicNum() != 0]
            if not nbrs:
                continue
            local_atom = nbrs[0]
            if not local_atom.HasProp("_OrigIdx"):
                continue
            local_idx = int(local_atom.GetIntProp("_OrigIdx"))

            candidates = adj.get(local_idx, [])
            match = None
            if candidates:
                cand1 = [c for c in candidates if label is None or c["self_label"] == label]
                cand2 = [c for c in candidates if label is not None and c["partner_label"] == label]
                for cset in (cand1, cand2, candidates):
                    for c in cset:
                        key = (c["self_idx"], c["partner_idx"])
                        if key not in used_pairs:
                            match = c
                            used_pairs.add(key)
                            break
                    if match:
                        break

            local_info = _atom_summary(m.GetAtomWithIdx(local_idx))
            partner_idx = match["partner_idx"] if match else None
            partner_info = _atom_summary(m.GetAtomWithIdx(partner_idx)) if partner_idx is not None else None
            bond_info = {
                "type": match["bond_type_str"] if match else "unknown",
                "in_ring": (match["in_ring"] if match else False),
                "is_aromatic": (match["is_aromatic"] if match else False),
                "is_conjugated": (match["is_conjugated"] if match else False),
            }
            local_neighbors = _neighbor_env(m, local_idx, exclude_idx=partner_idx)

            cut_entry = {
                "label": label,
                "local_atom_idx": local_idx,
                "partner_atom_idx": partner_idx,
                "local_atom": local_info,
                "partner_atom": partner_info,
                "bond": bond_info,
                "brics_types": {
                    "local": (match["self_label"] if match else None),
                    "partner": (match["partner_label"] if match else None),
                },
                "local_neighbors": local_neighbors,
                "text": _format_cut_text(label, bond_info, local_info, partner_info or local_info, local_neighbors)
            }
            cuts.append(cut_entry)

        # 片段级别分类：核心 or 可替换
        has_ring = fm.GetRingInfo().NumRings() > 0
        non_h_atoms = sum(1 for atom in fm.GetAtoms() if atom.GetAtomicNum() > 1)
        if has_ring:
            fragment_type = "core"
        else:
            fragment_type = "replacement_group" if non_h_atoms <= replacement_threshold else "core"

        ctx[s] = {
            "fragment": s,
            "labels": sorted([x for x in labels if x is not None]),
            "n_cuts": len(cuts),
            "cuts": cuts,
            "fragment_type": fragment_type,
            "non_h_atoms": non_h_atoms,
            "has_ring": has_ring,
        }

    return ctx