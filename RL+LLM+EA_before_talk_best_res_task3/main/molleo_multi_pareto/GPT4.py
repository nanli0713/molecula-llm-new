import openai
import re
from rdkit import Chem
import main.molleo_multi.crossover as co, main.molleo_multi.mutate as mu
from openai import OpenAI
from settings import settings
client = OpenAI(
    api_key=settings.api_key9,
    base_url=settings.base_url
)
client2 = OpenAI(
    api_key=settings.api_key8,
    base_url=settings.base_url
)
import random
import os
MINIMUM = 1e-10

def query_LLM(question, model=settings.model, temperature=0.0):
    # model = "gpt-4-0613"
    message = [{"role": "system", "content": "You are a helpful agent who can answer the question based on your molecule knowledge."}]
    prompt1 = question
    message.append({"role": "user", "content": prompt1})
    response = None
    for client_name, current_client in [("primary", client), ("fallback", client2)]:
        for retry in range(3):
            try:
                response = current_client.chat.completions.create(
                    model=model,
                    messages=message,
                    temperature=temperature,
                    stream=False,
                    timeout=90,
                ).choices[0].message.content
                message.append({"role": "assistant", "content": response})
                print("=>")
                return message, response
            except Exception as e:
                print(f"{client_name} retry {retry + 1}: {type(e).__name__} {e}")
    print("=>")
    return message, response

from rdkit import Chem
def is_valid_smiles(s: str, min_atoms=2, require_bond=True) -> bool:
    if not s or not isinstance(s, str):
        return False
    try:
        mol = Chem.MolFromSmiles(s, sanitize=True)
    except Exception:
        return False
    if mol is None:
        return False
    if mol.GetNumAtoms() < min_atoms:
        return False
    if require_bond and mol.GetNumBonds() == 0:
        return False
    return True
def canonicalize_smiles(s: str) -> str | None:
    mol = Chem.MolFromSmiles(s, sanitize=True)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)
class GPT4:
    def __init__(self):
        self.task2description_mul = {
            '1': '# Task:\nI have two molecules and their QED, SA (Synthetic Accessibility), JNK3 (biological activity against the kinase JNK3) scores.\n',
            '2': '# Task:\nI have two molecules and their QED, SA (Synthetic Accessibility), GSK3$eta$ (biological activity against Glycogen Synthase Kinase 3 Beta) scores.\n',
            '3': '# Task:\nI have two molecules and their QED, SA (Synthetic Accessibility), JNK3 (biological activity against the kinase JNK3), GSK3$eta$ (biological activity against Glycogen Synthase Kinase 3 Beta), DRD2 (biological activity against a biological target named the dopamine type 2 receptor (DRD2)) scores.\n',
        }

        self.task2objective_mul = {
            '1': '\n# Target:\nI want to maximize QED score, maximize JNK3 score, and minimize SA score. Please propose a new molecule better than the current molecules. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n',
            '2': '\n# Target:\nI want to maximize QED score, maximize GSK3$eta$ score, and minimize SA score. Please propose a new molecule better than the current molecules. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n',
            '3': '\n# Target:\nI want to maximize QED score, maximize JNK3 score, minimize GSK3$eta$ score, minimize SA score and minimize DRD2 score. Please propose a new molecule better than the current molecules. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n',
        }

        self.good_cases1 = (
            "\n# Note:\n"
            "The following fragments are examples only, not templates. Novel scaffolds are encouraged.\n"
            "\n## Current high frequency useful fragments:\n"
        )
        self.good_cases2 = "\n"

        self.bad_cases1 = (
            "\n## Current high frequency bad fragments:\n"
        )
        self.bad_cases2 = "\n"

        self.requirements = """\n\n# Output format:
        For each molecule, use {<<<Explaination>>>: $EXPLANATION, <<<Molecule>>>: \\box{$Molecule}}. \nRequirements:
        \n- Output exactly a molecule.\n- $EXPLANATION as short as you can.\n- $Molecule must be a valid SMILES.\n\n# Key points:\n- Avoid exact duplicates of previously generated molecules.\n- Prefer genuinely new chemotypes over trivial substitutions.\n- Prefer innovative and high score molecules.
        """


        self.requirements2 = """
# Output format:
{<<<Explaination>>>: $EXPLANATION, <<<Molecule>>>: \\box{$Molecule}}. \nRequirements:\n- Output exactly a molecule.\n- $EXPLANATION as short as you can.\n- $Molecule must be a valid SMILES.
\n\n# Key points:\n- Avoid exact duplicates of previously generated molecules.\n- Prefer genuinely new chemotypes over trivial substitutions.\n- Prefer innovative and high score molecules.
\n# Forbidden exact match high-score molecules:
"""

        self.requirements3 = """
# Output format:
{<<<Explaination>>>: $EXPLANATION, <<<Molecule>>>: \\box{$Molecule}}.\nRequirements:\n- Output exactly a molecule.\n- $EXPLANATION as short as you can.\n- $Molecule must be a valid SMILES.
\n\n# Key points:\n- Avoid exact duplicates of previously generated molecules.\n- Prefer genuinely new chemotypes over trivial substitutions.\n- Prefer innovative and high score molecules.
\n# Forbidden exact match low-score molecules:
"""

        self.general_requirements = """\n# Output format:
        For each molecule, use {<<<Explaination>>>: $EXPLANATION, <<<Molecule>>>: \\box{$Molecule}}. \nRequirements:\n
        1. $EXPLANATION should be your analysis.\n2. The $Molecule should be the smiles of your propsosed molecule.\n3. The molecule should be valid.
        """

    @staticmethod
    def _format_fragment_block(block, header):
        if block is None:
            return ""
        if isinstance(block, str):
            text = block.strip()
            if not text:
                return ""
            return text if text.startswith(header) or not header else f"{header}\n{text}"
        if isinstance(block, (list, tuple)):
            lines = []
            for item in block:
                if isinstance(item, dict):
                    frag = item.get("fragment", "")
                    labels = item.get("labels", [])
                    n_cuts = item.get("n_cuts", len(item.get("cuts", [])))
                    prefix = f"{header}\n" if header and not lines else ""
                    lines.append(
                        f"{prefix}[Mol FRAG] {frag} | This frag has {n_cuts} attachment points = {labels} | cut times={n_cuts}"
                    )
                    for idx, cut in enumerate(item.get("cuts", [])):
                        cut_text = cut.get("text", "") if isinstance(cut, dict) else str(cut)
                        if cut_text:
                            lines.append(f"  {idx} - {cut_text}")
                else:
                    text = str(item).strip()
                    if text:
                        lines.append(text)
            return "\n".join(lines)
        text = str(block).strip()
        if not text:
            return ""
        return f"{header}\n{text}" if header else text

    def edit(
        self,
        parent_mol,
        parent_scores,
        parent_scores_detail,
        mutation_rate,
        h_fragments=None,
        l_fragments=None,
        past_generation=None,
        past_generation_total=None,
        past_generation_total_low=None,
        action=2,
        iter=0
    ):
        task = self.task_mode
        print(f"task:{task}")

        task_definition = self.task2description_mul[task[0]]
        task_objective = self.task2objective_mul[task[0]]

        past_generation_total = past_generation_total or []
        past_generation_total_low = past_generation_total_low or []
        h_fragments = h_fragments or ""
        l_fragments = l_fragments or ""

        past_generation_total_temp = past_generation_total[:1]
        past_generation_total_temp_low = past_generation_total_low[:1]

        print(f"past_generation_comment:", "\n".join(past_generation_total_temp))
        print(f"past_generation_total_temp_low:", "\n".join(past_generation_total_temp_low))

        h_string = self._format_fragment_block(h_fragments, "High score fragments:")
        l_string = self._format_fragment_block(l_fragments, "Low score fragments:")

        try:
            mol_tuple = ''
            for i in range(2):
                tu = (
                    f'\nParent {i}:['
                    + Chem.MolToSmiles(parent_mol[i])
                    + ', total score: '
                    + str(parent_scores[i])
                    + ', details: '
                    + str(parent_scores_detail[i])
                    + ']'
                )
                mol_tuple += tu

            if action == 0:
                if iter % 2 == 0:
                    prompt = (
                        task_definition
                        + mol_tuple
                        + '\n'
                        + task_objective
                        + self.good_cases1
                        + h_string
                        + self.good_cases2
                        + self.requirements2
                        + "\n".join(past_generation_total_temp)
                    )
                else:
                    prompt = (
                        task_definition
                        + mol_tuple
                        + '\n'
                        + task_objective
                        + self.good_cases1
                        + h_string
                        + self.good_cases2
                        + self.requirements
                    )
            elif action == 1:
                if iter % 2 == 0:
                    prompt = (
                        task_definition
                        + mol_tuple
                        + '\n'
                        + task_objective
                        + self.bad_cases1
                        + l_string
                        + self.bad_cases2
                        + self.requirements3
                        + "\n".join(past_generation_total_temp_low)
                    )
                else:
                    prompt = (
                        task_definition
                        + mol_tuple
                        + '\n'
                        + task_objective
                        + self.bad_cases1
                        + l_string
                        + self.bad_cases2
                        + self.requirements
                    )
            elif action == 2:
                prompt = (
                    task_definition
                    + mol_tuple
                    + '\n'
                    + task_objective
                    + self.good_cases1
                    + h_string
                    + self.good_cases2
                    + self.bad_cases1
                    + l_string
                    + self.bad_cases2
                    + self.requirements
                )
            else:
                prompt = task_definition + mol_tuple + '\n' + task_objective + self.general_requirements

            _, r = query_LLM(prompt)

            proposed_smiles = re.findall(r'<<<Molecule>>>: \\box{(.*?)}', r)

            print(f"action: {action}")
            if action == 0:
                print(f"prompt: {prompt}")

            if len(proposed_smiles) == 0:
                proposed_smiles = re.findall(r'ox{(.*?)}', r)

            if len(proposed_smiles) == 0:
                proposed_smiles = re.findall(r'<<<Molecule>>>\s*:\s*\\box\{(.*?)\}', r, flags=re.S)

            if len(proposed_smiles) == 0:
                proposed_smiles = re.findall(r'\\box\{(.*?)\}', r, flags=re.S)

            if len(proposed_smiles) == 0:
                proposed_smiles = re.findall(r'\\boxed\{(.*?)\}', r, flags=re.S)

            if len(proposed_smiles) == 0:
                proposed_smiles = re.findall(r'<<<Molecule>>>\s*:\s*([^\n,}]+)', r, flags=re.S)

            proposed_smiles = [s.strip() for s in proposed_smiles]
            proposed_smiles = [s for s in proposed_smiles if is_valid_smiles(s)]
            proposed_smiles = [canonicalize_smiles(s) for s in proposed_smiles]
            proposed_smiles = [sanitize_smiles(item) for item in proposed_smiles]
            proposed_smiles = [item for item in proposed_smiles if item is not None]

            if len(proposed_smiles) == 0 and action in (0, 1, 2):
                retry_prompt = prompt + "\nReturn exactly one valid SMILES inside the required box."
                _, retry_r = query_LLM(retry_prompt)
                proposed_smiles = re.findall(r'<<<Molecule>>>: \\box{(.*?)}', retry_r)
                if len(proposed_smiles) == 0:
                    proposed_smiles = re.findall(r'\\box\{(.*?)\}', retry_r, flags=re.S)
                proposed_smiles = [s.strip() for s in proposed_smiles]
                proposed_smiles = [s for s in proposed_smiles if is_valid_smiles(s)]
                proposed_smiles = [canonicalize_smiles(s) for s in proposed_smiles]
                proposed_smiles = [sanitize_smiles(item) for item in proposed_smiles]
                proposed_smiles = [item for item in proposed_smiles if item is not None]

            # 去重，只保留一个候选
            uniq_smiles = []
            seen = set()
            for s in proposed_smiles:
                if s not in seen:
                    uniq_smiles.append(s)
                    seen.add(s)
            proposed_smiles = uniq_smiles[:1]

            print('proposed_smiles: ', proposed_smiles)
            assert proposed_smiles is not None and len(proposed_smiles) > 0

            new_childs = [Chem.MolFromSmiles(item) for item in proposed_smiles]

            return new_childs, parent_mol

        except Exception as e:
            print(f"{type(e).__name__} {e}")
            new_child = co.crossover(parent_mol[0], parent_mol[1])
            if new_child is not None:
                new_child = mu.mutate(new_child, mutation_rate)
            if new_child is None:
                return [], parent_mol
            return [new_child], parent_mol
           
def sanitize_smiles(smi):
    if smi == '':
        return None
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=True)
        smi_canon = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        return smi_canon
    except:
        return None
