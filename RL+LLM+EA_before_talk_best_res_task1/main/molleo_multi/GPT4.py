import openai
import re
from rdkit import Chem
import main.molleo_multi.crossover as co, main.molleo_multi.mutate as mu
from openai import OpenAI
from settings import settings
client = OpenAI(
    api_key=settings.api_key6,
    base_url=settings.base_url
)
client2 = OpenAI(
    api_key=settings.api_key2,
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
    flag = 0
    for retry in range(3):
        try:
            response = client2.chat.completions.create(
                model=model, messages=message, temperature=temperature, stream=False,
            ).choices[0].message.content
            message.append({"role": "assistant", "content": response})
            flag = 1
            break
        except Exception as e:
            print(f"{type(e).__name__} {e}")
    if not flag:
        for retry in range(3): 
            try:      
                response = client2.chat.completions.create(
                        model=model, messages=message, temperature=temperature, stream=False,
                    ).choices[0].message.content
                message.append({"role": "assistant", "content": response})
            except Exception as e:
                print(f"sencond times:{type(e).__name__} {e}")
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
            '1': '# Task:\nI have two molecules with QED, SA, and JNK3 scores. Propose 7 novel molecules expected to improve these properties.\n',
            '2': '# Task:\nI have two molecules with QED, SA, and GSK3β scores. Propose 7 novel molecules expected to improve these properties.\n',
            '3': '# Task:\nI have two molecules with QED, SA, JNK3, GSK3β, and DRD2 scores. Propose 7 novel molecules expected to improve these properties.\n',
        }

        self.task2objective_mul = {
            '1': '\n# Target:\nMaximize QED and JNK3, minimize SA. Propose 7 new molecules better than the parents. You can use crossover or mutation or design 7 new molecules directly.\n',
            '2': '\n# Target:\nMaximize QED and GSK3β, minimize SA. Propose 7 new molecules better than the parents. You can use crossover or mutation or design 7 new molecules directly.\n',
            '3': '\n# Target:\nMaximize QED and JNK3, minimize GSK3β, SA, and DRD2. Propose 7 new molecules better than the parents. You can use crossover or mutation or design 7 new molecules directly.\n',
        }

        self.good_cases1 = (
            "\n# Note:\n"
            "The following fragments are examples only, not templates. Novel scaffolds are encouraged.\n"
            "\n## Useful motifs:\n"
        )
        self.good_cases2 = "\n"

        self.bad_cases1 = (
            "\n## Avoid overusing these poorer patterns:\n"
        )
        self.bad_cases2 = "\n"

        self.requirements = """
# Output format:
{<<<Explaination>>>: $EXPLANATION, <<<Molecule>>>: \\box{$Molecule}}

Requirements:
- Output exactly 7 molecules.
- $EXPLANATION as short as you can.
- $Molecule must be a valid SMILES.

# Key points:
- Be as short as possible.
- Avoid exact duplicates of parent molecules.
- Prefer genuinely new chemotypes over trivial substitutions.
- You may use crossover/mutation or propose new molecules directly.
"""

        self.requirements2 = """
# Output format:
{<<<Explaination>>>: $EXPLANATION, <<<Molecule>>>: \\box{$Molecule}}

Requirements:
- Output exactly 7 molecules.
- $EXPLANATION as short as you can.
- $Molecule must be a valid SMILES.

# Key points:
- Be as short as possible.
- Avoid exact duplicates of previously generated molecules.
- Prefer genuinely new chemotypes over trivial substitutions.

# Forbidden exact match:
"""

        self.requirements3 = """
# Output format:
{<<<Explaination>>>: $EXPLANATION, <<<Molecule>>>: \\box{$Molecule}}

Requirements:
- Output exactly 7 molecules.
- $EXPLANATION as short as you can.
- $Molecule must be a valid SMILES.

# Key points:
- Be as short as possible.
- Avoid exact duplicates of previously generated molecules.
- Prefer genuinely new chemotypes over trivial substitutions.

# Forbidden low-score molecules:
"""

        self.general_requirements = """
# Output format:
{<<<Explaination>>>: $EXPLANATION, <<<Molecule>>>: \\box{$Molecule}}

Requirements:
- Output exactly 7 molecules.
- $EXPLANATION as short as you can.
- $Molecule must be a valid SMILES.
"""

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

        h_string = h_fragments
        l_string = l_fragments

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

            # 去重，尽量保留前两个
            uniq_smiles = []
            seen = set()
            for s in proposed_smiles:
                if s not in seen:
                    uniq_smiles.append(s)
                    seen.add(s)
            proposed_smiles = uniq_smiles[:2]

            print('proposed_smiles: ', proposed_smiles)
            assert proposed_smiles is not None and len(proposed_smiles) > 0

            new_childs = [Chem.MolFromSmiles(item) for item in proposed_smiles]

            return new_childs, parent_mol

        except Exception as e:
            print(f"{type(e).__name__} {e}")
            new_child = co.crossover(parent_mol[0], parent_mol[1])
            if new_child is not None:
                new_child = mu.mutate(new_child, mutation_rate)
            return new_child, parent_mol
    
def sanitize_smiles(smi):
    if smi == '':
        return None
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=True)
        smi_canon = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        return smi_canon
    except:
        return None