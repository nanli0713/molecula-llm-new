import openai
import re
from rdkit import Chem
import main.molleo.crossover as co, main.molleo.mutate as mu
from openai import OpenAI
import os
import random
from settings import settings
import random
import os
MINIMUM = 1e-10

def build_openai_client(api_key, base_url):
    if not api_key:
        return None
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)

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
    def __init__(self, api_key=None, base_url=None, fallback_api_key=None):
        resolved_base_url = base_url or settings.base_url
        self.client = build_openai_client(api_key or settings.api_key, resolved_base_url)
        self.client2 = build_openai_client(fallback_api_key or settings.api_key2, resolved_base_url)
        self.task2description = {
            'qed': 'I have two molecules and their QED scores. The QED score measures the drug-likeness of the molecule.\n\n',
            'jnk3': 'I have two molecules and their JNK3 scores. The JNK3 score measures a molecular\'s biological activity against JNK3.\n\n',
            'drd2': 'I have two molecules and their DRD2 scores. The DRD2 score measures a molecule\'s biological activity against a biological target named the dopamine type 2 receptor (DRD2).\n\n',
            'gsk3b': 'I have two molecules and their GSK3$\beta$ scores. The GSK3$\beta$ score measures a molecular\'s biological activity against Glycogen Synthase Kinase 3 Beta.\n\n',
            'isomers_C9H10N2O2PF2Cl': 'I have two molecules and their isomer scores. The isomer score measures a molecule\'s similarity in terms of atom counter to C9H10N2O2PF2Cl.\n\n',
            'perindopril_mpo': 'I have two molecules and their perindopril multiproperty objective scores. The perindopril multiproperty objective score measures the geometric means of several scores, including the molecule\'s Tanimoto similarity to perindopril and number of aromatic rings.\n\n',
            'sitagliptin_mpo': 'I have two molecules and their sitagliptin multiproperty objective scores. The sitagliptin rediscovery score measures the geometric means of several scores, including the molecule\'s Tanimoto similarity to sitagliptin, TPSA score, LogP score and isomer score with C16H15F6N5O.\n\n',
            'ranolazine_mpo': 'I have two molecules and their ranolazine multiproperty objective scores. The ranolazine multiproperty objective score measures the geometric means of several scores, including the molecule\'s Tanimoto similarity to ranolazine, TPSA score LogP score and number of fluorine atoms.\n\n',
            'thiothixene_rediscovery': 'I have two molecules and their thiothixene rediscovery measures a molecule\'s Tanimoto similarity with thiothixene\'s SMILES to check whether it could be rediscovered.\n\n',
            'mestranol_similarity': 'I have two molecules and their mestranol similarity scores. The mestranol similarity score measures a molecule\'s Tanimoto similarity with Mestranol.\n\n',
            'albuterol_similarity': 'I have two molecules and their albuterol similarity scores. The albuterol similarity score measures a molecule\'s Tanimoto similarity with Albuterol.\n\n',
            'celecoxib_rediscovery': 'I have two molecules and their celecoxib rediscovery scores. The celecoxib rediscovery score measures a molecule\'s Tanimoto similarity with celecoxib\'s SMILES to check whether it could be rediscovered.\n\n',
            'amlodipine_mpo': 'I have two molecules and their amlodipine multiproperty objective scores. The amlodipine multiproperty objective score measures the geometric means of several scores, including the molecule\'s Tanimoto similarity to amlodipine and other molecular property constraints.\n\n',
            'fexofenadine_mpo': 'I have two molecules and their fexofenadine multiproperty objective scores. The fexofenadine multiproperty objective score measures the geometric means of several scores, including the molecule\'s Tanimoto similarity to fexofenadine and other molecular property constraints.\n\n',
            'osimertinib_mpo': 'I have two molecules and their osimertinib multiproperty objective scores. The osimertinib multiproperty objective score measures the geometric means of several scores, including the molecule\'s Tanimoto similarity to osimertinib and other molecular property constraints.\n\n',
            'zaleplon_mpo': 'I have two molecules and their zaleplon multiproperty objective scores. The zaleplon multiproperty objective score measures the geometric means of several scores, including the molecule\'s Tanimoto similarity to zaleplon and other molecular property constraints.\n\n',
            'median1': 'I have two molecules and their median1 scores. The median1 score is a multi-objective score that measures how well a molecule balances several molecular properties.\n\n',
            'median2': 'I have two molecules and their median2 scores. The median2 score is a multi-objective score that measures how well a molecule balances several molecular properties.\n\n',
            'isomers_c7h8n2o2': 'I have two molecules and their isomer scores. The isomer score measures a molecule\'s similarity in terms of atom counter to C7H8N2O2.\n\n',
            'deco_hop': 'I have two molecules and their decoration hopping scores. The decoration hopping score measures how well a molecule preserves the core scaffold while changing the decorations to satisfy the task objective.\n\n',
            'scaffold_hop': 'I have two molecules and their scaffold hopping scores. The scaffold hopping score measures how well a molecule changes the core scaffold while preserving desired molecular features and biological relevance.\n\n',
            'Valsartan_SMARTS': 'I have two molecules and their valsartan SMARTS scores. The valsartan SMARTS score measures whether a molecule satisfies the SMARTS pattern and structural requirements related to valsartan.\n\n',
            'troglitazone_rediscovery': 'I have two molecules and their troglitazone rediscovery scores. The troglitazone rediscovery score measures a molecule\'s Tanimoto similarity with troglitazone\'s SMILES to check whether it could be rediscovered.\n\n',
        }
        self.task2objective = {
            'qed': 'Please propose a new molecule that has a higher QED score. You can either make crossover and mutations or just propose a new molecule based on your knowledge. \n\n',
            'jnk3': 'Please propose a new molecule that has a higher JNK3 score. You can either make crossover and mutations or just propose a new molecule based on your knowledge. \n\n',
            'drd2': 'Please propose a new molecule that has a higher DRD2 score. You can either make crossover and mutations or just propose a new molecule based on your knowledge. \n\n',
            'gsk3b': 'Please propose a new molecule that has a higher GSK3$\beta$ score. You can either make crossover and mutations or just propose a new molecule based on your knowledge. \n\n',
            'isomers_C9H10N2O2PF2Cl': 'Please propose a new molecule that has a higher isomer score. You can either make crossover and mutations or just propose a new molecule based on your knowledge. \n\n',
            'perindopril_mpo': 'Please propose a new molecule that has a higher perindopril multiproperty objective score. You can either make crossover and mutations or just propose a new molecule based on your knowledge. \n\n',
            'sitagliptin_mpo': 'Please propose a new molecule that has a higher sitagliptin multiproperty objective score. You can either make crossover and mutations or just propose a new molecule based on your knowledge. \n\n',
            'ranolazine_mpo': 'Please propose a new molecule that has a higher ranolazine multiproperty objective score. You can either make crossover and mutations or just propose a new molecule based on your knowledge. \n\n',
            'thiothixene_rediscovery': 'Please propose a new molecule that has a higher thiothixene rediscovery score. You can either make crossover and mutations or just propose a new molecule based on your knowledge. \n\n',
            'mestranol_similarity': 'Please propose a new molecule that has a higher mestranol similarity score. You can either make crossover and mutations or just propose a new molecule based on your knowledge. \n\n',
            'albuterol_similarity': 'Please propose a new molecule that has a higher albuterol similarity score. You can either make crossover and mutations or just propose a new molecule based on your knowledge. \n\n',
            'celecoxib_rediscovery': 'Please propose a new molecule that has a higher celecoxib rediscovery score. You can either make crossover and mutations or just propose a new molecule based on your knowledge. \n\n',
            'amlodipine_mpo': 'Please propose a new molecule that has a higher amlodipine multiproperty objective score. You can either make crossover and mutations or just propose a new molecule based on your knowledge. \n\n',
            'fexofenadine_mpo': 'Please propose a new molecule that has a higher fexofenadine multiproperty objective score. You can either make crossover and mutations or just propose a new molecule based on your knowledge. \n\n',
            'osimertinib_mpo': 'Please propose a new molecule that has a higher osimertinib multiproperty objective score. You can either make crossover and mutations or just propose a new molecule based on your knowledge. \n\n',
            'zaleplon_mpo': 'Please propose a new molecule that has a higher zaleplon multiproperty objective score. You can either make crossover and mutations or just propose a new molecule based on your knowledge. \n\n',
            'median1': 'Please propose a new molecule that has a higher median1 score. You can either make crossover and mutations or just propose a new molecule based on your knowledge. \n\n',
            'median2': 'Please propose a new molecule that has a higher median2 score. You can either make crossover and mutations or just propose a new molecule based on your knowledge. \n\n',
            'isomers_c7h8n2o2': 'Please propose a new molecule that has a higher isomer score. You can either make crossover and mutations or just propose a new molecule based on your knowledge. \n\n',
            'deco_hop': 'Please propose a new molecule that has a higher decoration hopping score. You can either make crossover and mutations or just propose a new molecule based on your knowledge. \n\n',
            'scaffold_hop': 'Please propose a new molecule that has a higher scaffold hopping score. You can either make crossover and mutations or just propose a new molecule based on your knowledge. \n\n',
            'Valsartan_SMARTS': 'Please propose a new molecule that has a higher valsartan SMARTS score. You can either make crossover and mutations or just propose a new molecule based on your knowledge. \n\n',
            'troglitazone_rediscovery': 'Please propose a new molecule that has a higher troglitazone rediscovery score. You can either make crossover and mutations or just propose a new molecule based on your knowledge. \n\n',
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
- Output exactly a molecule.
- $EXPLANATION as short as you can.
- $Molecule must be a valid SMILES.
"""

        self.requirements2 = """
# Output format:
{<<<Explaination>>>: $EXPLANATION, <<<Molecule>>>: \\box{$Molecule}}

Requirements:
- Output exactly a molecule.
- $EXPLANATION as short as you can.
- $Molecule must be a valid SMILES.

# Forbidden exact match:
"""

        self.requirements3 = """
# Output format:
{<<<Explaination>>>: $EXPLANATION, <<<Molecule>>>: \\box{$Molecule}}

Requirements:
- Output exactly one molecule.
- $EXPLANATION as short as you can.
- $Molecule must be a valid SMILES.

# Forbidden low-score molecules:
"""

        self.general_requirements = """
# Output format:
{<<<Explaination>>>: $EXPLANATION, <<<Molecule>>>: \\box{$Molecule}}

Requirements:
- Output exactly one molecule.
- $EXPLANATION as short as you can.
- $Molecule must be a valid SMILES.
"""

        self.task=None

    def query_LLM(self, question, model='gpt-5.2', temperature=0.0):
        # model = 'gpt-4-0613'
        message = [{"role": "system", "content": "You are a helpful agent who can answer the question based on your molecule knowledge."}]
        prompt1 = question
        message.append({"role": "user", "content": prompt1})
        response = None
        clients = [self.client, self.client2]
        for client_index, client in enumerate(clients):
            if client is None:
                continue
            for retry in range(3):
                try:
                    response = client.chat.completions.create(
                        model=model, messages=message, temperature=temperature, stream=False,
                    ).choices[0].message.content
                    message.append({"role": "assistant", "content": response})
                    print("=>")
                    return message, response
                except Exception as e:
                    prefix = "second times" if client_index == 1 else "first times"
                    print(f"{prefix}: {type(e).__name__} {e}")
        print("=>")
        return message, response

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

        task = self.task
        print(f"task:{task}")
        task_definition = self.task2description[task[0]]
        task_objective = self.task2objective[task[0]]
        past_generation_total_temp = past_generation_total[:1]
        past_generation_total_temp_low = past_generation_total_low[:1]
        print(f"past_generation_comment:", "\n".join(past_generation_total_temp))
        print(f"past_generation_total_temp_low:", "\n".join(past_generation_total_temp_low))
        # h_string = "\n".join(h_fragments)
        h_string = h_fragments
        # l_string = "\n".join(l_fragments)
        l_string = l_fragments
        try:
            mol_tuple = ''
            for i in range(2):
                tu = (
                    f'\nParent {i}:['
                    + Chem.MolToSmiles(parent_mol[i])
                    + ', total score: '
                    + str(parent_scores[i])
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

            _, r = self.query_LLM(prompt)
            if r is None:
                print("LLM did not return a response.")
                return None, parent_mol
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
    """
    Return a canonical smile representation of smi 

    Parameters
    ----------
    smi : str
        smile string to be canonicalized 

    Returns
    -------
    mol (rdkit.Chem.rdchem.Mol) : 
        RdKit mol object (None if invalid smile string smi)
    smi_canon (string)          : 
        Canonicalized smile representation of smi (None if invalid smile string smi)
    conversion_successful (bool): 
        True/False to indicate if conversion was  successful 
    """
    if smi == '':
        return None
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=True)
        smi_canon = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        return smi_canon
    except:
        return None
