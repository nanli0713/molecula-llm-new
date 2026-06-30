import openai
import re
from rdkit import Chem
import main.molleo.crossover as co, main.molleo.mutate as mu
from openai import OpenAI
import os
import random
from settings import settings
client = OpenAI(
    api_key=settings.api_key,
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
    message = [{"role": "system", "content": "You are a helpful agent who can answer the question based on your molecule knowledge."}]
    prompt1 = question
    message.append({"role": "user", "content": prompt1})
    flag = 0
    for retry in range(3):
        try:
            response = client.chat.completions.create(
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
                }
        self.task2objective = {
                'qed': 'Please propose a new molecule that has a higher QED score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'jnk3': 'Please propose a new molecule that has a higher JNK3 score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'drd2': 'Please propose a new molecule that has a higher DRD2 score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'gsk3b': 'Please propose a new molecule that has a higher GSK3$\beta$ score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'isomers_C9H10N2O2PF2Cl': 'Please propose a new molecule that has a higher isomer score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'perindopril_mpo': 'Please propose a new molecule that has a higher perindopril multiproperty objective score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'sitagliptin_mpo': 'Please propose a new molecule that has a higher sitagliptin multiproperty objective score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'ranolazine_mpo': 'Please propose a new molecule that has a higher ranolazine multiproperty objective score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'thiothixene_rediscovery': 'Please propose a new molecule that has a higher thiothixene rediscovery score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                'mestranol_similarity': 'Please propose a new molecule that has a higher mestranol similarity score. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n',
                }
        self.good_cases1 = f"\n\n# High-frequency fragments contained in five high-molecular molecules (Available or based on your knowledge)\n\n[The structures below represent molecular fragments, the distribution of surrounding functional groups, and what properties they are beneficial to.]: \n\n"
        self.good_cases2 = f"\n\nThese high substructures are recommended options for improving molecular behavior.\n"

        self.bad_cases1 = f"\n\n# High-frequency fragments contained in five low-molecular molecules (Strictly prohibited)\n\n[The structures below represent molecular fragments, the distribution of surrounding functional groups, what properties are harmful, and alternative fragments to consider.]: \n\n"
        self.bad_cases2 = f"\n\nThese low score substructures must be strictly prohibited to avoid generating undesirable molecules.\n\n"
        self.limitation = f"# Limitation: \n\n- 1. Ensure sufficient diversity among new molecules and their parents.\n- 2. **Strictly prohibited to  generate the same mols which has generated**.\n"
        self.requirements = """# Output formation: \n\nEach molecule is a separate block with the following fixed format (strictly adhered to for ease of regular expression parsing): {<<<Explaination>>>: $EXPLANATION, Line1<NL>Line2 <<<Molecule>>>: \\box{$Molecule}}.\n Here are the requirements:\n
        \n- 1. $EXPLANATION should be your analysis.\n- 2. The $Molecule should be the smiles of your propsosed molecule.\n- 3. The molecule should be valid. \n\n# Key Point: \n\n- 1. Answer as short as you can. \n- 2. You must avoid low socre substructures. \n- 3. Enforce checks for ring closure and valence; regenerate if noncompliant. \n- 4. Ensure sufficient diversity among new molecules and maintain distinct differences from parent molecules, make sure that every new molecule has at least two differences from parent molecules. \n- 5. After generation, we will manually calculate the scores of all results; molecules that do not meet the requirements will be discarded. \n
        """
        self.task=None

    def edit(self, parent_mol, parent_scores, mutation_rate, h_fragments=None, l_fragments=None,  past_generation=None, past_generation_total=None, action=2):
        task = self.task
        print(f"task:{task}")
        task_definition = self.task2description[task[0]]
        task_objective = self.task2objective[task[0]]
        
        # print(f"task_definition:{task_definition}")
        # print(f"task_objective:{task_objective}")
        print(f"past_generation_comment:", " ".join(past_generation_total))
        h_string = "\n".join(h_fragments)
        l_string = "\n".join(l_fragments)
        try:
            mol_tuple = ''
            for i in range(2):
                
                tu = '\n[' + Chem.MolToSmiles(parent_mol[i]) + ' is one of parent molecula, ' + str(parent_scores[i]) + ' is this parent score.]'
                
                mol_tuple = mol_tuple + tu
            if action==0:
                prompt = task_definition + mol_tuple + '\n' + task_objective + self.limitation + self.good_cases1 + h_string + self.good_cases2  + self.requirements + "\nLet's think step by step. Then please give us more non-duplicative options to broaden diversity (Do not generate the same as them). Finally, use the knowledge you have and consider all generation constraints you must satisfy. Please generate five molecules at a time, and return them in strict block format."
            elif action==1:
                prompt = task_definition + mol_tuple + '\n' + task_objective + self.limitation + self.bad_cases1 + l_string + self.bad_cases2 + self.requirements + "\nLet's think step by step. Then please give us more non-duplicative options to broaden diversity (Do not generate the same as them). Finally, use the knowledge you have and consider all generation constraints you must satisfy. Please generate five molecules at a time, and return them in strict block format."
            else:
                prompt = task_definition + mol_tuple + '\n' + task_objective + self.limitation + self.good_cases1 + h_string + self.good_cases2 + self.bad_cases1 + l_string + self.bad_cases2  + self.requirements  + "\nLet's think step by step. Then please give us more non-duplicative options to broaden diversity (Do not generate the same as them). Finally, use the knowledge you have and consider all generation constraints you must satisfy. Please generate five molecules at a time, and return them in strict block format. "            
            _, r = query_LLM(prompt)
            if action == 2:
                print(f"prompt: {prompt}")
            proposed_smiles = re.findall(r'<<<Molecule>>>: \\box{(.*?)}', r)
            if len(proposed_smiles) == 0:
                proposed_smiles = re.findall(r'ox{(.*?)}', r)
            proposed_smiles = [sanitize_smiles(item) for item in proposed_smiles]
            proposed_smiles = [item for item in proposed_smiles if item is not None]
            print('proposed_smiles: ', proposed_smiles)
            assert proposed_smiles != None and len(proposed_smiles) > 0

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