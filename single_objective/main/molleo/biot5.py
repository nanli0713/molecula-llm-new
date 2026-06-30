from transformers import T5Tokenizer, T5ForConditionalGeneration
import selfies as sf
from rdkit import Chem
device = 'cuda:0'
class BioT5:
    def __init__(self):

        self.tokenizer = T5Tokenizer.from_pretrained("QizhiPei/biot5-base-text2mol")
        self.model = T5ForConditionalGeneration.from_pretrained('QizhiPei/biot5-base-text2mol').to(device)

        self.task2description = {
                'qed': 'Definition: You are given a molecule SELFIES. Your job is to generate a SELFIES molecule that looks more like a drug.\n\n',
                'jnk3': 'Definition: You are given a molecule SELFIES. Your job is to generate a SELFIES molecule that is a greater inhibitor of JNK3.\n\n',
                'drd2': 'Definition: You are given a molecule SELFIES. Your job is to generate a SELFIES molecule that inhibits DRD2 more.\n\n',
                'gsk3b': 'Definition: You are given a molecule SELFIES. Your job is to generate a SELFIES molecule that inhibits GSK3B more.\n\n',
                'isomers_c9h10n2o2pf2cl': 'Definition: You are given a molecule SELFIES. Your job is to generate a SELFIES molecule that has the formula C9H10N2O2PF2Cl.\n\n',
                'perindopril_mpo': 'Definition: You are given a molecule SELFIES. Your job is to generate a SELFIES molecule that looks more like Perindopril.\n\n',
                'sitagliptin_mpo': 'Definition: You are given a molecule SELFIES. Your job is to generate a SELFIES molecule that looks more like Sitagliptin.\n\n',
                'ranolazine_mpo': 'Definition: You are given a molecule SELFIES. Your job is to generate a SELFIES molecule that looks more like Ranolazine.\n\n',
                'thiothixene_rediscovery': 'Definition: You are given a molecule SELFIES. Your job is to generate a SELFIES molecule that looks more like Thiothixene.\n\n',
                'mestranol_similarity': 'Definition: You are given a molecule SELFIES. Your job is to generate a SELFIES molecule that looks more like Mestranol.\n\n',
                'deco_hop': 'Definition: You are given a molecule SELFIES. Your job is to generate a SELFIES molecule that maintains the core scaffold but replaces decorative groups (R-groups) with novel substituents.\n\n',
                'scaffold_hop': 'Definition: You are given a molecule SELFIES. Your job is to generate a SELFIES molecule that preserves biological activity while altering the central molecular scaffold/core structure.\n\n',
                'Valsartan_SMARTS': 'Definition: You are given a molecule SELFIES. Your job is to generate a SELFIES molecule that matches the critical SMARTS substructure pattern of Valsartan (a biphenyl tetrazole derivative).\n\n',
                'isomers_c7h8n2o2': 'Definition: You are given a molecule SELFIES. Your job is to generate a SELFIES molecule that has the exact molecular formula C7H8N2O2.\n\n',
                'albuterol_similarity': 'Definition: You are given a molecule SELFIES. Your job is to generate a SELFIES molecule that has high structural similarity to Albuterol (salbutamol), particularly preserving its beta-hydroxy amine pharmacophore.\n\n',
                'celecoxib_rediscovery': 'Definition: You are given a molecule SELFIES. Your job is to generate a SELFIES molecule that exactly matches or closely resembles the COX-2 inhibitor Celecoxib.\n\n',
                'troglitazone_rediscovery': 'Definition: You are given a molecule SELFIES. Your job is to generate a SELFIES molecule that exactly matches or closely resembles the PPARγ agonist Troglitazone.\n\n',
                'amlodipine_mpo': 'Definition: You are given a molecule SELFIES. Your job is to generate a SELFIES molecule optimized for multiple properties characteristic of Amlodipine (calcium channel blocker), including logP, solubility, and target affinity.\n\n',
                'fexofenadine_mpo': 'Definition: You are given a molecule SELFIES. Your job is to generate a SELFIES molecule optimized for multiple properties characteristic of Fexofenadine (antihistamine), including polarity, metabolic stability, and hERG safety profile.\n\n',
                'osimertinib_mpo': 'Definition: You are given a molecule SELFIES. Your job is to generate a SELFIES molecule optimized for multiple properties characteristic of Osimertinib (EGFR inhibitor), including kinase selectivity, mutant specificity, and pharmacokinetics.\n\n',
                'zaleplon_mpo': 'Definition: You are given a molecule SELFIES. Your job is to generate a SELFIES molecule optimized for multiple properties characteristic of Zaleplon (sedative-hypnotic), including GABA_A receptor affinity, rapid onset, and short half-life.\n\n',
                'median1': 'Definition: You are given a molecule SELFIES. Your job is to generate a SELFIES molecule that achieves median property values across key drug-likeness metrics (e.g., logP, molecular weight, HBD/HBA counts) from a reference dataset.\n\n',
                'median2': 'Definition: You are given a molecule SELFIES. Your job is to generate a SELFIES molecule that achieves median property values across pharmacokinetic parameters (e.g., solubility, permeability, metabolic stability) from a reference dataset.\n\n'
                }
        self.good_cases1 = f"\n\n# Examples:\n\n"
        self.good_cases2 = f"\n\nThese high substructures are recommended options for improving molecular behavior.\n"

        self.bad_cases1 = (
            "\n\n# Forbidden molecular substructures (Negative Examples)\n\n"
            "The following substructures are associated with poor activity, toxicity, "
            "or low drug-likeness. Generated molecules SHOULD NOT contain any of these fragments.\n\n"
        )
        self.bad_cases2 = (
            "\n\nThese low score substructures must be strictly prohibited to avoid generating undesirable molecules.\n\n"
        )

        self.task=None

    def edit(self, smiles_list, h_fragments=None, l_fragments=None,  past_generation=None, past_generation_total=None, action=2):
        self.task = self.task
        task = self.task
        task_definition = self.task2description[task[0]]

        editted_molecules = []
        for i, MOL in enumerate(smiles_list):
            SMILES = Chem.MolToSmiles(MOL)

            print("===== for SMILES {} =====".format(SMILES))
            try:
                selfies_input = sf.encoder(SMILES)
            except:
                print("could not encode input smiles", SMILES)
                editted_molecules.append(None)
                continue
            task_input = f'Now complete the following example -\nInput: <bom>{selfies_input}<eom>\nOutput: '
            if h_fragments and action == 0:
                task_input += self.good_cases1 + h_fragments + self.good_cases2
            elif l_fragments and action == 1:
                task_input += self.bad_cases1 + l_fragments + self.bad_cases2
            elif h_fragments and l_fragments is not None and action == 2:
                task_input += self.good_cases1 + h_fragments + self.good_cases2 + self.bad_cases1 + l_fragments + self.bad_cases2
            else:
                task_input += ""
            model_input = task_definition + task_input
            if i == 0:
                print("sample model input", model_input)
            input_ids = self.tokenizer(model_input, return_tensors="pt").input_ids.to(device)
            
            generation_config = self.model.generation_config
            generation_config.max_length = 512
            generation_config.num_beams = 5
            generation_config.num_return_sequences = 5

            outputs = self.model.generate(input_ids, generation_config=generation_config).cpu()

            generated_mols = []
            for output in outputs:
                output_selfies = self.tokenizer.decode(output, skip_special_tokens=True).replace(' ', '')
                try:
                    output_smiles = sf.decoder(output_selfies)
                    mol = Chem.MolFromSmiles(output_smiles)
                    if mol:
                        generated_mols.append(mol)
                except:
                    pass

            editted_molecules.append(generated_mols)

        return editted_molecules
if __name__ == "__main__":
    model = BioT5()
    print(model.edit(["CC(O)CC(=O)C(=O)[O-1]"]))
