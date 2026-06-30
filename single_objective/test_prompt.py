# openai_oracle_test.py
import os
from openai import OpenAI
from rdkit import Chem
import tdc  # 你的代码用到 tdc.Oracle，因此要确保安装了 Therapeutics Data Commons
import numpy as np
import yaml
import re
from settings import settings
import math

class Oracle:
    def __init__(self, args=None, mol_buffer={}):
        self.name = None
        self.evaluator = None
        self.task_label = None
        if args is None:
            self.max_oracle_calls = 10000
            self.freq_log = 100
        else:
            self.args = args
            self.max_oracle_calls = args.max_oracle_calls
            self.freq_log = args.freq_log

        self.mol_buffer = mol_buffer
        self.sa_scorer = tdc.Oracle(name = 'SA')
        self.diversity_evaluator = tdc.Evaluator(name = 'Diversity')
        self.last_log = 0

        self.oracle_name=None


    @property
    def budget(self):
        return self.max_oracle_calls

    def assign_evaluator(self, evaluator):
        self.evaluator = evaluator

    def sort_buffer(self):
        self.mol_buffer = dict(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][0], reverse=True))

    def save_result(self, suffix=None):

        if suffix is None:
            output_file_path = os.path.join(self.args.output_dir, 'results.yaml')
        else:
            output_file_path = os.path.join(self.args.output_dir, 'results_' + suffix + '.yaml')

        self.sort_buffer()
        with open(output_file_path, 'w') as f:
            yaml.dump(self.mol_buffer, f, sort_keys=False)

    
    def log_intermediate(self, mols=None, scores=None, finish=False):

        if finish:
            temp_top100 = list(self.mol_buffer.items())[:100]
            smis = [item[0] for item in temp_top100]
            scores = [item[1][0] for item in temp_top100]
            n_calls = self.max_oracle_calls
        else:
            if mols is None and scores is None:
                if len(self.mol_buffer) <= self.max_oracle_calls:
                    # If not spefcified, log current top-100 mols in buffer
                    temp_top100 = list(self.mol_buffer.items())[:100]
                    smis = [item[0] for item in temp_top100]
                    scores = [item[1][0] for item in temp_top100]
                    n_calls = len(self.mol_buffer)
                else:
                    results = list(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][1], reverse=False))[:self.max_oracle_calls]
                    temp_top100 = sorted(results, key=lambda kv: kv[1][0], reverse=True)[:100]
                    smis = [item[0] for item in temp_top100]
                    scores = [item[1][0] for item in temp_top100]
                    n_calls = self.max_oracle_calls
            else:
                # Otherwise, log the input moleucles
                smis = [Chem.MolToSmiles(m) for m in mols]
                n_calls = len(self.mol_buffer)

        # Uncomment this line if want to log top-10 moelucles figures, so as the best_mol key values.
        # temp_top10 = list(self.mol_buffer.items())[:10]

        avg_top1 = np.max(scores)
        avg_top10 = np.mean(sorted(scores, reverse=True)[:10])
        avg_top100 = np.mean(scores)
        avg_sa = np.mean(self.sa_scorer(smis))
        diversity_top100 = self.diversity_evaluator(smis)


        print(f'{n_calls}/{self.max_oracle_calls} | '
                f'avg_top1: {avg_top1:.3f} | '
                f'avg_top10: {avg_top10:.3f} | '
                f'avg_top100: {avg_top100:.3f} | '
                f'avg_sa: {avg_sa:.3f} | '
                f'div: {diversity_top100:.3f}')

        # print({
        #     "avg_top1": avg_top1,
        #     "avg_top10": avg_top10,
        #     "avg_top100": avg_top100,
        #     "auc_top1": top_auc(self.mol_buffer, 1, finish, self.freq_log, self.max_oracle_calls),
        #     "auc_top10": top_auc(self.mol_buffer, 10, finish, self.freq_log, self.max_oracle_calls),
        #     "auc_top100": top_auc(self.mol_buffer, 100, finish, self.freq_log, self.max_oracle_calls),
        #     "avg_sa": avg_sa,
        #     "diversity_top100": diversity_top100,
        #     "n_oracle": n_calls,
        # })



    def __len__(self):
        return len(self.mol_buffer)

    def score_smi(self, smi):
        """
        Function to score one molecule

        Argguments:
            smi: One SMILES string represnets a moelcule.

        Return:
            score: a float represents the property of the molecule.
        """
        if len(self.mol_buffer) > self.max_oracle_calls:
            return 0
        if smi is None:
            return 0
        mol = Chem.MolFromSmiles(smi)
        if mol is None or len(smi) == 0:
            return 0
        else:
            smi = Chem.MolToSmiles(mol)
            if smi in self.mol_buffer:
                pass
            else:
                fitness = float(self.evaluator(smi))
                #print(fitness, type(fitness))
                if math.isnan(fitness):
                    fitness = 0
                if "docking" in self.args.oracles[0]:
                    fitness *= -1

                self.mol_buffer[smi] = [fitness, len(self.mol_buffer)+1]
            return self.mol_buffer[smi][0]

    def __call__(self, smiles_lst):
        """
        Score
        """
        if type(smiles_lst) == list:
            score_list = []
            for smi in smiles_lst:
                score_list.append(self.score_smi(smi))
                if len(self.mol_buffer) % self.freq_log == 0 and len(self.mol_buffer) > self.last_log:
                    self.sort_buffer()
                    self.log_intermediate()
                    self.last_log = len(self.mol_buffer)
                    self.save_result(self.task_label)
        else:  ### a string of SMILES
            score_list = self.score_smi(smiles_lst)
            if len(self.mol_buffer) % self.freq_log == 0 and len(self.mol_buffer) > self.last_log:
                self.sort_buffer()
                self.log_intermediate()
                self.last_log = len(self.mol_buffer)
                self.save_result(self.task_label)
        return score_list

    @property
    def finish(self):
        return len(self.mol_buffer) >= self.max_oracle_calls


# 创建一个简单的 args 类
class Args:
    def __init__(self):
        self.max_obj = ['qed', 'jnk3']
        self.min_obj = ['sa']
        self.output_dir = './results'

# ------------------- OpenAI 调用部分 -------------------
client = OpenAI(api_key='sk-kTWvnPDUYT6uvnvt4b6xKdYQ3qdB69D2FVp1Y1J7Yiapv0EZ', base_url=settings.base_url)
# , base_url=
task_prompt = """
 # Task Desc: 

 I have two molecules and their QED, SA (Synthetic Accessibility), JNK3 (biological activity against the kinase JNK3) scores. Generate 5 diverse candidate molecules each time, non-duplicate, easy for regex parsing.

# Eval Rules:

 Total score = QED + JNK3 + (1 - (SA - 1) / 9).

[N#CC1=CC=NC(NC2=CC=C(Cl)C=C2C(N)=O)=C1 is one of parent molecula, 2.124729268392161 is this parent score. According to the Eval Rules, ['jnk3_current: 0.01', 'qed: 0.8951507213153015', 'sa: 2.2171144209705105;'] are the score details.]
[N#CC1=CC=NC(NC2=CC=C(C(N)=O)C(C#N)=C2)=N1 is one of parent molecula, 2.058051805799902 is this parent score. According to the Eval Rules, ['jnk3_current: 0.0', 'qed: 0.799535802965004', 'sa: 2.090521636628697;'] are the score details.]

# Task Target: 

 I want to maximize QED score, maximize JNK3 score, and minimize SA score. You can either make crossover and mutations based on the given molecules or substructures or just propose five new molecules based on your knowledge.

# Limitation: 

- 1. Ensure sufficient diversity among new molecules and their parents.
- 2. **Strictly prohibited to  generate the same mols which has generated**.


# High-frequency fragments contained in five high-molecular molecules (Available or based on your knowledge)

[The structures below represent molecular fragments, the distribution of surrounding functional groups, and what properties they are beneficial to.]: 

[14*]c1cc(C#N)ccn1; Pyridine ring, nitrile substitution, N heteroatom; QED
[14*]c1cc(Cl)ccn1; Pyridine ring, chloro substitution, N heteroatom; SA
[16*]c1ccncc1Cl; Pyridine ring, chloro substitution, C6 pattern; SA
[16*]c1cncc(C#N)c1; Pyridine ring, nitrile substitution, C6 pattern; QED
[14*]c1ccc(C#N)cn1; Pyridine ring, nitrile substitution, N heteroatom; QED

These high substructures are recommended options for improving molecular behavior.


# High-frequency fragments contained in five low-molecular molecules (Strictly prohibited)

[The structures below represent molecular fragments, the distribution of surrounding functional groups, what properties are harmful, and alternative fragments to consider.]: 

{[5*][NH+]([5*])C; Two sp3 aliphatic nitrogens, each linked to sp3 ring carbons via single bonds in side chains; QED; Neutral amine or amide}
{[5*][NH+]([5*])[5*]; Sp3 aliphatic nitrogen with three single-bonded carbons, mix of ring and non-ring attachments; SA; Tertiary amine without charge}
{[5*][NH+]1CCCC1; Sp3 aliphatic nitrogen in 5-membered ring, linked to non-ring sp3 carbon; SA; Piperidine or morpholine}
{[8*]CC=C; Sp3 aliphatic carbon linked to sp2 nitrogen in ring via single bond; JNK3; Saturated linker to heteroaryl}
{[5*][NH+]1CCCCC1; Sp3 aliphatic nitrogen in 6-membered ring, linked to non-ring sp3 carbon; SA; Piperidine or morpholine}

These low score substructures must be strictly prohibited to avoid generating undesirable molecules.

# Output formation: 

Each molecule is a separate block with the following fixed format (strictly adhered to for ease of regular expression parsing): {<<<Explaination>>>: $EXPLANATION, Line1<NL>Line2 <<<Molecule>>>: \box{$Molecule}}.
 Here are the requirements:

        
- 1. $EXPLANATION should be your analysis.
- 2. The $Molecule should be the smiles of your propsosed molecule.
- 3. The molecule should be valid. 

# Key Point: 

- 1. Answer as short as you can. 
- 2. You must avoid low socre substructures. 
- 3. Enforce checks for ring closure and valence; regenerate if noncompliant. 
- 4. Ensure sufficient diversity among new molecules and maintain distinct differences from parent molecules, make sure that every new molecule has at least two differences from parent molecules. 
- 5. After generation, we will manually calculate the scores of all results; molecules that do not meet the requirements will be discarded. 

        
Let's think step by step. Then please give us more non-duplicative options to broaden diversity (Do not generate the same as them). Finally, use the knowledge you have and consider all generation constraints you must satisfy. Please generate five molecules at a time, and return them in strict block format. 
"""

task_prompt2 = """
# Task Desc: 

I have two molecules and their QED, SA (Synthetic Accessibility), JNK3 (biological activity against the kinase JNK3) scores. Generate 5 diverse candidate molecules each time, non-duplicate, easy for regex parsing.

# Eval Rules:

Total score = QED + JNK3 + (1 - (SA - 1) / 9).

[N#CC1=CC=NC(NC2=CC=C(Cl)C=C2C(N)=O)=C1 is one of parent molecula, 2.124729268392161 is this parent score. According to the Eval Rules, ['jnk3_current: 0.01', 'qed: 0.8951507213153015', 'sa: 2.2171144209705105;'] are the score details.]
[N#CC1=CC=NC(NC2=CC=C(C(N)=O)C(C#N)=C2)=N1 is one of parent molecula, 2.058051805799902 is this parent score. According to the Eval Rules, ['jnk3_current: 0.0', 'qed: 0.799535802965004', 'sa: 2.090521636628697;'] are the score details.]

# Task Target: 

I want to maximize QED score, maximize JNK3 score, and minimize SA score. You can either make crossover and mutations based on the given molecules or substructures or just propose five new molecules based on your knowledge.

🔧 **Generation Strategy (IMPORTANT):**
- First, prioritize structural exploration and diversity to propose chemically reasonable candidate molecules.
- Then, apply the following fragment constraints and validity checks to refine or regenerate candidates if needed.
- Do not overly bias toward minimal mutations; moderate scaffold or substituent changes are encouraged if chemically valid.

# High-frequency fragments contained in five high-molecular molecules (Available or based on your knowledge)

[The structures below represent molecular fragments, the distribution of surrounding functional groups, and what properties they are beneficial to.]: 

[14*]c1cc(C#N)ccn1; Pyridine ring, nitrile substitution, N heteroatom; QED
[14*]c1cc(Cl)ccn1; Pyridine ring, chloro substitution, N heteroatom; SA
[16*]c1ccncc1Cl; Pyridine ring, chloro substitution, C6 pattern; SA
[16*]c1cncc(C#N)c1; Pyridine ring, nitrile substitution, C6 pattern; QED
[14*]c1ccc(C#N)cn1; Pyridine ring, nitrile substitution, N heteroatom; QED

These high substructures are recommended options for improving molecular behavior.


# High-frequency fragments contained in five low-molecular molecules (Strictly prohibited)

[The structures below represent molecular fragments, the distribution of surrounding functional groups, what properties are harmful, and alternative fragments to consider.]: 

{[5*][NH+]([5*])C; Two sp3 aliphatic nitrogens, each linked to sp3 ring carbons via single bonds in side chains; QED; Neutral amine or amide}
{[5*][NH+]([5*])[5*]; Sp3 aliphatic nitrogen with three single-bonded carbons, mix of ring and non-ring attachments; SA; Tertiary amine without charge}
{[5*][NH+]1CCCC1; Sp3 aliphatic nitrogen in 5-membered ring, linked to non-ring sp3 carbon; SA; Piperidine or morpholine}
{[8*]CC=C; Sp3 aliphatic carbon linked to sp2 nitrogen in ring via single bond; JNK3; Saturated linker to heteroaryl}
{[5*][NH+]1CCCCC1; Sp3 aliphatic nitrogen in 6-membered ring, linked to non-ring sp3 carbon; SA; Piperidine or morpholine}

These low score substructures must be strictly prohibited to avoid generating undesirable molecules.

# Output formation: 

Each molecule is a separate block with the following fixed format (strictly adhered to for ease of regular expression parsing): {<<<Explaination>>>: $EXPLANATION, Line1<NL>Line2 <<<Molecule>>>: \box{$Molecule}}.
 Here are the requirements:

        
- 1. $EXPLANATION should be your analysis.
- 2. The $Molecule should be the smiles of your propsosed molecule.
- 3. The molecule should be valid. 

# Key Point:
- Keep answers concise. 
- Strictly avoid low-score substructures. 
- Ensure valid SMILES (ring closure and valence). 
- Each molecule must differ from parents by at least two features and satisfy all requirements, otherwise it will be discarded.

Let's think step by step. Please generate five molecules at a time and return them in strict block format.
"""
base_prompt2 = """
 # Task Desc: 

 I have two molecules and their QED, SA (Synthetic Accessibility), JNK3 (biological activity against the kinase JNK3) scores. Generate 5 diverse candidate molecules each time, non-duplicate, easy for regex parsing.

# Eval Rules:

 Total score = QED + JNK3 + (1 - (SA - 1) / 9).

[N#CC1=CC=NC(NC2=CC=C(Cl)C=C2C(N)=O)=C1 is one of parent molecula, 2.124729268392161 is this parent score. According to the Eval Rules, ['jnk3_current: 0.01', 'qed: 0.8951507213153015', 'sa: 2.2171144209705105;'] are the score details.]
[N#CC1=CC=NC(NC2=CC=C(C(N)=O)C(C#N)=C2)=N1 is one of parent molecula, 2.058051805799902 is this parent score. According to the Eval Rules, ['jnk3_current: 0.0', 'qed: 0.799535802965004', 'sa: 2.090521636628697;'] are the score details.]

# Task Target: 

 I want to maximize QED score, maximize JNK3 score, and minimize SA score. You can either make crossover and mutations based on the given molecules or substructures or just propose five new molecules based on your knowledge.

# Output formation: \n\nYour output should follow the format:: {<<<Explaination>>>: $EXPLANATION, <<<Molecule>>>: \\box{$Molecule}}.\n Here are the requirements:
        \n- 1. $EXPLANATION should be your analysis.\n- 2. The $Molecule should be the smiles of your propsosed molecule.\n- 3. The molecule should be valid.

"""

base_prompt = """
I have two molecules and their QED, SA (Synthetic Accessibility), GSK3$\beta$ (biological activity against Glycogen Synthase Kinase 3 Beta) scores.

[N#CC1=CC=NC(NC2=CC=C(Cl)C=C2C(N)=O)=C1]
[N#CC1=CC=NC(NC2=CC=C(C(N)=O)C(C#N)=C2)=N1]

I want to maximize QED score, maximize GSK3$\beta$ score, and minimize SA score. Please propose a new molecule better than the current molecules. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.\n\n

\n\nYour output should follow the format: {<<<Explaination>>>: $EXPLANATION, <<<Molecule>>>: \\box{$Molecule}}. Here are the requirements:\n
\n\n1. $EXPLANATION should be your analysis.\n2. The $Molecule should be the smiles of your propsosed molecule.\n3. The molecule should be valid.

"""
# - 按相同的评估规则计算所有结果的得分；不满足要求的分子要舍弃。

def generate_and_score(oracle, client, model_name, input_data, loops=10, **kwargs):
    """
    通用分子生成与评分函数
    :param oracle: Oracle2 实例
    :param client: gpt client
    :param model_name: 模型名称，例如 "gpt-4o"
    :param input_data: 输入数据，可以是字符串或列表
    :param loops: 循环次数
    :param kwargs: 额外传入 client.responses.create 的参数
    :return: None
    """
    all_smiles = []
    results = []

    for _ in range(loops):
        # 第一次尝试
        response = client.responses.create(model=model_name, input=input_data, **kwargs)
        smiles_list = extract_smiles(response.output_text)

        # 如果没有匹配到，再尝试一次
        if not smiles_list:
            response = client.responses.create(model=model_name, input=input_data, **kwargs)
            smiles_list = extract_smiles(response.output_text)

        print(f"[{model_name}] 本轮生成分子：", smiles_list)

        # 评分
        for smi in smiles_list:
            total_score, detail = oracle.score_smi(smi)
            all_smiles.append(smi)
            results.append((smi, total_score, detail))

    # 输出统计
    report_results(all_smiles, results)


def extract_smiles(output_text):
    """提取 SMILES 列表"""
    smiles_list = re.findall(r'<<<Molecule>>>: \\box{(.*?)}', output_text)
    if not smiles_list:
        smiles_list = re.findall(r'ox{(.*?)}', output_text)
    return smiles_list


def report_results(all_smiles, results):
    """输出统计信息"""
    unique_smiles = set(all_smiles)
    unique_ratio = len(unique_smiles) / len(all_smiles) if all_smiles else 0

    print(f"总生成分子数: {len(all_smiles)}")
    print(f"不重复分子数: {len(unique_smiles)}")
    print(f"多样性比例: {unique_ratio:.2f}")
    print("=" * 50)

    # 最高分 TOP5
    top5 = sorted(results, key=lambda x: x[1], reverse=True)[:5]
    print("最高分的 5 个分子：")
    for smi, score, detail in top5:
        print(f"SMILES: {smi}")
        print(f"总评分: {score:.6f}")
        print(f"详情: {detail}")
        print("-" * 40)

    scores = [r[1] for r in results]
    if scores:
        print(f"总分数量: {len(scores)}")
        print(f"平均评分: {np.mean(scores):.6f}")
        print(f"评分方差: {np.var(scores):.6f}")
    print("=" * 100)


def main():
    args = Args()
    oracle = Oracle(args=args)
    oracle.assign_evaluator(args)

    task_prompt_str = f"{task_prompt}"
    input_list = [
        {"role": "system", "content": "You are a great expert of molecula generation."},
        {"role": "user", "content": task_prompt_str},
    ]

    task_prompt_str = f"{task_prompt2}"
    input_list3 = [
        {"role": "system", "content": "You are a great expert of molecula generation."},
        {"role": "user", "content": task_prompt_str},
    ]

    input_list2 = [
        {"role": "system", "content": "You are a great expert of molecula generation."},
        {"role": "user", "content": base_prompt2},
    ]

    base_input_list = [
        {"role": "system", "content": "You are a great expert of molecula generation."},
        {"role": "user", "content": base_prompt},
    ]
    # 三种模型调用
    # generate_and_score(oracle, client, model_name="gpt-4o", input_data=task_prompt_str, temperature=0.7)
    
    # generate_and_score(oracle, client, model_name="gpt-5", input_data=input_list)

    # generate_and_score(oracle, client, model_name="gpt-4o", input_data=base_input_list,
    #                    reasoning={"effort": "low"}, text={"verbosity": "low"}, loops=50)
    # generate_and_score(oracle, client, model_name="gpt-4o", input_data=input_list,
    #                    reasoning={"effort": "low"}, text={"verbosity": "low"})
    

    generate_and_score(oracle, client, model_name="gpt-4", input_data=base_input_list, loops=50)
    generate_and_score(oracle, client, model_name="gpt-4", input_data=input_list)
    # generate_and_score(oracle, client, model_name="gpt-4", input_data=input_list2)
    generate_and_score(oracle, client, model_name="gpt-4", input_data=input_list3)

if __name__ == "__main__":
    main()
