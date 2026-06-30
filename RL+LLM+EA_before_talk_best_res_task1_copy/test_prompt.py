# openai_oracle_test.py
import os
from openai import OpenAI
from rdkit import Chem
import tdc  # 你的代码用到 tdc.Oracle，因此要确保安装了 Therapeutics Data Commons
import numpy as np
import yaml
import re
from settings import settings

# 引入 Oracle2 类（直接复制你提供的代码）--------------------
class Oracle2:
    def __init__(self, args=None, mol_buffer={}):
        self.name = None
        self.max_obj = args.max_obj
        self.min_obj = args.min_obj
        self.max_evaluator = None
        self.min_evaluator = None
        self.task_label = None
        if args is None:
            self.max_oracle_calls = 100000
            self.freq_log = 100
        else:
            self.args = args
            self.max_oracle_calls = 1000000
            self.freq_log = 100
        self.mol_buffer = mol_buffer
        self.sa_scorer = tdc.Oracle(name='SA')
        self.diversity_evaluator = tdc.Evaluator(name='Diversity')
        self.last_log = 0

    def assign_evaluator(self, args):
        self.max_evaluator = []
        self.min_evaluator = []
        for idx in range(len(self.max_obj)):
            eva = tdc.Oracle(name=self.max_obj[idx])
            self.max_evaluator.append(eva)
        for idx in range(len(self.min_obj)):
            eva = tdc.Oracle(name=self.min_obj[idx])
            self.min_evaluator.append(eva)

    def evaluate(self, smi):
        score = 0
        score_list = []
        for eva in self.max_evaluator:
            val = eva(smi)
            score += val
            score_list.append(f"{eva.name}: {val}")
        for eva in self.min_evaluator:
            val = eva(smi)
            if eva.name == 'sa':
                score += (1 - ((val - 1) / 9))
                score_list.append(f"{eva.name}: {val};")
            else:
                score += (1 - val)
                score_list.append(f"{eva.name}: {val};")
        return score, score_list

    def score_smi(self, smi):
        if len(self.mol_buffer) > self.max_oracle_calls:
            return 0, []
        if smi is None:
            return 0, []
        mol = Chem.MolFromSmiles(smi)
        if mol is None or len(smi) == 0:
            return 0, []
        else:
            smi = Chem.MolToSmiles(mol)
            if smi not in self.mol_buffer:
                eval_score, eval_score_detail = self.evaluate(smi)
                self.mol_buffer[smi] = [float(eval_score), eval_score_detail, len(self.mol_buffer) + 1]
            return self.mol_buffer[smi][0], self.mol_buffer[smi][1]

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
    oracle = Oracle2(args=args)
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
