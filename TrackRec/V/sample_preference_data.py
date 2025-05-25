import json
import random
from tqdm import tqdm
import os

import argparse

parser = argparse.ArgumentParser(description='这是一个简单的命令行参数解析示例')

parser.add_argument('--folder_path', type=str, required=True, help='输入文件的路径')
parser.add_argument('--path_prefix', type=str, required=True, help='数据地址前缀的路径')
parser.add_argument('--sample_dir', type=str, help='保存推理结果文件夹')
parser.add_argument('--iter', type=str, required=True, help='迭代次数')

args = parser.parse_args()

merged_data = []
path_prefix = args.path_prefix
folder_path = args.folder_path
# 遍历文件夹中的文件
for filename in os.listdir(folder_path):
    if filename.startswith(path_prefix) and filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        
        # 读取 JSON 文件
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            merged_data.extend(data)

qid = 0
sample_dir = args.sample_dir
iter = args.iter
with open(f'{sample_dir}/train_sdpo_step_{iter}.json', 'w') as f:
    for i, data in enumerate(tqdm(merged_data)):
        ground_truth = data['label']
        ans = data['ans']
        # 预估全错的直接扔掉
        abandon_flag = True
        if ground_truth == 1:
            for a in ans:
                if 'Yes' or 'yes' in a:
                    abandon_flag = False
                    break
        elif ground_truth == 0:
            for a in ans:
                if 'No' or 'no' in a:
                    abandon_flag = False
                    break
        if abandon_flag:
            continue
        
        response_candidates = data['cot']
        response_rewards = [prob if ground_truth == 1.0 else (1 - prob) for i, prob in enumerate(data['probs'])]
        sorted_responses = [(reward, response) for reward, response in sorted(zip(response_rewards, response_candidates), reverse=True)]
        dd = dict()
        dd['qid'] = qid
        dd['input'] = data['question']
        dd['answer'] = sorted_responses[0][1]
        dd['other_answers'] = [sorted_responses[_][1] for _ in range(1, len(sorted_responses))]
        
        f.write(json.dumps(dd, ensure_ascii=False) + '\n')
        qid += 1
