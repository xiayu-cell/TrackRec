import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score,log_loss
import argparse

parser = argparse.ArgumentParser(description='这是一个简单的命令行参数解析示例')

parser.add_argument('--folder_path', type=str, required=True, help='输入文件的路径')
parser.add_argument('--path_prefix', type=str, required=True, help='数据地址前缀的路径')
parser.add_argument('--iter', type=str, required=True, help='迭代次数')

# 解析参数
args = parser.parse_args()

# 读取 JSON 文件
# 定义文件夹路径和输出文件路径
folder_path = args.folder_path
iter = args.iter
# 初始化一个列表来存储合并后的数据
merged_data = []
path_prefix = args.path_prefix

# 遍历文件夹中的文件
for filename in os.listdir(folder_path):
    if filename.startswith(path_prefix) and filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        
        # 读取 JSON 文件
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            merged_data.extend(data)

preds = []
labels = []
scores = []
print(len(merged_data))
total = 0
# for user_id,v in merged_data.items():
#     for item_id, iv in v.items():
#         total +=1
#         ans = iv['ans']
    
#         if ans[0] == 'y' or ans[0] == 'Y':
#             preds.append(1)
#             labels.append(iv['label'])
#         elif ans[0] == 'n' or ans[0] == 'N':
#             preds.append(0)
#             labels.append(iv['label'])
#         else:
#             continue

# for user_id,v in merged_data.items():
#     for item_id, iv in v.items():
#         total +=1
#         ans = iv['ans']
    
#         if ans == "yes":
#             preds.append(1)
#             labels.append(iv['label'])
#         elif ans == "no":
#             preds.append(0)
#             labels.append(iv['label'])
#         else:
#             continue

for item in merged_data:
    total +=1
    ans = item['ans']
    # label = item["label"]
    
    # if label == 1:
    #     labels.append(1)
    # else:
    #     labels.append(0)
    if ans[0] == 'Y':
        preds.append(1)
        labels.append(item["label"])
        scores.append(item['pred'])
        # scores.append(float(item['pred']))
    elif ans[0] == 'N':
        preds.append(0)
        labels.append(item["label"])
        scores.append(item['pred'])
    else:
        print(ans[0])
        # labels.append(iv['label'])
        # scores.append(1-float(iv['score']))
    # else:
    #     continue
print(total)
print(len(preds))
labels = np.array(labels)
preds = np.array(preds)
acc = accuracy_score(labels, preds)
print(len(preds))
print(f"ACC: {acc:.4f}")

# 计算AUC
auc = roc_auc_score(labels, scores)
print(f"AUC: {auc:.4f}")
logloss = log_loss(labels, scores)
print(f"logloss: {logloss:.4f}")
