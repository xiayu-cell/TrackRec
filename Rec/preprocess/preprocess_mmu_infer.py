# -*- coding:UTF-8 -*-


import pandas
import json
import os

res = []
DATA_DIR = '../data/'
DATA_SET_NAME = 'amz'
# DATA_SET_NAME = 'ml-1m'
PROCESSED_DIR = os.path.join(DATA_DIR, DATA_SET_NAME, 'proc_data')
item_prompt_file = PROCESSED_DIR + '/prompt.item_refill'
with open(item_prompt_file) as f:
    d = json.load(f)
    print(len(d.keys()))
    query_id = 0
    for k in d.keys():
        dd = {}
        dd['query_id'] = str(query_id)
        dd['id'] = str(k)
        dd['input'] = str(d[k])
        query_id += 1
        res.append(dd)

with open(PROCESSED_DIR + '/prompt.item_refill.mmu', 'w') as f:
    for d in res:
        f.write(json.dumps(d, ensure_ascii=False))
        f.write('\n')
