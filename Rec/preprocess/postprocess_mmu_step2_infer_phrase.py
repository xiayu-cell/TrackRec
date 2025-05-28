# -*- coding:UTF-8 -*-


import pandas
import json
import os
import re
import numpy as np
from collections import defaultdict
from pre_utils import load_json, save_json, save_pickle, GENDER_MAPPING, \
    AGE_MAPPING, OCCUPATION_MAPPING
import ast
#nltk.download('stopwords')


def is_json(data):
    try:
        json.loads(data)
    except Exception as e:
        return False
    return True

res = []
DATA_DIR = '../data/'
#DATA_SET_NAME = 'amz'
DATA_SET_NAME = 'ml-1m'
PROCESSED_DIR = os.path.join(DATA_DIR, DATA_SET_NAME, 'proc_data')
item_prompt_file = PROCESSED_DIR + '/prompt.item.mmu.gpt35_output.step2_infer.gpt35_output'
res = []
count = 0
items2attributes = {}
attribute_lens = []
attribute2id = {}
id2attribute = {}
attributeid2num = defaultdict(int)
DATAMAP_PATH = os.path.join(PROCESSED_DIR, 'datamaps.json')
ITEM2ATTRIBUTE_PATH = os.path.join(PROCESSED_DIR, 'item2attributes.json')
datamap = load_json(DATAMAP_PATH)
item2attribute = load_json(ITEM2ATTRIBUTE_PATH)
attribute_id = max(datamap['attribute2id'].values()) + 1
attribute_ft_num = 40
err_cnt = 0
print(len(datamap['id2attribute']))
with open(item_prompt_file) as f:
    for line in f.readlines():
        close_label_list,close_content_list = '',''
        dd = json.loads(line.strip())
        iid = dd['id']
        item_id = str(datamap['item2id'][iid])
        close_key = dd['label_answer']
        #infer_res = re.findall(r'\[(.*?)\]', dd['label_answer'])
        if 'I\'m sorry' in close_key:
            continue
        if is_json(close_key):
            s = json.loads(close_key)
            for cur in s:
                if 'label' in cur and 'content' in cur:
                    label = str(cur['label']).replace('{','').replace('}','').replace('[','').replace(']','').replace(' ','').replace('，','').replace('"','').replace(',','')
                    content = str(cur['content']).replace('{','').replace('}','').replace('[','').replace(']','').replace(' ','').replace('，',',').replace('"','')
                    close_label_list = close_label_list + label +'#'
                    close_content_list = close_content_list + content +'#'
        else:
            #s = close_key.replace('```', '').replace("\n", "").replace('json', '').split("},")
            s = close_key.replace('```', '').replace("\n", "").replace('json', '').replace('  ', '')
            s = s.split("},")
            for cur in s:
                cur = cur.replace('{','').replace('}','').replace('[','').replace(']','').replace(' ','').replace('，',',').replace('"','').split(':')
                if len(cur) < 3:
                    continue
                label = cur[1].split(',')[0]
                content = cur[-1]
                close_label_list = close_label_list + label +'#'
                close_content_list = close_content_list + content +','
        for close_content in close_content_list.split(','):
            if close_content not in datamap['attribute2id']:
                #print(attribute_id)
                datamap['attribute2id'][close_content] = attribute_id
                datamap['id2attribute'][attribute_id] = close_content
                attribute_id += 1
            if str(datamap['attribute2id'][close_content]) not in datamap['attributeid2num']:
                datamap['attributeid2num'][str(datamap['attribute2id'][close_content])] = 0
            datamap['attributeid2num'][str(datamap['attribute2id'][close_content])] += 1
            item2attribute[item_id].append(datamap['attribute2id'][close_content])
        attribute_lens.append(len(item2attribute[item_id]))
print(f'attributes len, Min:{np.min(attribute_lens)}, Max:{np.max(attribute_lens)}, 'f'Avg.:{np.mean(attribute_lens):.4f}')

for item in item2attribute.keys():
    if len(item2attribute[item]) < attribute_ft_num:
        item2attribute[item].extend([0] * (attribute_ft_num - len(item2attribute[item])))
    else:
        item2attribute[item] = item2attribute[item][0:attribute_ft_num]
#print(f'before delete, attribute num:{len(attribute2id)}')

datamap['attribute_ft_num'] = attribute_ft_num
statis = load_json(PROCESSED_DIR + '/stat.json')
statis['attribute_ft_num'] = datamap['attribute_ft_num']
statis['attribute_num'] = len(datamap['id2attribute'])

save_json(item2attribute, ITEM2ATTRIBUTE_PATH)
save_json(datamap, DATAMAP_PATH)
save_json(statis,PROCESSED_DIR + '/stat.json')