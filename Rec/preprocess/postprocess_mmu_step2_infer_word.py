# -*- coding:UTF-8 -*-


import pandas
import json
import os
import re
import numpy as np
from collections import defaultdict
from pre_utils import load_json, save_json, save_pickle, GENDER_MAPPING, \
    AGE_MAPPING, OCCUPATION_MAPPING
import nltk
from nltk.corpus import stopwords
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
PROCESSED_DIR = os.path.join(DATA_DIR, DATA_SET_NAME, 'proc_data_merge')
item_prompt_file = PROCESSED_DIR + '/prompt.open.item.mmu.gpt35_output.step2_infer.gpt35_output'
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
statis = load_json(PROCESSED_DIR + '/stat.json')
attribute_id = max(datamap['attribute2id'].values()) + 1
attribute_ft_num = statis['attribute_ft_num'] + 64
err_cnt = 0
stop_words = set(stopwords.words('english'))
print(len(datamap['id2attribute']))
with open(item_prompt_file) as f:
    for line in f.readlines():
        close_label_list = ''
        close_content_list = []
        dd = json.loads(line.strip())
        iid = dd['id']
        item_id = str(datamap['item2id'][iid])
        close_key = dd['label_answer']
        #infer_res = re.findall(r'\[(.*?)\]', dd['label_answer'])
        if 'I\'m sorry' in close_key:
            continue
        if is_json(close_key):
            s = json.loads(close_key)
            if len(s) <= 2 and type(s) == dict:
                s = s[list(s.keys())[0]]
                #print(s)
            label_cnt = 0
            for cur in s:
                label_cnt += 1
                if 'label' in cur and 'content' in cur:
                    label = str(cur['label']).replace('{','').replace('}','').replace('[','').replace(']','').replace(' ','').replace('，','').replace('"','').replace(',','')
                    contents = str(cur['content']).replace('{','').replace('}','').replace('[','').replace(']','').replace('，',',').replace('"','').split(',')
                    for content in contents:
                        content = content.split(' ')
                        for c in content:
                            c = c.replace(' ', '')
                            if c in stop_words or c == '':
                                continue
                            close_content_list.append(c)
                            close_label_list = close_label_list + label +'#'
        else:
            #s = close_key.replace('```', '').replace("\n", "").replace('json', '').split("},")
            s = close_key.replace('```', '').replace("\n", "").replace('json', '')
            s = s.split("},")
            label_cnt = 0
            for cur in s:
                label_cnt += 1
                tmp_close_content_list = []
                cur = cur.replace('{','').replace('}','').replace('[','').replace(']','').replace('，',',').replace('"','').split(':')
                #print(cur)
                if len(cur) < 3:
                    continue
                label = cur[1].split(',')[0]
                contents = cur[-1].split(',')
                close_label_list = close_label_list + label +'#'
                for content in contents:
                    content = content.split(' ')
                    for c in content:
                        c = c.strip('\t').replace(' ', '').lower()
                        if c in stop_words or c == '' or c is None:
                            continue
                        close_content_list.append(c)
        for close_content in close_content_list:
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
statis['attribute_ft_num'] = datamap['attribute_ft_num']
statis['attribute_num'] = len(datamap['id2attribute'])

save_json(item2attribute, ITEM2ATTRIBUTE_PATH)
save_json(datamap, DATAMAP_PATH)
save_json(statis,PROCESSED_DIR + '/stat.json')

