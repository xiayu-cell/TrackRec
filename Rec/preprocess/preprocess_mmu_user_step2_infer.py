# -*- coding:UTF-8 -*-


import pandas
import json
import os

res = []
DATA_DIR = '../data/'
#DATA_SET_NAME = 'amz'
DATA_SET_NAME = 'ml-1m'
PROCESSED_DIR = os.path.join(DATA_DIR, DATA_SET_NAME, 'proc_data')
item_prompt_file = PROCESSED_DIR + '/prompt.open.hist.mmu.gpt35_output'
prompt = 'Your task now is to meticulously analyze the user\'s inclination preferences for real knowledge and recommendation reasons, from which you can extract keywords and key information about the user\'s inclination and preferences, please follow the guidelines below to do so:\n\n1. **Extract accurately**: carefully read the analyzed content for the user, accurately identify and extract the key information, and each information should not be more than three words.\n\n2. **Return structured data**: return the extracted information in JSON format, for example:\n``json\n[\n "Key Information Phrase 1",\n "Key Information Phrase 2",\n "Key Information Phrase 3"\n]\n```\n\nNow you are asked to perform key information extraction on the inputs in the following ***, you need to determine which are the important information and output the extracted important information in json format.\nInput:\n***\n@@\n***'
res = []
with open(item_prompt_file) as f:
    for line in f.readlines():
        dd = json.loads(line.strip())
        dd['input'] = prompt.replace('@@', dd['label_answer'])
        dd.pop('label_answer')
        res.append(dd)

with open(item_prompt_file + '.step2_infer', 'w') as f:
    for d in res:
        f.write(json.dumps(d, ensure_ascii=False))
        f.write('\n')
