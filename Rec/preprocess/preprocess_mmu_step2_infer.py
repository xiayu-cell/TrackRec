# -*- coding:UTF-8 -*-


import pandas
import json
import os

res = []
DATA_DIR = '../data/'
DATA_SET_NAME = 'amz'
#DATA_SET_NAME = 'ml-1m'
PROCESSED_DIR = os.path.join(DATA_DIR, DATA_SET_NAME, 'proc_data_new_open')
item_prompt_file = PROCESSED_DIR + '/prompt.open.item.mmu.gpt35_output'
prompt = 'Please help me from the text in *** below into the form of a key phrase sequence and output it in json format. Specific requirements: Only sentences containing colons should be retained. The \"label\" in each group of output is the information label that summarizes the content of the text after each colon, such as \"Quality of content\", \"Author\'s reputation\", \"recommendations and reviews\", \"brand image\", etc. The \"content\" is the information label extracted from the text after each colon, and constitutes a sequence of several keywords, each keyword is no more than 4 words, separated by commas. Refer to the following example:\nInput:\n***\nBased on the provided book title, author\'s brand and categories, we can analyze the following aspects:\n1. Quality of content: William Shakespearei\'s works are renowned for their in-depth exploration of human nature, complex characters, and timeless themes. the content of this book is likely to offer readers new knowledge and insightful ideas about love, power, and societal expectations.\n2. Author\'s reputation: William Shakespeare is a highly respected and influential playwright, with a wealth of experience and recognized authority in literature. William Shakespeare is a highly respected and influential playwright, with a wealth of experience and recognized authority in literature. His other works have been well-received for centuries, making his name associated with quality and depth.\nHis other works have been well-received for centuries, making his name associated with quality and depth.\n***\nOutput:\n[\n{\"label\": \"Quality of content\", \"content\": "exploration of human nature, complex characters, timeless themes, new knowledge, insightful ideas, love, power, and societal expectations"},\n{\"label\": \"Author\'s reputation\", \"content\": \"Highly respected, influential playwright, wealth of experience, recognized authority, well-received works, quality , depth\"}\n]\n\nNow please do key information extraction and keyword slicing on the input in *** below and output all the information in json format with label and content.\nInput:\n***\n@@\n***'
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
