import torch
import clip
import json
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import normalize
from utils import save_json


DATA_DIR = '../data/'
DATA_SET_NAME = 'amz'
#DATA_SET_NAME = 'ml-1m'
DATA_PATH = os.path.join(DATA_DIR, DATA_SET_NAME, 'proc_data')

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def is_json(data):
    try:
        json.loads(data)
    except Exception as e:
        return False
    return True


def _preprocess_text(text):
    # adapt the text to Chinese BERT vocab
    text = text.strip().lower().replace("“", "\"").replace("”", "\"")
    return text


class MyDataset(Dataset):
    def __init__(self, file_path):
        self.tag_total, self.input_total = [], []
        with open(file_path, 'r') as f_in:
            for idx, line in enumerate(f_in.readlines()):
                data = json.loads(line.strip())
                item_id = data['id']
                answer = data['label_answer']
                close_label_list, close_content_list = '',''
                if 'I\'m sorry' in answer:
                    continue
                if is_json(answer):
                    s = json.loads(answer)
                    for cur in s:
                        if 'label' in cur and 'content' in cur:
                            label = str(cur['label']).replace('{','').replace('}','').replace('[','').replace(']','').replace(' ','').replace('，','').replace('"','').replace(',','')
                            content = str(cur['content']).replace('{','').replace('}','').replace('[','').replace(']','').replace(' ','').replace('，',',').replace('"','')
                            close_label_list = close_label_list + label +'#'
                            close_content_list = close_content_list + content +'#'
                else:
                    #s = close_key.replace('```', '').replace("\n", "").replace('json', '').split("},")
                    s = answer.replace('```', '').replace("\n", "").replace('json', '').replace('  ', '')
                    s = s.split("},")
                    for cur in s:
                        cur = cur.replace('{','').replace('}','').replace('[','').replace(']','').replace('，',',').replace('"','').split(':')
                        if len(cur) < 3:
                            continue
                        label = cur[1].split(',')[0]
                        content = cur[-1]
                        close_label_list = close_label_list + label +'#'
                        close_content_list = close_content_list + content +','
                close_text = close_content_list.split(',')
                self.tag_total += [item_id + '_close'] * len(close_text)
                self.input_total += close_text
        assert len(self.tag_total) == len(self.input_total)
        self.dataset_len = len(self.tag_total)
        print('total samples: ', self.dataset_len)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        tag, raw_text = self.tag_total[index], self.input_total[index]
        raw_text = clip.tokenize([_preprocess_text(raw_text)])[0]
        return tag, raw_text


def get_dataset(db_path, batch_size):
    print("="*50, db_path)
    dataset = MyDataset(db_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=64,
        shuffle=False,
        drop_last=False
    )
    return dataloader, dataset


db_path = os.path.join(DATA_PATH, 'prompt.item.mmu.gpt35_output.step2_infer.gpt35_output')
dataloader, dataset = get_dataset(db_path=db_path, batch_size=16)
tag_dict = {}
close_item_vec_map = {}
open_item_vec_map = {}
with torch.no_grad():
    for idx, (tag_batch, text_batch) in enumerate(tqdm(dataloader)):
        text_batch = text_batch.to(device)
        text_features = model.encode_text(text_batch)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        for i in range(len(tag_batch)):
            tag, emb = tag_batch[i], text_features[i]
            emb = emb.detach().cpu().numpy()
            itemid, mode = tag.split('_')
            assert mode in ['open', 'close']
            if itemid not in tag_dict:
                tag_dict[itemid] = {}
            if mode not in tag_dict[itemid]:
                tag_dict[itemid][mode] = []
            tag_dict[itemid][mode].append(emb)

for itemid, text in tqdm(tag_dict.items()):
    if 'open' in text:
        open_emb = [float(item) for item in np.mean(normalize(np.array(text['open']), axis=1, norm='l2'), axis=0).tolist()]
        open_item_vec_map[itemid] = open_emb
    if 'close' in text:
        close_emb = [float(item) for item in np.mean(normalize(np.array(text['close']), axis=1, norm='l2'), axis=0).tolist()]
        close_item_vec_map[itemid] = close_emb

save_json(close_item_vec_map, os.path.join(DATA_PATH, '{}_{}_augment.item'.format('clip', 'mean')))
stat_path = os.path.join(DATA_PATH, 'stat.json')
with open(stat_path, 'r') as f:
    stat = json.load(f)

stat['dense_dim'] = 512
with open(stat_path, 'w') as f:
        stat = json.dumps(stat)
        f.write(stat)