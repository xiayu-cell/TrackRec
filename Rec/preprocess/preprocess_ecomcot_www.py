'''
split train/test by user IDs, train: test= 9: 1
RS history: recent rated 10 items (pos & neg), ID & attributes & rating
LM history: one lm history for each user(max_len=15, item ID, attributes, rating)
attributes include category and brand
rating > 4 as positive, rating <= 4 as negative, no negative sampling
'''
import os
import random
import tqdm
import html
from collections import defaultdict
import numpy as np
import json
from string import ascii_letters, digits, punctuation, whitespace
from pre_utils import set_seed, parse, add_comma, save_json

import difflib


lm_hist_max = 15
train_ratio = 0.9
rating_score = 0.0  # rating score smaller than this score would be deleted
# user 60-core item 40-core
user_core = 60
item_core = 40
attribute_core = 0


def pad_list(lst, target_length = 5, pad_value=''):
    # 计算需要补充的次数
    num_to_pad = target_length - len(lst)
    # 如果num_to_pad是负数或者列表已经达到目标长度，直接返回原列表
    if num_to_pad <= 0:
        return lst
    # 初始化一个包含pad_value的列表
    pad_list = [pad_value] * num_to_pad
    # 拼接原列表和补充列表
    lst.extend(pad_list)
    return lst


def preprocess(data_file, test_file, processed_dir, data_type='Amazon'):

    #datas = ECOM_COT(data_file, rating_score=rating_score)

    # user_items = get_interaction(datas)
    # print(f'{data_file} Raw data has been processed! Lower than {rating_score} are deleted!')
    # # raw_id user: [item1, item2, item3...]

    user_dict = {}
    item_dict = {}
    train_user_items = []
    idx = 0
    two_count = 0
    three_count = 0
    sims_list = [0.0, 0.0, 0.0, 0.0]
    views = 0
    pays = 0

    merged_data = []
    for filename in os.listdir(data_file):
        if filename.startswith('train_') and filename.endswith(".json"):
            file_path = os.path.join(data_file, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                merged_data.extend(data)
    print(len(merged_data))
    for data in merged_data:
        user_profile = data['user_profile']
        user_features = user_profile.split('。')[0].split(',')
        user_features = pad_list(user_features, target_length = 10)
        for i, fea in enumerate(user_features):
            if '，' in fea:
                user_features[i] = user_features[i].split('，')[0]
                break
        user_attrs = user_profile.split('。')[1].split(',')[0:5]
        user_attrs[0] = user_attrs[0].replace('该用户喜欢浏览', '')
        user_attrs = pad_list(user_attrs)
        user_dict[user_profile] = (user_features + user_attrs)

        item_profile = data['ctr_prompt'].split('用户是否会购买<目标商品>：')[1].split("请直接回答")[0]
        item_profile = item_profile.replace("。", "").replace("\n", "")

        item_features = item_profile.split(',')[0].split('-')[0:4]
        item_features = item_features[0].split('的') + item_features[1:]
        item_features = pad_list(item_features)
        item_dict[item_profile] = item_features
        
        label = data['label']
        train_user_items.append((user_profile, item_profile, label))
        idx += 1

    train_idx = idx
    test_user_items = []
    with open(test_file) as f:
        test_data = json.load(f)
        for data in test_data:
            user_profile = data['user_profile']
            user_features = user_profile.split('。')[0].split(',')
            user_features = pad_list(user_features, target_length = 10)
            for i, fea in enumerate(user_features):
                if '，' in fea:
                    user_features[i] = user_features[i].split('，')[0]
                    break
            user_attrs = user_profile.split('。')[1].split(',')[0:5]
            user_attrs[0] = user_attrs[0].replace('该用户喜欢浏览', '')
            user_attrs = pad_list(user_attrs)
            user_dict[user_profile] = (user_features + user_attrs)

            item_profile = data['ctr_prompt'].split('用户是否会购买<目标商品>：')[1].split("请直接回答")[0]
            item_profile = item_profile.replace("。", "").replace("\n", "")

            item_features = item_profile.split(',')[0].split('-')[0:4]
            item_features = item_features[0].split('的') + item_features[1:]
            item_features = pad_list(item_features)
            item_dict[item_profile] = item_features
            
            label = data['label']
            test_user_items.append((user_profile, item_profile, label))

    user2id = {}  # raw 2 uid
    item2id = {}  # raw 2 iid
    id2user = {}  # uid 2 raw
    id2item = {}  # iid 2 raw
    user_id = 1
    item_id = 1
    final_data = {}
    lm_hist_idx = {}
    for user_item in train_user_items:
        user = user_item[0]
        item = user_item[1]
        if user not in user2id:
            user2id[user] = user_id
            id2user[user_id] = user
            user_id += 1
        if item not in item2id:
            item2id[item] = item_id
            id2item[item_id] = item
            item_id += 1
    for user_item in test_user_items:
        user = user_item[0]
        item = user_item[1]
        if user not in user2id:
            user2id[user] = user_id
            id2user[user_id] = user
            user_id += 1
        if item not in item2id:
            item2id[item] = item_id
            id2item[item_id] = item
            item_id += 1

    data_maps = {
        'user2id': user2id,
        'item2id': item2id,
        'id2user': id2user,
        'id2item': id2item,
    }
    attributes = defaultdict(int)
    for iid, info in item_dict.items():
        for cate in info:
            # attributes[cates[1].strip()] += 1
            attributes[cate] += 1
    for uid, info in user_dict.items():
        for cate in info:
            # attributes[cates[1].strip()] += 1
            attributes[cate] += 1
    print(f'before delete, attribute num:{len(attributes)}')
    new_meta = {}
    user_new_meta = {}
    for iid, info in item_dict.items():
        new_meta[iid] = []
        for cate in info:
            new_meta[iid].append(cate)
    for uid, info in user_dict.items():
        user_new_meta[uid] = []
        for cate in info:
            user_new_meta[uid].append(cate)
    # mapping
    attribute2id = {}
    id2attribute = {}
    attributeid2num = defaultdict(int)
    attribute_id = 1
    items2attributes = {}
    users2attributes = {}
    attribute_lens = []

    for iid, attributes in new_meta.items():
        item_id = data_maps['item2id'][iid]
        items2attributes[item_id] = []
        for attribute in attributes:
            if attribute not in attribute2id:
                attribute2id[attribute] = attribute_id
                id2attribute[attribute_id] = attribute
                attribute_id += 1
            attributeid2num[attribute2id[attribute]] += 1
            items2attributes[item_id].append(attribute2id[attribute])
        attribute_lens.append(len(items2attributes[item_id]))

    for uid, attributes in user_new_meta.items():
        user_id = data_maps['user2id'][uid]
        users2attributes[user_id] = []
        for attribute in attributes:
            if attribute not in attribute2id:
                attribute2id[attribute] = attribute_id
                id2attribute[attribute_id] = attribute
                attribute_id += 1
            attributeid2num[attribute2id[attribute]] += 1
            users2attributes[user_id].append(attribute2id[attribute])
        attribute_lens.append(len(users2attributes[user_id]))
    print(f'before delete, attribute num:{len(attribute2id)}')
    print(f'attributes len, Min:{np.min(attribute_lens)}, Max:{np.max(attribute_lens)}, '
            f'Avg.:{np.mean(attribute_lens):.4f}')
    # update datamap
    data_maps['attribute2id'] = attribute2id
    data_maps['id2attribute'] = id2attribute
    data_maps['attributeid2num'] = attributeid2num
    data_maps['attribute_ft_num'] = 15
    data_maps['train_idx'] = train_idx

    # -------------- Save Data ---------------
    save_data_file = processed_dir + '/sequential_data.json'  # interaction sequence between user and item
    test_save_data_file = processed_dir + '/test_sequential_data.json'  # interaction sequence between user and item
    items2attributes_file = processed_dir + '/item2attributes.json'  # item and corresponding attributes
    users2attributes_file = processed_dir + '/user2attributes.json'
    datamaps_file = processed_dir + '/datamaps.json'  # datamap
    '''
    data_maps = {
        'user2id': user2id,
        'item2id': item2id,
        'id2user': id2user,
        'id2item': id2item,
        datamaps['attribute2id'] = attribute2id
        datamaps['id2attribute'] = id2attribute
        datamaps['attributeid2num'] = attributeid2num
    }
    '''
    train_data = []
    test_data = []
    for i in train_user_items:
        train_data.append([user2id[i[0]], item2id[i[1]],i[2]])
    for i in test_user_items:
        test_data.append([user2id[i[0]], item2id[i[1]], i[2]])
    save_json(train_data, save_data_file)
    save_json(test_data, test_save_data_file)
    save_json(items2attributes, items2attributes_file)
    save_json(users2attributes, users2attributes_file)
    save_json(data_maps, datamaps_file)


if __name__ == '__main__':
    set_seed(1234)
    DATA_DIR = '/mmu_nlp_hdd/xiayu12/data/'
    DATA_SET_NAME = 'ecom'
    #DATA_FILE = os.path.join(DATA_DIR, DATA_SET_NAME, 'raw_data', 'ecom_sft_test_20240905_postprocess_res.json')
    DATA_FILE = os.path.join(DATA_DIR, DATA_SET_NAME)
    META_FILE = os.path.join(DATA_DIR, DATA_SET_NAME, 'test.json')
    PROCESSED_DIR = os.path.join('../data/', 'ecom_www', 'proc_data')

    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    preprocess(DATA_FILE, META_FILE, PROCESSED_DIR, data_type='ecom_cot')
