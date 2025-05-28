import json
import os
import pickle
from datetime import date
import random
from collections import defaultdict
import csv
from pre_utils import load_json, save_json, save_pickle, GENDER_MAPPING, \
    AGE_MAPPING, OCCUPATION_MAPPING

rerank_item_from_hist = 4
rerank_hist_len = 10
rerank_list_len = 10
ctr_hist_len = 10


def generate_ctr_data(sequence_data, item_set):
    # print(list(lm_hist_idx.values())[:10])
    full_data = []
    total_label = []
    idx_len = len(item_set)
    for ii, data in enumerate(sequence_data):
        if ii % 3 :
            continue
        pos_ids = data[1:4]
        for i in range(3):
            full_data.append([data[0], data[1 + i], data[4]])
            total_label.append(data[4])
            # neg_ids = []
            # for i in range(5):
            #     random_int = random.randint(0, idx_len - 1)
            #     if item_set[random_int] not in neg_ids and item_set[random_int] not in pos_ids:
            #         neg_ids.append(item_set[random_int])
            # for id in neg_ids:
            #     full_data.append([data[0], id, 0.0])
            #     total_label.append(0.0)
    print('item num', len(item_set), 'data num', len(full_data), 'pos ratio',
            sum(total_label) / len(total_label))
    print(full_data[:150])
    return full_data


if __name__ == '__main__':
    random.seed(12345)
    DATA_DIR = '../data/'
    # DATA_SET_NAME = 'amz'
    #DATA_SET_NAME = 'ecom_cot'
    DATA_SET_NAME = 'ecom_dpo'
    PROCESSED_DIR = os.path.join(DATA_DIR, DATA_SET_NAME, 'proc_data')
    SEQUENCE_PATH = os.path.join(PROCESSED_DIR, 'sequential_data.json')
    TEST_SEQUENCE_PATH = os.path.join(PROCESSED_DIR, 'test_sequential_data.json')
    ITEM2ATTRIBUTE_PATH = os.path.join(PROCESSED_DIR, 'item2attributes.json')
    USER2ATTRIBUTE_PATH = os.path.join(PROCESSED_DIR, 'user2attributes.json')
    DATAMAP_PATH = os.path.join(PROCESSED_DIR, 'datamaps.json')

    sequence_data = load_json(SEQUENCE_PATH)
    test_sequence_data = load_json(TEST_SEQUENCE_PATH)
    item2attribute = load_json(ITEM2ATTRIBUTE_PATH)
    user2attribute = load_json(USER2ATTRIBUTE_PATH)
    item_set = list(map(int, item2attribute.keys()))
    user_set = list(map(int, user2attribute.keys()))
    print('final loading data')
    # print(list(item2attribute.keys())[:10])

    print('generating ctr train dataset')
    train_ctr = generate_ctr_data(sequence_data, item_set)
    print(len(train_ctr))
    print('generating ctr test dataset')
    #test_ctr = generate_ctr_data(test_sequence_data, item_set)
    print('Total user num: {}'.format(len(user_set)))
    extract_ratio = 0.2
    extract_num = int(len(user_set) * extract_ratio)
    test_users = random.sample(user_set, extract_num)
    test_ctr = []
    for data in train_ctr:
        if data[0] in test_users:
            test_ctr.append(data)
    print(len(test_ctr))
    train_ctr = [f for f in train_ctr if f[0] not in test_users]
    print(len(train_ctr))
    print('save ctr data')
    save_pickle(train_ctr, PROCESSED_DIR + '/ctr.train')
    save_pickle(test_ctr, PROCESSED_DIR + '/ctr.test')
    train_ctr, test_ctr = None, None

    datamap = load_json(DATAMAP_PATH)

    statis = {
        'attribute_ft_num': datamap['attribute_ft_num'],
        'item_num': len(datamap['id2item']),
        'user_num': len(datamap['id2user']),
        'attribute_num': len(datamap['id2attribute']),
        'dense_dim': 128,
    }
    save_json(statis, PROCESSED_DIR + '/stat.json')

