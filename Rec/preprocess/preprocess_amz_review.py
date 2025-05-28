import json
import os
import pickle
from datetime import date
import random
from collections import defaultdict
import csv
from pre_utils import parse, load_json, save_json, save_pickle, GENDER_MAPPING, \
    AGE_MAPPING, OCCUPATION_MAPPING


def Amazon(data_file, rating_score):
    '''
    reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
    asin - ID of the product, e.g. 0000013714
    reviewerName - name of the reviewer
    helpful - helpfulness rating of the review, e.g. 2/3
    --"helpful": [2, 3],
    reviewText - text of the review
    --"reviewText": "I bought this for my husband who plays the piano. ..."
    overall - rating of the product
    --"overall": 5.0,
    summary - summary of the review
    --"summary": "Heavenly Highway Hymns",
    unixReviewTime - time of the review (unix time)
    --"unixReviewTime": 1252800000,
    reviewTime - time of the review (raw)
    --"reviewTime": "09 13, 2009"
    '''
    datas = {}
    # older Amazon
    # data_file = './raw_data/reviews_' + dataset_name + '.json.gz'
    # latest Amazon
    # data_file = '/home/hui_wang/data/new_Amazon/' + dataset_name + '.json.gz'
    all_num, wo_rating_num = 0, 0
    for inter in parse(data_file):
        all_num += 1
        # print(inter)
        if 'overall' not in inter:
            wo_rating_num += 1
            continue
        if float(inter['overall']) <= rating_score:  # 小于一定分数去掉
            continue
        user = inter['reviewerID']
        item = inter['asin']
        time = inter['unixReviewTime']
        rating = inter['overall']
        if user not in datas:
            datas[user] = {}
        if 'reviewText' in inter.keys():
            review_text = inter['reviewText']
            datas[user][item] = review_text
    print('total review', all_num, 'review w/o rating', wo_rating_num, wo_rating_num / all_num)
    return datas

data_file = os.path.join('../data/', 'amz', 'raw_data', 'Books' + '_5.json.gz')
data = Amazon(data_file, 0.0)
save_data_file = os.path.join('../data/', 'amz', 'raw_data', 'amz_reviews.json')
save_json(data, save_data_file)