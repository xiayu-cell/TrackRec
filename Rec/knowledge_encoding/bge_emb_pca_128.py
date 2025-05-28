import argparse
parser = argparse.ArgumentParser(description='argparse testing')
parser.add_argument('--gpu', type=int, default=0, help="哪一个gpu")
parser.add_argument('--input-path', type=str, required=True, help="输入文件，jsonl格式，每一行必须包含question、answer、source 3个key")
parser.add_argument('--item-out-path', type=str, required=True, help="item任务输出的文件")
parser.add_argument('--user-out-path', type=str, required=True, help="user任务输出的文件")
parser.add_argument('--pca-path', type=str, default="./pca_model_500w.pkl", help="pca的参数")
parser.add_argument('--batch-size', type=int, default=64, help="bge推理时的batch")

args = parser.parse_args()
gpu = args.gpu
FILE_INPUT_PATH = args.input_path
ITEM_OUTPUT_PATH = args.item_out_path
USER_OUTPUT_PATH = args.user_out_path
PCA_PATH = args.pca_path
BATCH_SIZE = args.batch_size

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

import torch
import tqdm, json
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import numpy as np
from FlagEmbedding import BGEM3FlagModel
import joblib
import logging
logging.basicConfig(format='%(asctime)s  %(levelname)s: %(message)s', level=logging.INFO)


def get_bge_embeddings(model, texts):
    embeddings = model.encode(
        texts, 
        batch_size=BATCH_SIZE, 
        max_length=4096, 
    )['dense_vecs']
    embeddings = embeddings.tolist()
    return embeddings


def get_pca_128(pca, embeddings):
    pca_emb = pca.transform(embeddings).astype(np.float32).tolist()
    return pca_emb


def batch_deal(question_texts, answer_texts, sources, task_type):
    question_bge_embedding = get_bge_embeddings(bge_model, question_texts)
    answer_bge_embedding = get_bge_embeddings(bge_model, answer_texts)

    #question_pca_embedding = get_pca_128(pca_model, question_bge_embedding)
    #answer_pca_embedding = get_pca_128(pca_model, answer_bge_embedding)

    item_write_jsons, user_write_jsons = [], []

    for now_source, now_question_emb, now_answer_emb in zip(sources, question_bge_embedding, answer_bge_embedding):
        task_id = now_source
        if task_type == "item":
            write_json = {"item_id": int(task_id), "sft_input_emb": now_question_emb, "sft_output_emb": now_answer_emb}
            item_write_jsons.append(write_json)
        elif task_type == "user":
            write_json = {"user_id": int(task_id), "sft_input_emb": now_question_emb, "sft_output_emb": now_answer_emb}
            user_write_jsons.append(write_json)
        else:
            raise RuntimeError("任务不对")
    
    return item_write_jsons, user_write_jsons


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)



def process_prompt(prompt):
    search_str = ",常住"
    if search_str in prompt:
        start_index = prompt.index(search_str) + len(search_str)
        first_comma_index = prompt.find('，', start_index)
        if first_comma_index != -1:
            second_comma_index = prompt.find(',', first_comma_index + 1)
            if second_comma_index != -1:
                # Remove the content between the first and second comma after ",常住"
                new_prompt = prompt[:first_comma_index] + prompt[second_comma_index:]
                return new_prompt, True
    return prompt, False


if __name__ == '__main__':
    logging.info("开始加载模型")
    bge_model = BGEM3FlagModel('/mmu_vcg_wjc_hdd/duyong/plms/BAAI/bge-m3', use_fp16=True)
    #pca_model = joblib.load(PCA_PATH)

    question_texts = []
    answer_texts = []
    sources = []
    f_item_out = open(ITEM_OUTPUT_PATH, "w", encoding="utf8")
    f_user_out = open(USER_OUTPUT_PATH, "w", encoding="utf8")
    logging.info("开始读取文件")
    
    data_maps = load_json('../data/ecom_dpo/proc_data/datamaps.json')
    user2id = data_maps['user2id']
    item2id = data_maps['item2id']

    total_count = 0
    with open(FILE_INPUT_PATH, encoding="utf8") as f_in:
        for line in f_in.readlines():
            total_count += 1
    miss_cnt = 0
    with open(FILE_INPUT_PATH, encoding="utf8") as f_in:
        for line in tqdm.tqdm(f_in.readlines(), total=total_count):
            line = json.loads(line)
            question = line['question']
            answer = line['answer_0'].split('。')[0].replace('该用户接下来可能会发生的商品浏览或购买行为是:', '')
            source = line["source"]
            #question = question.replace("你是一名专业的电商广告分析师，精通销售、广告制作技巧，精通营销心理学，精通逻辑论证。请基于下面的用户画像与用户行为信息，帮我预测该用户下一个可能会购买的商品是什么。让我们一步步思考，采用常识和各学科知识。", "")
            #question = question.replace("你是一名专业的电商广告分析师，精通销售、广告制作技巧，精通营销心理学，精通逻辑论证。请基于下面的商品信息与商品浏览与购买记录，帮我预测该商品接下来可能会被什么用户购买。让我们一步步思考，采用常识和各学科知识。", "")
            question = question.replace("你是一名专业的电商广告分析师，精通销售、广告制作技巧，精通营销心理学，精通逻辑论证。请基于下面的用户画像与用户行为信息，帮我预测该用户下一个可能会发生哪些商品浏览与购买行为。让我们一步步思考，采用常识和各学科知识。", "")
            
            task_type = ('user' if 'usertask_' in source else 'item')
            if 'usertask_' in source:
                profile = question.split('用户的基本画像如下：')[1].split('\n')[0]
                profile = question.split("用户的基本画像如下：")[1].split('该用户最近存在以下历史行为：')[0]
                id = user2id[profile]
            else:
                item_profile = question.split('商品的基本信息如下：')[1].split('\n')[0]
                id = item2id[profile]
            #question, _ = process_prompt(question)
            question = question.split("该用户最近存在以下历史行为：")[0]
            # try:
            #     answer = answer.split('。')[-2]
            # except:
            #     print(answer)
            # #print(question)
            
            question_texts.append(question)
            answer_texts.append(answer)
            sources.append(id)

            if len(question_texts) == BATCH_SIZE:
                item_write_jsons, user_write_jsons = batch_deal(question_texts, answer_texts, sources, task_type)
                for val in item_write_jsons:
                    f_item_out.write(json.dumps(val, ensure_ascii=False) + "\n")
                for val in user_write_jsons:
                    f_user_out.write(json.dumps(val, ensure_ascii=False) + "\n")
                question_texts, answer_texts, sources = [], [], []
                f_item_out.flush()
                f_user_out.flush()

    if len(question_texts) > 0:
        item_write_jsons, user_write_jsons = batch_deal(question_texts, answer_texts, sources, task_type)
        for val in item_write_jsons:
            f_item_out.write(json.dumps(val, ensure_ascii=False) + "\n")
        
        for val in user_write_jsons:
            f_user_out.write(json.dumps(val, ensure_ascii=False) + "\n")
        f_item_out.flush()
        f_user_out.flush()

    f_item_out.close()
    f_user_out.close()

