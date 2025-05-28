import argparse
import json
parser = argparse.ArgumentParser(description='argparse testing')
parser.add_argument('--gpu', type=int, default=1, help="哪一个gpu")
parser.add_argument('--batch-size', type=int, default=64, help="bge推理时的batch")
parser.add_argument('--model-type', type=str, default="", help="")
parser.add_argument('--data-type', type=str, default="", help="")

args = parser.parse_args()
gpu = args.gpu
BATCH_SIZE = args.batch_size
model_type, data_type = args.model_type, args.data_type
data_set = "amz"
if data_set == "amz":
    infer_fn = "/mmu_vcg_wjc_hdd/wanglei16/paper/www2025/CoLLM/dataset/amazon_book/{}.json".format(data_type)
    cot_fn = "/mmu_vcg_wjc_hdd/wanglei16/paper/www2025/reco2-v1/data/book2_rlpf_{}.json".format(model_type)
    USER_OUTPUT_PATH = "/mmu_vcg_wjc_hdd/wanglei16/paper/www2025/Open-World-Knowledge-Augmented-Recommendation/data/amz/proc_data/rlpf_{}_bge_{}.hist".format(model_type, data_type)
else:
    infer_fn = "/mmu_vcg_wjc_hdd/wanglei16/paper/www2025/CoLLM/dataset/ml-1m/{}.json".format(data_type)
    cot_fn = "/mmu_vcg_wjc_hdd/wanglei16/paper/www2025/reco2-v1/data/movielen2_rlpf_{}.json".format(model_type)
    USER_OUTPUT_PATH = "/mmu_vcg_wjc_hdd/wanglei16/paper/www2025/Open-World-Knowledge-Augmented-Recommendation/data/ml-1m/proc_data/rlpf_{}_bge_{}.hist".format(model_type, data_type)

def load_cot(cot_fn):
    cot_dict = {}
    f = open(cot_fn, "r")
    for line in f:
        data = json.loads(line)
        if "means dislike:\n\nPlease summarize the user's" in data["question"]: # no history
            assert 0
            cot_dict[data["idx"]] = {}
        else:
            cot_dict[data["idx"]] = {"question": data["question"], "answer": data["answer"]}
    f.close()
    return cot_dict
cot_dict = load_cot(cot_fn)

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


def batch_deal(question_texts, answer_texts, sources):
    question_bge_embedding = get_bge_embeddings(bge_model, question_texts)
    answer_bge_embedding = get_bge_embeddings(bge_model, answer_texts)

    user_write_jsons = []

    for now_source, now_question_emb, now_answer_emb in zip(sources, question_bge_embedding, answer_bge_embedding):
        task_id = now_source
        write_json = {"user_id": task_id[0], "item_id": task_id[1], "sft_input_emb": now_question_emb, "sft_output_emb": now_answer_emb}
        user_write_jsons.append(write_json)
    
    return user_write_jsons


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


if __name__ == '__main__':
    logging.info("开始加载模型")
    bge_model = BGEM3FlagModel('/mmu_vcg_wjc_hdd/duyong/plms/BAAI/bge-m3', use_fp16=True)

    question_texts = []
    answer_texts = []
    sources = []
    f_user_out = open(USER_OUTPUT_PATH, "w", encoding="utf8")
    logging.info("开始读取文件")
    
    data_maps = load_json('../data/{}/proc_data/datamaps.json'.format(data_set))
    user2id = data_maps['user2id']
    item2id = data_maps['item2id']

    merged_data = []
    f = open(infer_fn, "r")
    for line in f:
        data = json.loads(line)
        idx = data["idx"]
        data["question"] = cot_dict[idx]["question"]
        assert data["history"] in cot_dict[idx]["question"]
        data["answer"] = cot_dict[idx]["answer"]
        merged_data.append(data)

    print(len(merged_data))
    cnt = 0
    for line in tqdm.tqdm(merged_data):
        keep = True
        if (str(line["uid"]) not in user2id) or (str(line["iid"]) not in item2id):
            keep = False
        if data_set == "amz":
            for iid in line["his"]:
                if str(iid) not in item2id:
                    keep = False
        if not keep:
            continue
        cnt += 1
        question = line['question']
        answer = line['answer']
        source = [user2id[str(line["uid"])], item2id[str(line["iid"])]]
        
        question_texts.append(question)
        answer_texts.append(answer)
        sources.append(source)

        if len(question_texts) == BATCH_SIZE:
            user_write_jsons = batch_deal(question_texts, answer_texts, sources)
            for val in user_write_jsons:
                f_user_out.write(json.dumps(val, ensure_ascii=False) + "\n")
            question_texts, answer_texts, sources = [], [], []
            f_user_out.flush()

    print(cnt)
    if len(question_texts) > 0:
        user_write_jsons = batch_deal(question_texts, answer_texts, sources)        
        for val in user_write_jsons:
            f_user_out.write(json.dumps(val, ensure_ascii=False) + "\n")
        f_user_out.flush()

    f_user_out.close()

