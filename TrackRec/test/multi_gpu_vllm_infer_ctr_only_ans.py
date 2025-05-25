import json
import os
import argparse
import math
parser = argparse.ArgumentParser(description='这是一个简单的命令行参数解析示例')

parser.add_argument('--folder_path', type=str, required=True, help='输入文件的路径')
parser.add_argument('--path_prefix', type=str, required=True, help='数据地址前缀的路径')
parser.add_argument('--save_infer_n_dir', type=str, help='保存推理结果文件夹')
parser.add_argument('--iter', type=str, required=True, help='迭代次数')
parser.add_argument('--model_path', type=str, required=True, help='model地址')

# 解析参数
args = parser.parse_args()

# 读取 JSON 文件
# 定义文件夹路径和输出文件路径
folder_path = args.folder_path
model_path = args.model_path
save_infer_n_dir = args.save_infer_n_dir
iter = args.iter
# 初始化一个列表来存储合并后的数据
merged_data = []
path_prefix = args.path_prefix

# 遍历文件夹中的文件
for filename in os.listdir(folder_path):
    if filename.startswith(path_prefix) and filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        
        # 读取 JSON 文件
        with open(file_path, 'r') as file:
            data = json.load(file)
            merged_data.extend(data)

data = merged_data

# for index, (key, value) in enumerate(data.items()):
#     for i in range(len(value)):
#         data_list.append({'id':key,'seq_id':value[i][0],'question':value[i][1],'label':value[i][2]})
# test
# data_list = data_list[0:8]

# 计算每份数据的大小
total_prompts = len(data)

# gpus = 4
import torch
ng = torch.cuda.device_count()
batch_size = total_prompts // ng

# 分割数据
batches = [data[i:i + batch_size] for i in range(0, total_prompts, batch_size)]

# 如果数据不能均匀分割，最后一组可能会有剩余的数据
if len(batches) > ng:
    batches[ng-1].extend(batches[ng])
    batches = batches[:ng]

import multiprocessing
from vllm import LLM, SamplingParams
# chat_template = '<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n'
from tqdm import tqdm
from transformers import AutoTokenizer
import torch
norm_qwen_prompt = "You are a helpful assistant."
short_qwen_prompt = "You are a helpful assistant. You will provide very concise and helpful response."

def run_inference(model_path,batch, bs, thread_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{thread_id}'
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False,trust_remote_code=True)
    # 初始化 VLLM 模型，指定 GPU
    sampling_params = SamplingParams(temperature=0.0, repetition_penalty=1.05, max_tokens=10,logprobs=20)

    # Input the model name or path. Can be GPTQ or AWQ models.
    llm = LLM(model=model_path,trust_remote_code=True,max_logprobs=20)
    
    # 将 batch 中的 prompt 提取出来
    # prompts = [chat_template.format(item['question']) for item in batch]
    prompts = []
    questions = []
    for item in batch:
        ctr = item['ctr']
        ans = item['ans']
        qq = '''And we also provide the summary (between <summary> and </summary>) of user's movie watch preference for your reference.
<summary>{}</summary>
Please considering the user's preferences, historical viewing records to predict. Only output the result (Yes or No)'''.format(ans)
        pp = ctr + " "+qq
        # pp = ctr
        messages = [
                {"role": "system", "content": norm_qwen_prompt},
                {"role": "user", "content": pp}
            ]

        text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        questions.append(pp)
        prompts.append(text)
    labels = [item['label'] for item in batch]

    # 保存结果到文件
    generated_texts = []
    # batch_sample = []

    with open(f'{save_infer_n_dir}/model_B_iter_{iter}_{thread_id}.json', 'w') as f:
        for i in tqdm(range(0,len(prompts),bs)):
            msg = prompts[i:i+bs]
            ques = questions[i:i+bs]
            label = labels[i:i+bs]
            # 进行批量推理
            outputs = llm.generate(msg, sampling_params)
    
            # 处理输出结果
            for q,l,output in zip(ques,label, outputs):
                generated_text = output.outputs[0].text
                logprobs = output.outputs[0].logprobs
                target_p = logprobs[0]
                probs = []
                token_ids = output.outputs[0].token_ids
                yes_prob , no_prob = 1e-9,1e-9
                for i in range(len(logprobs)):
                    logprobs_record = logprobs[i]
                    if token_ids[i] in (9454, 2753):
                        if 9454 in logprobs_record:
                            yes_prob = math.exp(logprobs_record[9454].logprob)
                        else:
                            print("error: yes_id is missing!")
                        if 2753 in logprobs_record:
                            no_prob = math.exp(logprobs_record[2753].logprob)
                        else:
                            print("error: no_id is missing!")
                        break
                logit = math.exp(yes_prob)/(math.exp(yes_prob)+math.exp(no_prob))
                # logit = yes_prob/(yes_prob+no_prob)

                # for k,v in target_p.items():
                #     if  k == 9454:
                #         p_yes = v.logprob
                #     if k == 2753:
                #         p_no = v.logprob
                # if p_yes is None:
                #     probs = torch.tensor([p_no])
                #     probs = torch.exp(probs).numpy().astype(float)
                #     probs = 1-probs
                # elif p_no is None:
                #     probs = torch.tensor([p_yes])
                #     probs = torch.exp(probs).numpy().astype(float)
                # else:
                #     probs = torch.tensor([p_yes,p_no])
                #     probs = torch.exp(probs)
                #     probs = torch.softmax(probs, dim=0).numpy().astype(float)
                generated_texts.append({'question':q,'label':l,'ans':generated_text,'pred':logit})

                # if i not in generated_texts:
                #     generated_texts[str(i)] = {}
                #     generated_texts[str(i)][str(si)] = {'question':q,'label':l,'ans':generated_text,'pred':target,'score':probs[0]}
                # else:
                #     if str(si) not in generated_texts[str(i)]:
                #         generated_texts[str(i)][str(si)] = {'question':q,'label':l,'ans':generated_text,'pred':target,'score':probs[0]}

                # generated_texts.append({'id':i,'seq_id':si,'question':q,'label':l,'ans':generated_text})
        
        json.dump(generated_texts,f,ensure_ascii=False,indent=4)
            # break

if __name__ == "__main__":
    processes = []
    # model_path = '/share/ad/xiayu12/TALLRec/my_paper/self_play/model_B/best_B/output/s_1/checkpoint-120'
    for i in range(ng):
        p = multiprocessing.Process(target=run_inference, args=(model_path,batches[i], 50, i))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()