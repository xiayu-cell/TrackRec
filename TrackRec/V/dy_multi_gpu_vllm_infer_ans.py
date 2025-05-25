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
print(args.folder_path)
print(args.path_prefix)
print(args.save_infer_n_dir)
print(args.iter)
print(args.model_path)
iter = args.iter
folder_path = args.folder_path

# 初始化一个列表来存储合并后的数据
merged_data = []
path_prefix = args.path_prefix
# 遍历文件夹中的文件
for filename in os.listdir(folder_path):
    if filename.startswith(path_prefix) and filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        
        # 读取 JSON 文件
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            merged_data.extend(data)

data_list = merged_data
# 计算每份数据的大小
import torch
ng = torch.cuda.device_count()

total_prompts = len(data_list)
batch_size = total_prompts // ng

# 分割数据
batches = [data_list[i:i + batch_size] for i in range(0, total_prompts, batch_size)]

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

def run_inference(model_path,batch, bs, thread_id,n):
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{thread_id}'
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False,trust_remote_code=True)
    # 初始化 VLLM 模型，指定 GPU
    sampling_params = SamplingParams(temperature=0.0, repetition_penalty=1.05, max_tokens=10,logprobs=20)
    vocab = tokenizer.get_vocab()
    yes_token_ids = [token_id for token, token_id in vocab.items() if ("yes" in token) or ("Yes" in token)]
    no_token_ids = [token_id for token, token_id in vocab.items() if ("no" in token) or ("No" in token)]

    # Input the model name or path. Can be GPTQ or AWQ models.
    llm = LLM(model=model_path,dtype='half',trust_remote_code=True)
    
    # 将 batch 中的 prompt 提取出来
    # prompts = [chat_template.format(item['question']) for item in batch]
    prompts = []
    questions = []
    for item in batch:
        ctr = item['ctr']
        ans = item['ans']
        for a in ans:
            qq = '''And we also provide the summary (between <summary> and </summary>) of user's watch preference for your reference.
<summary>{}</summary>
Please considering the user's preferences, historical viewing records to predict.'''.format(a)
            pp = ctr + " "+qq
            messages = [
                {"role": "system", "content": norm_qwen_prompt},
                {"role": "user", "content": pp}
            ]

            text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            prompts.append(text)

        questions.append(item['question'])

    labels = [item['label'] for item in batch]
    ctrs = [item['ctr'] for item in batch]
    label_tags = [item['label_tag'] for item in batch]
    anss = [item['ans'] for item in batch]
    
    # 保存结果到文件
    generated_texts = []
    # batch_sample = []
    save_infer_n_dir = args.save_infer_n_dir
    with open(f'{save_infer_n_dir}/infer_train_ans_{iter}_{thread_id}.json', 'w') as f:
        for i in tqdm(range(0,len(labels),bs)):
            msg = prompts[i*n:min((i+bs),len(labels))*n]
            ques = questions[i:min((i+bs),len(labels))]
            ctr = ctrs[i:min((i+bs),len(labels))]
            label_tag = label_tags[i:min((i+bs),len(labels))]
            label = labels[i:min((i+bs),len(labels))]
            ans = anss[i:min((i+bs),len(labels))]

            # 进行批量推理
            outputs = llm.generate(msg, sampling_params)
            ots = []
            for num in range(0,len(outputs),n):
                ots.append(outputs[num:num+n])
            # 处理输出结果
            
            for a,c,lt,q,l,output in zip(ans,ctr,label_tag,ques,label, ots):
                # generated_text = output.outputs[0].text
                generated_text = []
                probs = []
                for o in output:
                    token_ids = o.outputs[0].token_ids
                    logprobs = o.outputs[0].logprobs
                    target_p = logprobs[0]
                    target = token_ids[0]
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
                    # logit = yes_prob/(yes_prob+no_prob)
                    logit = math.exp(yes_prob)/(math.exp(yes_prob)+math.exp(no_prob))

                    probs.append(logit)
                    generated_text.append(o.outputs[0].text)

                generated_texts.append({'question':q,'ctr':c,'label_tag':lt,'label':l,'cot':a,'ans':generated_text,'probs':probs})
        
        json.dump(generated_texts,f,ensure_ascii=False,indent=4)
            # break

if __name__ == "__main__":
    processes = []
    model_path = args.model_path
    for i in range(ng):
        p = multiprocessing.Process(target=run_inference, args=(model_path,batches[i], 200, i,10))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()