import json
import os
import argparse

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
model_path = args.model_path

# 遍历文件夹中的文件
for filename in os.listdir(folder_path):
    if filename.startswith(path_prefix) and filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        
        # 读取 JSON 文件
        with open(file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                merged_data.append(data)

iters = 5
# data_list = merged_data[(int(iter)-1)*(len(merged_data)//iters):(int(iter)*(len(merged_data)//iters))]

data_list = merged_data
# 计算每份数据的大小
print(len(data_list))
# print((int(iter)-1)*(len(merged_data)//iters))
# print(int(iter)*(len(merged_data)//iters))
total_prompts = len(data_list)
import torch
ng = torch.cuda.device_count()
#batch_size = total_prompts // 8
batch_size = total_prompts // ng


# 分割数据
# batches = [data_list[i:i + batch_size] for i in range(0, total_prompts, batch_size)]
batches = [data_list[i:i + batch_size] for i in range(0, total_prompts, batch_size)]
# 如果数据不能均匀分割，最后一组可能会有剩余的数据
if len(batches) > ng:
    batches[ng - 1].extend(batches[ng])
    batches = batches[:ng]
print("总数据: {}".format(total_prompts))
print("每份数据:", [len(val) for val in batches])



import multiprocessing
from vllm import LLM, SamplingParams
# chat_template = '<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n'
from tqdm import tqdm
from transformers import AutoTokenizer
import torch
import tqdm as tq
import random


norm_qwen_prompt = "You are a helpful assistant."
short_qwen_prompt = "You are a helpful assistant. You will provide very concise and helpful response."

def run_inference(model_path,batch, bs, thread_id, n):
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{thread_id}'
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False,trust_remote_code=True)
    # 初始化 VLLM 模型，指定 GPU
    #sampling_params = SamplingParams(temperature=1.0, top_p=0.9, repetition_penalty=1.05, max_tokens=512)
    sampling_params = [SamplingParams(temperature=1.0,max_tokens=512, seed=random.randint(0,1e9)) for _ in tq.trange(0, 10 ,desc='sample param')]

    # Input the model name or path. Can be GPTQ or AWQ models.
    llm = LLM(model=model_path,dtype='half',trust_remote_code=True)

    # 将 batch 中的 prompt 提取出来
    # prompts = [chat_template.format(item['question']) for item in batch]
    prompts = []
    questions = []
    for item in batch:
        messages = [
                {"role": "system", "content": norm_qwen_prompt},
                {"role": "user", "content": item['cot_prompt']}
            ]

        text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        questions.append(item['cot_prompt'])
        prompts.append(text)

    labels = []
    for item in batch:
        if item['completion'] == "No":
            labels.append(0)
        elif item['completion'] == "Yes":
            labels.append(1)
    
    ctrs = [item['prompt'] for item in batch]
    label_tags = [item['completion'] for item in batch]
    
    # 保存结果到文件
    generated_texts = []
    # batch_sample = []
    save_infer_n_dir = args.save_infer_n_dir
    with open(f'{save_infer_n_dir}/train_cot_step_{iter}_{thread_id}.json', 'w') as f:
        for i in tqdm(range(0,len(prompts),bs)):
            msg = prompts[i:min(i+bs,len(prompts))]
            ques = questions[i:min(i+bs,len(prompts))]
            ctr = ctrs[i:min(i+bs,len(prompts))]
            label_tag = label_tags[i:min(i+bs,len(prompts))]
            label = labels[i:min(i+bs,len(prompts))]
            # 进行批量推理
            oots = [[] for num in range(len(msg))] 
            for j, sampling_param in enumerate(sampling_params):
                outputs = llm.generate(msg, sampling_param)
                for num in range(len(outputs)):
                    oots[num].append(outputs[num].outputs[0].text)
            # 处理输出结果
            for c,lt,q,l,output in zip(ctr,label_tag,ques,label, oots):
                # generated_text = output.outputs[0].text
                generated_texts.append({'question':q,'ctr':c,'label_tag':lt,'label':l,'ans':output})
        
        json.dump(generated_texts,f,ensure_ascii=False,indent=4)
            # break

if __name__ == "__main__":
    processes = []
    #model_path = '/share/ad/xiayu12/Open-World-Knowledge-Augmented-Recommendation_Gang/checkpoints/Qwen/Qwen2___5-7B-Instruct'
    for i in range(ng):
        p = multiprocessing.Process(target=run_inference, args=(model_path,batches[i], 2000, i,10))
        processes.append(p)
        p.start()
    # for i, batch in enumerate(batches):
    #     run_inference(model_path, batch, 20, i, 10)
    
    # for p in processes:
    #     p.join()