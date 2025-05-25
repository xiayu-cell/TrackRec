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
            for line in file:
                data = json.loads(line)
                merged_data.append(data)
data_list = merged_data
# test
# data_list = data_list[0:8]

# 计算每份数据的大小
total_prompts = len(data_list)

import torch
ng = torch.cuda.device_count()

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

norm_qwen_prompt = "You are a helpful assistant."
short_qwen_prompt = "You are a helpful assistant. You will provide very concise and helpful response."

def run_inference(model_path,batch, bs, thread_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{thread_id}'
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False,trust_remote_code=True)
    # 初始化 VLLM 模型，指定 GPU
    sampling_params = SamplingParams(temperature=0.0, repetition_penalty=1.05, max_tokens=512)

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

    with open(f'{save_infer_n_dir}/infer_model_A_step_{iter}_{thread_id}.json', 'w') as f:
        for i in tqdm(range(0,len(prompts),bs)):
            msg = prompts[i:min(i+bs,len(prompts))]
            ques = questions[i:min(i+bs,len(prompts))]
            label = labels[i:min(i+bs,len(prompts))]
            ctr = ctrs[i:min(i+bs,len(prompts))]
            ans = label_tags[i:min(i+bs,len(prompts))]
            # 进行批量推理
            outputs = llm.generate(msg, sampling_params)
    
            # 处理输出结果
            for c,a,q,l,output in zip(ctr,ans,ques,label, outputs):
                generated_text = output.outputs[0].text
                if l == 1:
                    a = "Yes"
                elif l == 0:
                    a = "No"
                # generated_texts.append({'question':q,'ctr':c,'label_tag':a,'label':l,'ans':generated_text})
                qq = '''And we also provide the summary (between <summary> and </summary>) of user's watch preference for your reference.
<summary>{}</summary>
Please considering the user's preferences, historical viewing records to predict. Only output the result (Yes or No)'''.format(generated_text)
                pp = c + " "+qq
                message_B = {
                    "ctr_prompt":pp,
                    "ans":a
                }
                generated_texts.append(message_B)
        json.dump(generated_texts,f,ensure_ascii=False,indent=4)

if __name__ == "__main__":
    processes = []
    # model_path = '/share/ad/xiayu12/TALLRec/my_paper/self_play/model_A/output/s_1_1e-6/checkpoint-180'
    for i in range(ng):
        p = multiprocessing.Process(target=run_inference, args=(model_path,batches[i], 100, i))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()