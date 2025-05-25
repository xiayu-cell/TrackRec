import json

file_path = "/mmu_nlp_hdd/xiayu12/safe-rlhf_official/TrackRec/data/book/book1_tallrec_train.json"
data = []
with open(file_path, 'r') as file:
    for line in file:
        a = json.loads(line)
        qid = a['qid']
        a = a["data"][0]
        question = a['question']
        answer = a['answer']
        system = a['system']
        strs = question.split('\n')
        strs[2] = "Please summarize the user's book watch preference within 200 words."
        cot_prompt = '\n'.join(strs)

        data.append({
            "prompt": question,
            "completion": answer,
            "system":system,
            "cot_prompt": cot_prompt,
            "qid":qid
        })

with open(file_path, 'w') as file:
    for obj in data:
        file.write(json.dumps(obj) + '\n')