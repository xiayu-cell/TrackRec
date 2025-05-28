import torch
import torch.utils.data as Data
import pickle
from utils import load_json, load_pickle
import json
from tqdm import tqdm
import numpy as np


class AmzDataset(Data.Dataset):
    def __init__(self, data_path, set='train', task='ctr', max_hist_len=10, augment=False, aug_prefix=None):
        self.task = task
        self.max_hist_len = max_hist_len
        self.augment = augment
        self.set = set
        self.data = load_pickle("/mmu_vcg_wjc_hdd/wanglei16/paper/www2025/CoLLM/dataset/amazon_book" + f'/{set}_filtered.pkl')
        self.stat = load_json(data_path + '/stat.json')
        self.item_num = self.stat['item_num']
        self.attr_num = self.stat['attribute_num']
        self.attr_ft_num = self.stat['attribute_ft_num']
        self.rating_num = self.stat['rating_num']
        self.dense_dim = self.stat['dense_dim']
        if task == 'rerank':
            self.max_list_len = self.stat['rerank_list_len']
        self.length = len(self.data)
        print(self.length)
        self.item2attribution = load_json(data_path + '/item2attributes.json')
        datamaps = load_json(data_path + '/datamaps.json')
        self.id2item = datamaps['id2item']
        self.id2user = datamaps['id2user']
        self.user2id = datamaps['user2id']
        self.item2id = datamaps['item2id']
        self.user_num = len(self.id2user.keys())
        cnt1 = 0
        cnt2 = 0
        for _, row in self.data.iterrows():
            uid = row["uid"]
            iid = row["iid"]
            if (uid not in self.user2id) or (iid not in self.item2id):
                continue
            cnt1 += 1
            cnt2 += 1
            for iid in row["his"][1:][-10:]:
                if iid not in self.item2id:
                    print("exposure {} not in item set!!!".format(iid))
                    cnt2 -= 1
                    break
        print(cnt1, cnt2)
        exit()
        self.sequential_data = dict()
        for idx, row in self.data.iterrows():
            uid = self.user2id[str(row["uid"])]
            iid = self.item2id[str(row['iid'])]
            his = row["his"][1:][-10:]
            ratings = row["his_rating"][1:][-10:]
            if uid not in self.sequential_data:
                self.sequential_data[uid] = dict()
            if iid not in self.sequential_data[uid]:
                self.sequential_data[uid][iid] = []
            else:
                print("already exposure!!!", row["uid"], row['iid'])
            self.sequential_data[uid][iid] = [his, ratings]
        self.input_aug_data = dict()
        self.output_aug_data = dict()
        if augment:
            with open(data_path + f'/{aug_prefix}_{set}.hist', "r") as f:
                for idx, line in enumerate(tqdm(f.readlines())):
                    d = json.loads(line.strip())
                    if d['user_id'] not in self.input_aug_data:
                        self.input_aug_data[d['user_id']] = dict()
                    self.input_aug_data[d['user_id']][d['item_id']] = d['sft_input_emb']
                    
                    if d['user_id'] not in self.output_aug_data:
                        self.output_aug_data[d['user_id']] = dict()
                    self.output_aug_data[d['user_id']][d['item_id']] = d['sft_output_emb']

    def __len__(self):
        return self.length

    def __getitem__(self, _id):
        if self.task == 'ctr':
            uid, seq_idx, lb = self.data[_id]
            item_seq, rating_seq = self.sequential_data[str(uid)]
            iid = item_seq[seq_idx]
            hist_seq_len = seq_idx - max(0, seq_idx - self.max_hist_len)
            attri_id = self.item2attribution[str(iid)]
            hist_item_seq = item_seq[max(0, seq_idx - self.max_hist_len): seq_idx]
            hist_rating_seq = rating_seq[max(0, seq_idx - self.max_hist_len): seq_idx]
            hist_attri_seq = [self.item2attribution[str(idx)] for idx in hist_item_seq]
            out_dict = {
                'iid': torch.tensor(iid).long(),
                'aid': torch.tensor(attri_id).long(),
                'lb': torch.tensor(lb).long(),
                'hist_iid_seq': torch.tensor(hist_item_seq).long(),
                'hist_aid_seq': torch.tensor(hist_attri_seq).long(),
                'hist_rate_seq': torch.tensor(hist_rating_seq).long(),
                'hist_seq_len': torch.tensor(hist_seq_len).long()
            }
            if self.augment:
                if str(self.id2item[str(iid)]) in self.item_aug_data:
                    item_aug_vec = self.item_aug_data[str(self.id2item[str(iid)])]
                else:
                    item_aug_vec = [0.0] * self.dense_dim
                if str(self.id2user[str(uid)]) in self.hist_aug_data:
                    hist_aug_vec = self.hist_aug_data[str(self.id2user[str(uid)])]
                else:
                    hist_aug_vec = [0.0] * self.dense_dim
                out_dict['item_aug_vec'] = torch.tensor(item_aug_vec).float()
                out_dict['hist_aug_vec'] = torch.tensor(hist_aug_vec).float()
        elif self.task == 'rerank':
            uid, seq_idx, candidates, candidate_lbs = self.data[_id]
            candidates_attr = [self.item2attribution[str(idx)] for idx in candidates]
            item_seq, rating_seq = self.sequential_data[str(uid)]
            hist_seq_len = seq_idx - max(0, seq_idx - self.max_hist_len)
            hist_item_seq = item_seq[max(0, seq_idx - self.max_hist_len): seq_idx]
            hist_rating_seq = rating_seq[max(0, seq_idx - self.max_hist_len): seq_idx]
            hist_attri_seq = [self.item2attribution[str(idx)] for idx in hist_item_seq]
            out_dict = {
                'iid_list': torch.tensor(candidates).long(),
                'aid_list': torch.tensor(candidates_attr).long(),
                'lb_list': torch.tensor(candidate_lbs).long(),
                'hist_iid_seq': torch.tensor(hist_item_seq).long(),
                'hist_aid_seq': torch.tensor(hist_attri_seq).long(),
                'hist_rate_seq': torch.tensor(hist_rating_seq).long(),
                'hist_seq_len': torch.tensor(hist_seq_len).long()
            }
            #TODO: update for cot emb
            if self.augment:
                item_aug_vec = [torch.tensor(self.item_aug_data[str(self.id2item[str(idx)])]).float()
                                for idx in candidates]
                hist_aug_vec = self.hist_aug_data[str(self.id2user[str(uid)])]
                out_dict['item_aug_vec_list'] = item_aug_vec
                out_dict['hist_aug_vec'] = torch.tensor(hist_aug_vec).float()
        else:
            raise NotImplementedError

        return out_dict


class EcomDataset(Data.Dataset):
    def __init__(self, data_path, set='train', task='ctr', max_hist_len=10, augment=False, aug_prefix=None):
        self.task = task
        self.max_hist_len = max_hist_len
        self.augment = augment
        self.set = set
        self.data = load_pickle(data_path + f'/{task}.{set}')
        self.stat = load_json(data_path + '/stat.json')
        self.item_num = self.stat['item_num']
        self.user_num = self.stat['user_num']
        self.attr_num = self.stat['attribute_num']
        self.attr_ft_num = self.stat['attribute_ft_num']
        self.dense_dim = self.stat['dense_dim']
        if task == 'rerank':
            self.max_list_len = self.stat['rerank_list_len']
        self.length = len(self.data)
        self.sequential_data = load_json(data_path + '/sequential_data.json')
        self.test_sequential_data = load_json(data_path + '/test_sequential_data.json')
        self.item2attribution = load_json(data_path + '/item2attributes.json')
        self.user2attribution = load_json(data_path + '/user2attributes.json')
        datamaps = load_json(data_path + '/datamaps.json')
        self.id2item = datamaps['id2item']
        self.id2user = datamaps['id2user']
        if augment:
            #self.hist_aug_data = load_json(data_path + f'/{aug_prefix}_augment.hist')
            #self.item_aug_data = load_json(data_path + f'/{aug_prefix}_augment.item')
            self.input_aug_data = {}
            self.output_aug_data = {}
            with open(data_path + f'/{set}_{aug_prefix}.emb', "r") as f:
                for idx, line in enumerate(tqdm(f.readlines())):
                    d = json.loads(line.strip())
                    self.input_aug_data[d['user_id']] = d['sft_input_emb']
                    self.output_aug_data[d['user_id']] = d['sft_output_emb'] 
            # print('item key', list(self.item_aug_data.keys())[:6], len(self.item_aug_data), self.item_num)

    def __len__(self):
        return self.length

    def __getitem__(self, _id):
        if 'ctr' in self.task:
            uid, iid, lb = self.data[_id]
            attri_id = self.item2attribution[str(iid)]
            user_attri_id = self.user2attribution[str(uid)]
            out_dict = {
                'iid': torch.tensor(iid).long(),
                'uid': torch.tensor(uid).long(),
                'aid': torch.tensor(attri_id).long(),
                'uaid': torch.tensor(user_attri_id).long(),
                'lb': torch.tensor(lb).long(),
            }
            if self.augment:
                # if str(self.id2item[str(iid)]) in self.item_aug_data:
                #     item_aug_vec = self.item_aug_data[str(self.id2item[str(iid)])]
                # else:
                #     item_aug_vec = [0.0] * self.dense_dim
                # hist_aug_vec = []
                # for idx in hist_item_seq:
                #     item_id = str(self.id2item[str(idx)])
                #     if item_id in self.item_aug_data:
                #         aug_vec = self.item_aug_data[item_id]
                #         hist_aug_vec.append(aug_vec)
                #     else:
                #         hist_aug_vec.append([0.0] * self.dense_dim)
                # hist_aug_vec = torch.tensor(hist_aug_vec).float().mean(0)
                #item_aug_vec = self.item_aug_data[str(self.id2item[str(iid)])]
                if int(uid) in self.input_aug_data:
                    intput_aug_vec = self.input_aug_data[int(uid)]
                else:
                    intput_aug_vec = [0.0] * self.dense_dim
                if int(uid) in self.output_aug_data:
                    output_aug_vec = self.output_aug_data[int(uid)]
                else:
                    output_aug_vec = [0.0] * self.dense_dim
                out_dict['input_aug_vec'] = torch.tensor(intput_aug_vec).float()
                out_dict['output_aug_vec'] = torch.tensor(output_aug_vec).float()
        else:
            raise NotImplementedError

        return out_dict


class MovieCoLLMDataset(Data.Dataset):
    def __init__(self, data_path, set='train', task='ctr', max_hist_len=10, augment=False, aug_prefix=None):
        self.task = task
        self.max_hist_len = max_hist_len
        self.augment = augment
        self.set = set
        self.data = load_pickle("/mmu_vcg_wjc_hdd/wanglei16/paper/www2025/CoLLM/dataset/ml-1m" + f'/{set}_filtered.pkl')
        self.stat = load_json(data_path + '/stat.json')
        self.item_num = self.stat['item_num'] # ???
        self.attr_num = self.stat['attribute_num']
        self.attr_ft_num = self.stat['attribute_ft_num']
        self.rating_num = self.stat['rating_num']
        self.dense_dim = self.stat['dense_dim']

        self.length = len(self.data)
        #self.sequential_data = load_json(data_path + '/sequential_data.json')
        self.item2attribution = load_json(data_path + '/item2attributes.json')
        datamaps = load_json(data_path + '/datamaps.json')
        self.id2item = datamaps['id2item']
        self.id2user = datamaps['id2user']
        self.user2id = datamaps['user2id']
        self.item2id = datamaps['item2id']
        self.user_num = len(self.id2user.keys())
        self.sequential_data = dict()
        for idx, row in self.data.iterrows():
            uid = self.user2id[str(row["uid"])]
            iid = self.item2id[str(row['iid'])]
            his = row["his"][1:][-10:]
            ratings = row["his_rating"][1:][-10:]
            if uid not in self.sequential_data:
                self.sequential_data[uid] = dict()
            if iid not in self.sequential_data[uid]:
                self.sequential_data[uid][iid] = []
            else:
                print("already exposure!!!", row["uid"], row['iid'])
            self.sequential_data[uid][iid] = [his, ratings]
        self.input_aug_data = dict()
        self.output_aug_data = dict()
        if augment:
            with open(data_path + f'/{aug_prefix}_{set}.hist', "r") as f:
                for idx, line in enumerate(tqdm(f.readlines())):
                    d = json.loads(line.strip())
                    if d['user_id'] not in self.input_aug_data:
                        self.input_aug_data[d['user_id']] = dict()
                    self.input_aug_data[d['user_id']][d['item_id']] = d['sft_input_emb']
                    
                    if d['user_id'] not in self.output_aug_data:
                        self.output_aug_data[d['user_id']] = dict()
                    self.output_aug_data[d['user_id']][d['item_id']] = d['sft_output_emb']
            # print('item key', list(self.item_aug_data.keys())[:6], len(self.item_aug_data), self.item_num)

    def __len__(self):
        return self.length

    def __getitem__(self, _id):
        if self.task == 'ctr':
            uid = self.user2id[str(self.data.iloc[_id]['uid'])]
            iid = self.item2id[str(self.data.iloc[_id]['iid'])]
            lb = self.data.iloc[_id]['label']
            item_seq, rating_seq = self.sequential_data[uid][iid]
            #iid = item_seq[seq_idx]
            hist_seq_len = 10
            hist_seq_len = len(item_seq)
            assert hist_seq_len == 10
            attri_id = self.item2attribution[str(iid)]
            hist_item_seq = [self.item2id[str(iid)] for iid in item_seq]
            hist_rating_seq = rating_seq
            #hist_attri_seq = [self.item2attribution[str(idx)] for idx in hist_item_seq]
            # pad list
            hist_rating_seq = np.pad(hist_rating_seq, (self.max_hist_len - hist_seq_len, 0), 'constant', constant_values=(0)) # ???
            hist_item_seq = np.pad(hist_item_seq, (self.max_hist_len - hist_seq_len, 0), 'constant', constant_values=(0))
            hist_attri_seq = [self.item2attribution.get(str(idx), [0]) for idx in hist_item_seq]
            out_dict = {
                'uid': torch.tensor(uid).long(),
                'iid': torch.tensor(iid).long(),
                'aid': torch.tensor(attri_id).long(),
                'lb': torch.tensor(lb).long(),
                'hist_iid_seq': torch.tensor(hist_item_seq).long(),
                'hist_aid_seq': torch.tensor(hist_attri_seq).long(),
                'hist_rate_seq': torch.tensor(hist_rating_seq).long(),
                'hist_seq_len': torch.tensor(hist_seq_len).long()
            }
            if self.augment:
                if uid in self.input_aug_data and iid in self.input_aug_data[uid]:
                    hist_aug_vec = self.input_aug_data[uid][iid]
                else:
                    hist_aug_vec = [0.0] * self.dense_dim
                if uid in self.output_aug_data and iid in self.output_aug_data[uid]:
                    item_aug_vec = self.output_aug_data[uid][iid]
                else:
                    item_aug_vec = [0.0] * self.dense_dim
                out_dict['input_aug_vec'] = torch.tensor(hist_aug_vec).float()
                out_dict['output_aug_vec'] = torch.tensor(item_aug_vec).float()

        return out_dict

if __name__ == '__main__':
    # train_set = EcomDataset('../data/ecom_dpo/proc_data', 'train', 'ctr', 5, True, 'sft_bge')
    # train_loader = Data.DataLoader(dataset=train_set, batch_size=1, shuffle=False)
    # for i, data in enumerate(train_loader):
    #     print(data)

    train_set = MovieCoLLMDataset('../data/ml-1m/proc_data', 'train', 'ctr', 10, False, aug_prefix='bge')
    train_set = AmzDataset('../data/amz/proc_data', 'train', 'ctr', 10, False, aug_prefix='bge')
    train_loader = Data.DataLoader(dataset=train_set, batch_size=1, shuffle=False)
    #from torch.nn.utils.rnn import pad_sequence
    #import torch.nn.functional as F
    for i, data in enumerate(train_loader):
        #print(data)
        print(data)
