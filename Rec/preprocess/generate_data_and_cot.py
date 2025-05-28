import json
import os
import pickle
from datetime import date
import random
from collections import defaultdict
import csv
from pre_utils import parse, load_json, save_json, save_pickle, GENDER_MAPPING, \
    AGE_MAPPING, OCCUPATION_MAPPING

rerank_item_from_hist = 4
rerank_hist_len = 10
rerank_list_len = 10
ctr_hist_len = 10


def generate_ctr_data(sequence_data, lm_hist_idx, uid_set):
    # print(list(lm_hist_idx.values())[:10])
    full_data = []
    total_label = []
    for uid in uid_set:
        start_idx = lm_hist_idx[str(uid)]
        item_seq, rating_seq = sequence_data[str(uid)]
        for idx in range(start_idx, len(item_seq)):
            label = 1 if rating_seq[idx] > rating_threshold else 0
            full_data.append([uid, idx, label])
            total_label.append(label)
    print('user num', len(uid_set), 'data num', len(full_data), 'pos ratio',
          sum(total_label) / len(total_label))
    print(full_data[:5])
    return full_data


def generate_rerank_data(sequence_data, lm_hist_idx, uid_set, item_set):
    full_data = []
    for uid in uid_set:
        start_idx = lm_hist_idx[str(uid)]
        item_seq, rating_seq = sequence_data[str(uid)]
        idx = start_idx
        seq_len = len(item_seq)
        while idx < seq_len:
            end_idx = min(idx + rerank_item_from_hist, seq_len)
            chosen_iid = item_seq[idx:end_idx]
            neg_sample_num = rerank_list_len - len(chosen_iid)
            neg_sample = random.sample(item_set, neg_sample_num)
            candidates = chosen_iid + neg_sample
            chosen_rating = rating_seq[idx:end_idx]
            candidate_lbs = [1 if rating > rating_threshold else 0 for rating in
                             chosen_rating] + [0 for _ in range(neg_sample_num)]
            list_zip = list(zip(candidates, candidate_lbs))
            random.shuffle(list_zip)
            candidates[:], candidate_lbs[:] = zip(*list_zip)
            full_data.append([uid, idx, candidates, candidate_lbs])
            idx = end_idx
    print('user num', len(uid_set), 'data num', len(full_data))
    print(full_data[:5])
    return full_data


def load_cot_data(data_file):
    res = {}
    with open(data_file) as f:
        for line in f.readlines():
            dd = json.loads(line.strip())
            iid = dd['id']
            answer = dd['label_answer']
            res[iid] = answer
    return res


def generate_hist_prompt(sequence_data, item2attribute, datamap, lm_hist_idx, dataset_name):
    itemid2title = datamap['itemid2title']
    attrid2name = datamap['id2attribute']
    id2user = datamap['id2user']
    if dataset_name == 'ml-1m':
        user2attribute = datamap['user2attribute']
    hist_prompts = []
    print('item2attribute', list(item2attribute.items())[:10])
    if dataset_name == 'ml-1m':
        rating_threshold = 3
    else:
        rating_threshold = 4
    if dataset_name == 'amz':
        data_file = os.path.join('../data/', dataset_name, 'raw_data', 'amz_reviews.json')
        review_data = load_json(data_file)
    data_file = os.path.join('../data/', dataset_name, 'proc_data', 'prompt.item.mmu.gpt35_output')
    cot_data = load_cot_data(data_file)
    for uid, item_rating in sequence_data.items():
        user = id2user[uid]
        item_seq, rating_seq = item_rating
        cur_idx = lm_hist_idx[uid]
        hist_item_seq = item_seq[:cur_idx]
        hist_rating_seq = rating_seq[:cur_idx]
        history_texts = []
        item_cnt = 1
        for iid, rating in zip(hist_item_seq, hist_rating_seq):
            #tmp = '"{}", {} stars; '.format(itemid2title[str(iid)], int(rating))
            item_id = datamap['id2item'][str(iid)]
            if dataset_name == 'amz':
                review = ''
                if user in review_data and item_id in review_data[user]:
                    review = review_data[user][item_id]
                cot = ''
                if item_id in cot_data:
                    cot = cot_data[item_id]
                tmp = '###\nThe Book is {}.\nrecommended reasons and factual knowledge:\n{}\nuser rating:\n{}\n###\n'.format(itemid2title[str(iid)], cot, int(rating))
                history_texts.append(tmp)
            elif dataset_name == 'ml-1m':
                cot = ''
                if item_id in cot_data:
                    cot = cot_data[item_id]
                tmp = '###\nThe Movie is {}.\nrecommended reasons and factual knowledge:\n{}\nuser rating:\n{}\n###\n'.format(itemid2title[str(iid)], cot, int(rating))
                history_texts.append(tmp)
            if item_cnt == 15 or (iid == hist_item_seq[-1]):
                item_cnt = 1
                user_prompt = {}
                if dataset_name == 'amz':
                    # prompt = 'Analyze user\'s preferences on product (consider factors like genre, functionality, quality, ' \
                    #          'price, design, reputation. Provide clear explanations based on ' \
                    #          'relevant details from the user\'s product viewing history and other pertinent factors.'
                    # hist_prompts[uid] = 'Given user\'s product rating history: ' + ''.join(history_texts) + prompt
                    prompt = 'Analyze user\'s preferences on books about factors like genre, author, writing style, theme, ' \
                            'setting, length and complexity, time period, literary quality, critical acclaim (Provide ' \
                            'clear explanations based on relevant details from the user\'s book viewing history and other ' \
                            'pertinent factors.'
                    prompt = 'You are a professional book reviewer and psychoanalyst.\nPlease analyze the user\'s real preferences when it comes to rating books based on that user and his collection of book rating history, specifically based on the recommended reasons and factual knowledge and the user\'s rating of the each book in the history.\n\nPlease note that a rating of 5 is a positive rating, and a rating of less than 5 is a negative rating\n\n'
                    user_prompt[user] = prompt + 'Given user\'s book rating history:\n' + '\n'.join(history_texts)
                    hist_prompts.append(user_prompt)
                elif dataset_name == 'ml-1m':
                    gender, age, occupation = user2attribute[uid]
                    user_text = 'Given a {} user who is aged {} and {}, this user\'s movie viewing history over time' \
                                ' is listed below. '.format(GENDER_MAPPING[gender], AGE_MAPPING[age],
                                                            OCCUPATION_MAPPING[occupation])
                    question = 'Analyze user\'s preferences on movies (consider factors like genre, director/actors, time ' \
                            'period/country, character, plot/theme, mood/tone, critical acclaim/award, production quality, ' \
                            'and soundtrack). Provide clear explanations based on relevant details from the user\'s movie ' \
                            'viewing history and other pertinent factors.'
                    prompt = 'You are a professional movie reviewer and psychoanalyst.\nPlease analyze the user\'s real preferences when it comes to rating movies based on that user and his collection of movie rating history, specifically based on the recommended reasons and factual knowledge and the user\'s rating of the each movie in the history.\n\nPlease note that a rating of 5 or 4 is a positive rating, and a rating of less than 4 is a negative rating\n\n'
                    user_prompt[user] = prompt + 'Given user\'s movie rating history:\n' + '\n'.join(history_texts)
                    hist_prompts.append(user_prompt)
                else:
                    raise NotImplementedError
                history_texts = []
            else:
                item_cnt += 1
    print('data num', len(hist_prompts))
    #print(list(hist_prompts.items())[0])
    return hist_prompts


def generate_item_prompt(item2attribute, datamap, dataset_name):
    itemid2title = datamap['itemid2title']
    attrid2name = datamap['id2attribute']
    id2item = datamap['id2item']
    item_prompts = {}
    for iid, title in itemid2title.items():
        item = id2item[iid]
        if dataset_name == 'amz':
            brand, cate = item2attribute[str(iid)][0:2]
            brand_name = attrid2name[str(brand)].replace('Visit Amazon\'s', '')
            cate_name = attrid2name[str(cate)]
            item_prompts[item] = 'Introduce book {}, which is from brand {} and describe its attributes including but' \
                                ' not limited to genre, author, writing style, theme, setting, length and complexity, ' \
                                'time period, literary quality, critical acclaim.'.format(title, brand_name)
            #item_prompts[item] = 'You are a professional marketing analyst, well versed in sales, marketing psychology, and logical argumentation. Please help me analyze how the following book is arguing to the user that the user should buy the book item? Please analyze it in several ways: \n1. Quality of content: whether the content of the book is in-depth and practical, and whether it can give readers new knowledge or insightful ideas.\n2. Author\'s reputation: whether the author has recognized authority or experience, and whether their other works have been well received.\n3. recommendations and reviews: what are the book reviews, testimonials, media coverage and reader feedback.\n4. brand image: the reputation of the publisher and whether there is a good brand image to attract readers to buy the book.\n5. Readability: whether the font of the book is clear, whether the layout is reasonable, how is the quality of the paper, and so on.\n6. Originality: whether the content of the book is original, or what is the quality of translation.\n7. Specialization: whether the book is aimed at professional readers in a particular field, and the operation of market segmentation can often create greater market value.\n8. Frequency of update: If it is a series of books, the frequency of update and the quality of new books may also affect the readers\' willingness to buy.\n9. Visibility and exposure: The visibility and exposure of the book in the media and market may also affect its sales.\n\nLet us think step by step, using common sense and knowledge from various disciplines.\n\nThe title of the book is {}.\nThe brand & author of the book is {}.\nThe categories of the book is {}.\n'.format(title, brand_name, cate_name)
            item_prompts[item] = 'You are a professional book reviewer with broad knowledge, deep understanding, critical thinking, sensitive aesthetic perception and excellent presentation skills. Please help me analyze how the following book is arguing to the user that the user should rate this book high or low? Please analyze it in several ways: \n1. **Theme and Meaning**: analyze the central ideas, themes, or messages conveyed that the book explores. Discuss whether these themes have depth, novelty, and how they relate to the reader\'s real life.\n2. **Plot Structure**: evaluate the layout and flow of the story, including the beginning, development, climax, and ending, whether it is reasonably compelling, with or without suspense and momentum to move the story forward.\n3. **Character Development**: Examine whether the characterization in the book is comprehensive and rich, whether the characters have a clear growth trajectory, and whether the interactions among the characters are realistic and believable.\n4. **Genre and Style**: Evaluate the author\'s writing style, whether it is unique, whether the language expression is fluent, and whether the literary techniques used, such as symbols and metaphors, are appropriate and effective.\n5. **Background Setting**: Explore whether the background of the story is detailed and accurate, and whether cultural and historical elements are properly presented and deeply understood.\n6. **Emotional impact**: analyze how the book affects the reader on an emotional level, whether it resonates, and whether the emotional descriptions are moving.\n7. **Philosophical and Moral Exploration**: examines whether the book contains discussions of philosophical, ethical, or moral issues, and the depth and impact of these discussions.\n8. **Originality and Innovation**: assesses the degree of innovation in the work, including the originality of the subject matter, the uniqueness of the story, and novel narrative techniques.\n9. **Social and Cultural Impact**: Analyze the work\'s reflection and impact on society and culture, and whether it reveals social issues and provides cultural criticism or reflection.\n10. **Audience and Readability**: Consider the target readership, analyze the degree of difficulty and general acceptability, and evaluate its generalizability and popularity among different readers.\n\nLet us think step by step, using common sense and knowledge from various disciplines.\n\nThe title of the book is {}.\nThe brand & author of the book is {}.\nThe categories of the book is {}.\n'.format(title, brand_name, cate_name)
            # item_prompts[iid] = 'Introduce product {}, which is from brand {} and describe its attributes (including but' \
            #                     ' not limited to genre, functionality, quality, price, design, reputation).'.format(title, brand_name)
        elif dataset_name == 'ml-1m':
            cate = item2attribute[str(iid)][0]
            cate_name = attrid2name[str(cate)]
            item_prompts[item] = 'Introduce movie {} and describe its attributes (including but not limited to genre, ' \
                                'director/cast, country, character, plot/theme, mood/tone, critical ' \
                                'acclaim/award, production quality, and soundtrack).'.format(title)
            item_prompts[item] = 'You are a professional movie critic with a deep knowledge base of movies, a love and passion for movies, and an excellent taste and appreciation for movies. Please help me analyze how the following movie is arguing to users that they should rate the movie high or low? Please analyze it in several ways:\n1. storyline: the attractiveness of the story, the coherence of the plot, whether the story is deep and touching.\n2. Directing skills: how the director manages the pace of the movie, how he/she guides the actors, how he/she expresses the theme of the story through the language of the camera.\n3. Actors\' performances: whether the actors are able to portray their characters vividly, and whether their performances are real and infectious.\n4. Audio-visual effects: whether the camera technique, visual effects, music and sound effects match the theme of the movie, and whether they are excellent.\n5. Editing: whether the editing is smooth, whether the transition is natural, and whether it gives a sense of coherence and completeness.\n6. Artistic design: whether the artistic design of the sets, costumes, make-up, etc. is excellent and whether it can increase the enjoyment of the movie.\n7. Theme and value: Whether the theme and value conveyed by the movie are profound and can stimulate people\'s thinking.\n8. Dialogue and language: whether the dialogue is natural, whether the language is charming, and whether it adds color to the story.\n9. Originality: whether the movie has unique and innovative elements, whether it imitates, repeats or has original spirit and style.\n10. Audience feedback: the movie\'s audience feedback and impact, including word-of-mouth and box office.\n\nLet us think step by step, using common sense and knowledge from various disciplines.\n\nThe title of the movies is {} and the category of the movie is {}.'.format(title, cate_name)
        else:
            raise NotImplementedError
    print('data num', len(item_prompts))
    print(list(item_prompts.items())[0])
    return item_prompts


if __name__ == '__main__':
    random.seed(12345)
    DATA_DIR = '../data/'
    #DATA_SET_NAME = 'amz'
    DATA_SET_NAME = 'ml-1m'
    if DATA_SET_NAME == 'ml-1m':
        rating_threshold = 3
    else:
        rating_threshold = 4
    PROCESSED_DIR = os.path.join(DATA_DIR, DATA_SET_NAME, 'proc_data')
    SEQUENCE_PATH = os.path.join(PROCESSED_DIR, 'sequential_data.json')
    ITEM2ATTRIBUTE_PATH = os.path.join(PROCESSED_DIR, 'item2attributes.json')
    DATAMAP_PATH = os.path.join(PROCESSED_DIR, 'datamaps.json')
    SPLIT_PATH = os.path.join(PROCESSED_DIR, 'train_test_split.json')

    sequence_data = load_json(SEQUENCE_PATH)
    train_test_split = load_json(SPLIT_PATH)
    item2attribute = load_json(ITEM2ATTRIBUTE_PATH)
    item_set = list(map(int, item2attribute.keys()))
    print('final loading data')
    # print(list(item2attribute.keys())[:10])

    datamap = load_json(DATAMAP_PATH)

    print('generating item prompt')
    item_prompt = generate_item_prompt(item2attribute, datamap, DATA_SET_NAME)
    print('generating history prompt')
    hist_prompt = generate_hist_prompt(sequence_data, item2attribute, datamap,
                                       train_test_split['lm_hist_idx'], DATA_SET_NAME)
    print('save prompt data')
    save_json(item_prompt, PROCESSED_DIR + '/prompt.item')
    save_json(hist_prompt, PROCESSED_DIR + '/prompt.hist')
    item_prompt, hist_prompt = None, None

