import json
from datetime import datetime
import spacy
import pytextrank
import pickle
from tqdm import tqdm
import os
import random
from config import keywords_from_text

fd_source_data = './source_data/'
fn_train_data = 'train_data_1.json'
fn_valid_data = 'valid_data.json'
fn_expert_attributes = 'expert_attributes'
fn_paper_attributtes = 'paper_attributes.json'
fn_output_reviewer_keywords = './middleware/reviewer_keywords.pkl'
fn_output_reviewer_paper = './middleware/reviewer_write_paper.pkl'
fn_output_paper_keywords = './middleware/paper_keywords.pkl'
fn_output_paper_reviewer = './middleware/paper_assignto_reviewer.pkl'
fn_output_valid_paper_keywords = './middleware/valid_paper_keywords.pkl'
fn_dataset_entity = './dataset/entities.dict'
fn_dataset_relation = './dataset/relations.dict'
fn_dataset_train = './dataset/train.txt'
fn_dataset_valid = './dataset/valid.txt'
fn_dataset_predict = './dataset/predict.txt'

def format_reviewer_keywords_edition(filename, number_start, number_end):
    '''
    由专家profile得出专家的关键词的方法：
    目前，除专家自填关键词外，基于题目和摘要的抽取采用textrank，之后考虑使用别的方法
    '''
    start_time = datetime.now()
    print('启动时间：', start_time.strftime('%Y-%m-%d %H:%M:%S'))
    reviewer_keywords = {}
    reviewer_paper = {}
    if keywords_from_text == 'textrank':
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe("textrank")
    for number in range(number_start, number_end+1):
        current_json = json.load(open(filename+str(number)+'.json'))
        print('本文件共计有',len(current_json),'条数据...')
        for i, each_reviewer in tqdm(enumerate(current_json)):
            if i%10000 == 0:
                print('正在处理第',number, '个文件的第', i, '组reviewer数据...')
            if each_reviewer['id'] not in reviewer_keywords:
                reviewer_keywords[each_reviewer['id']] = {}
            if 'interests' in each_reviewer and each_reviewer['interests'] != None and len(each_reviewer['interests']) > 0:
                for each_word in each_reviewer['interests']:
                    temp_word = each_word['t'].strip().replace('·','').lower()
                    if temp_word not in reviewer_keywords[each_reviewer['id']]:
                        reviewer_keywords[each_reviewer['id']][temp_word] = 0
                    try:
                        reviewer_keywords[each_reviewer['id']][temp_word] += int(each_word['w'])
                    except KeyError as ker:
                        print('[Warning] Reviewer',each_reviewer['id'], each_reviewer['name'],' has interest issue, skip.')
                        reviewer_keywords[each_reviewer['id']][temp_word] += 1
                        continue

            if 'tags' in each_reviewer and each_reviewer['tags'] != None and len(each_reviewer['tags']) > 0:
                for each_word in each_reviewer['tags']:
                    temp_word = each_word['t'].strip().replace('·','').lower()
                    if temp_word not in reviewer_keywords[each_reviewer['id']]:
                        reviewer_keywords[each_reviewer['id']][temp_word] = 0
                    try:
                        reviewer_keywords[each_reviewer['id']][temp_word] += int(each_word['w'])
                    except KeyError as ker:
                        print('[Warning] Reviewer',each_reviewer['id'], each_reviewer['name'],' has tag issue, skip.')
                        reviewer_keywords[each_reviewer['id']][temp_word] += 1
                        continue
            
            if each_reviewer['id'] not in reviewer_paper:
                reviewer_paper[each_reviewer['id']] = []
            for j, each_pub in enumerate(each_reviewer['pub_info']):
                if each_pub['id'] not in reviewer_paper[each_reviewer['id']]:
                    reviewer_paper[each_reviewer['id']].append(each_pub['id'])
                for each_paper_keyword in each_pub['keywords']:
                    temp_word = each_paper_keyword.strip().replace('·','').lower()
                    if temp_word not in reviewer_keywords[each_reviewer['id']]:
                        reviewer_keywords[each_reviewer['id']][temp_word] = 0
                    reviewer_keywords[each_reviewer['id']][temp_word] += 1
                if keywords_from_text == 'textrank':
                    for phrase in nlp(each_pub['title'])._.phrases:
                        if phrase.rank >= 0.15:
                            temp_word = phrase.text.strip().replace('·','').lower()
                            if temp_word not in reviewer_keywords[each_reviewer['id']]:
                                reviewer_keywords[each_reviewer['id']][temp_word] = 0
                            reviewer_keywords[each_reviewer['id']][temp_word] += 1
                    if 'abstract' not in each_pub:
                        continue
                    for phrase in nlp(each_pub['abstract'])._.phrases:
                        if phrase.rank >= 0.15:
                            temp_word = phrase.text.strip().replace('·','').lower()
                            if temp_word not in reviewer_keywords[each_reviewer['id']]:
                                reviewer_keywords[each_reviewer['id']][temp_word] = 0
                            reviewer_keywords[each_reviewer['id']][temp_word] += 1
    end_time = datetime.now()
    diff_time = end_time - start_time
    print('reviewer处理完成时间：', end_time.strftime('%Y-%m-%d %H:%M:%S'), '，耗时：',diff_time.total_seconds()/3600.0, 'hours.')
    return reviewer_keywords, reviewer_paper


def format_paper(filename_paper, filename_review):
    '''
    由Paper profile得出专家的关键词的方法：
    目前，除专家自填关键词外，基于题目和摘要的抽取采用textrank，之后考虑使用别的方法
    '''
    start_time = datetime.now()
    print('启动时间：', start_time.strftime('%Y-%m-%d %H:%M:%S'))
    paper_keywords = {}
    paper_reviewer = {}
    paper_info = json.load(open(filename_paper,'r'))
    for i, each_paper in enumerate(paper_info):
        if each_paper['id'] not in paper_keywords:
            paper_keywords[each_paper['id']] = {}
        
        if 'keywords' in each_paper and each_paper['keywords'] != None and len(each_paper['keywords']) > 0:
            for each_word in each_paper['keywords']:
                temp_word = each_word.strip().replace('·','').lower()
                if temp_word not in paper_keywords[each_paper['id']]:
                    paper_keywords[each_paper['id']][temp_word] = 0
                try:
                    paper_keywords[each_paper['id']][temp_word] += 1
                except KeyError as ker:
                    print('[Warning] Paper',each_paper['id'], each_paper['title'],' has no keywords, skip.')
                    continue
    review_info = json.load(open(filename_review,'r'))
    for i, each_record in enumerate(review_info):
        if i == 2:
            print(each_record.keys())
        paper_reviewer[each_record['pub_id']] = each_record['experts']
    end_time = datetime.now()
    diff_time = end_time - start_time
    print('paper处理完成时间：', end_time.strftime('%Y-%m-%d %H:%M:%S'), '，耗时：',diff_time.total_seconds()/3600.0, 'hours.')
    return paper_keywords, paper_reviewer


def format_valid_data(filename):
    start_time = datetime.now()
    valid_papers = json.load(open(filename, 'r'))
    print(valid_papers[0].keys())
    valid_paper_keywords = {}
    for i, each_paper in enumerate(valid_papers):
        if each_paper['id'] not in valid_paper_keywords:
            valid_paper_keywords[each_paper['id']] = {}
        
        if 'keywords' in each_paper and each_paper['keywords'] != None and len(each_paper['keywords']) > 0:
            for each_word in each_paper['keywords']:
                temp_word = each_word.strip().replace('·','').lower()
                if temp_word not in valid_paper_keywords[each_paper['id']]:
                    valid_paper_keywords[each_paper['id']][temp_word] = 0
                try:
                    valid_paper_keywords[each_paper['id']][temp_word] += 1
                except KeyError as ker:
                    print('[Warning] Paper',each_paper['id'], each_paper['title'],' has no keywords, skip.')
                    continue
    end_time = datetime.now()
    diff_time = end_time - start_time
    print('paper处理完成时间：', end_time.strftime('%Y-%m-%d %H:%M:%S'), '，耗时：',diff_time.total_seconds()/3600.0, 'hours.')
    return valid_paper_keywords


def generate_dataset_for_graph_learning(reviewer_keywords, reviewer_paper, paper_keywords, paper_reviewer, valid_paper_keywords):
    entity_count = 0 
    entities_dict = {}
    train_dataset = []
    valid_dataset = []
    predict_set = []
    print('analyzing reviewer_keywords...')
    for each_reviewer in tqdm(reviewer_keywords):
        if 'r_'+each_reviewer not in entities_dict:
            entities_dict['r_'+each_reviewer] = entity_count
            entity_count += 1
        for each_keywords in reviewer_keywords[each_reviewer]:
            if 'k_'+each_keywords not in entities_dict:
                entities_dict['k_'+each_keywords] = entity_count
            entity_count += 1
            train_dataset.append(('r_'+each_reviewer, 'reviewer_keywords', 'k_'+each_keywords))
    print('analyzing reviewer_paper...')
    #这里是reviewer发表过哪些文章
    for each_reviewer in tqdm(reviewer_paper):
        if 'r_'+each_reviewer not in entities_dict:
            entities_dict['r_'+each_reviewer] = entity_count
            entity_count += 1
        for each_paper in reviewer_paper[each_reviewer]:
            if 'p_'+each_paper not in entities_dict:
                entities_dict['p_'+each_paper] = entity_count
            entity_count += 1
            train_dataset.append(('r_'+each_reviewer, 'reviewer_paper', 'p_'+each_paper))
    #
    print('analyzing paper_keywords...')
    for each_paper in tqdm(paper_keywords):
        if 'p_'+each_paper not in entities_dict:
            entities_dict['p_'+each_paper] = entity_count
            entity_count += 1
        for each_keywords in paper_keywords[each_paper]:
            if 'k_'+each_keywords not in entities_dict:
                entities_dict['k_'+each_keywords] = entity_count
            entity_count += 1
            train_dataset.append(('p_'+each_paper, 'paper_keywords', 'k_'+each_keywords))
    print('analyzing paper_reviewer...')
    # 这里是给每个paper分配reviewer的记录，需要抽取验证集
    for each_paper in tqdm(paper_reviewer):
        if 'p_'+each_paper not in entities_dict:
            entities_dict['p_'+each_paper] = entity_count
            entity_count += 1
        for each_reviewer in paper_reviewer[each_paper]:
            if 'r_'+each_reviewer not in entities_dict:
                entities_dict['r_'+each_reviewer] = entity_count
            entity_count += 1
            dice = random.random()
            if dice >= 0.8:
                train_dataset.append(('p_'+each_paper, 'paper_reviewer', 'r_'+each_reviewer))
            else:
                valid_dataset.append(('p_'+each_paper, 'paper_reviewer', 'r_'+each_reviewer))
    print('analyzing valid_paper_keywords...')
    for each_paper in tqdm(valid_paper_keywords):
        if 'p_'+each_paper not in entities_dict:
            entities_dict['p_'+each_paper] = entity_count
            entity_count += 1
        for each_keywords in valid_paper_keywords[each_paper]:
            if 'k_'+each_keywords not in entities_dict:
                entities_dict['k_'+each_keywords] = entity_count
            entity_count += 1
            train_dataset.append(('p_'+each_paper, 'paper_keywords', 'k_'+each_keywords))
        predict_set.append('p_'+each_paper)
    
    #开始写入
    with(open(fn_dataset_entity,'w')) as file_writer:
        for each_key in entities_dict:
            file_writer.write(each_key+'    '+str(entities_dict[each_key]) + '\n')
    with(open(fn_dataset_relation,'w')) as file_writer:
        file_writer.write('reviewer_keywords    0')
        file_writer.write('reviewer_paper    1')
        file_writer.write('paper_keywords    2')
        file_writer.write('paper_reviewer    3')
    with(open(fn_dataset_train,'w')) as file_writer:
        for each_record in train_dataset:
            file_writer.write(each_record[0]+'    '+ each_record[1] + '  ' + each_record[2] + '\n')
    with(open(fn_dataset_train,'w')) as file_writer:
        for each_record in valid_dataset:
            file_writer.write(each_record[0]+'    '+ each_record[1] + '  ' + each_record[2] + '\n')
    with(open(fn_dataset_predict,'w')) as file_writer:
        for each_record in predict_set:
            file_writer.write(each_record+'\n')





if __name__ == '__main__':
    
    if os.path.exists(fn_output_reviewer_keywords) and os.path.exists(fn_output_reviewer_paper):
        reviewer_keywords = pickle.load(open(fn_output_reviewer_keywords, 'rb'))
        reviewer_paper = pickle.load(open(fn_output_reviewer_paper, 'rb'))
    else:
        reviewer_keywords, reviewer_paper = format_reviewer_keywords_edition(fd_source_data+fn_expert_attributes, 0, 4)
        pickle.dump(reviewer_keywords, open(fn_output_reviewer_keywords, 'wb'))
        pickle.dump(reviewer_paper, open(fn_output_reviewer_paper, 'wb'))
    
    if os.path.exists(fn_output_paper_keywords) and os.path.exists(fn_output_paper_reviewer):
        paper_keywords = pickle.load(open(fn_output_paper_keywords, 'rb'))
        paper_reviewer = pickle.load(open(fn_output_paper_reviewer, 'rb'))
    else:
        paper_keywords, paper_reviewer = format_paper(fd_source_data+fn_paper_attributtes, fd_source_data+fn_train_data)
        pickle.dump(paper_keywords, open(fn_output_paper_keywords, 'wb'))
        pickle.dump(paper_reviewer, open(fn_output_paper_reviewer, 'wb'))
    
    if os.path.exists(fn_output_valid_paper_keywords):
        valid_paper_keywords = pickle.load(open(fn_output_valid_paper_keywords, 'rb'))
    else:
        valid_paper_keywords = format_valid_data(fd_source_data+fn_valid_data)
        pickle.dump(valid_paper_keywords, open(fn_output_valid_paper_keywords, 'wb'))
    
    generate_dataset_for_graph_learning(reviewer_keywords, reviewer_paper, paper_keywords, paper_reviewer, valid_paper_keywords)
    
    

