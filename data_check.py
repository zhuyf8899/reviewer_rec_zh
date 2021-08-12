import json


fd_source_data = './source_data/'
fn_train_data = 'train_data_1.json'
fn_valid_data = 'valid_data.json'
fn_expert_attributes_0 = 'expert_attributes0.json'
fn_paper_attributtes = 'paper_attributes.json'


def check_paper_info(filename):
    data = json.load(open(filename))
    for i,each_record in enumerate(data):
        if i == 0:
            print(data[0].keys())
        if each_record['abstract_zh'] != '' and each_record['abstract_zh'] != None and each_record['abstract'] == '':
            print('[Chinese abs only]' + each_record['id'], each_record['title'], each_record['abstract_zh'])
        if (each_record['abstract'] == '' or each_record['abstract'] == None) and (each_record['abstract_zh'] == '' or each_record['abstract_zh'] == None):
            # print('[Empty abs]' + each_record['id'], each_record['title'], each_record['keywords'])
            pass
        if each_record['keywords'] == '' and each_record['abstract_zh']:
            print('[Empty keywords]' + each_record['id'], each_record['title'])
    return


def check_reviewer_info(filename):
    data = json.load(open(filename))
    for i,each_record in enumerate(data):
        if i == 0:
            print(data[0].keys())
        try:
            if 'interests' not in each_record or each_record['interests'] == None or len(each_record['interests']) == 0:
                if 'tags' not in each_record or len(each_record['tags']) == 0:
                    print('[Empty keywords]' + each_record['id'], each_record['name'])
            if len(each_record['pub_info']) == 0 :
                print('[Empty pubs]' + each_record['id'], each_record['name'])
        except TypeError as  err:
            print('EXCEPTION:',each_record['id'],each_record['name'])
            print(err.__traceback__)
    return


if __name__ == "__main__":
    check_paper_info(fd_source_data+fn_paper_attributtes)
    # check_reviewer_info(fd_source_data+fn_expert_attributes_0)
