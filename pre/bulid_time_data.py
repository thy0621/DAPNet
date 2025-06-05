import pandas as pd
import numpy as np
import json
import time
from tqdm import tqdm
import pickle
import random
def patient(new_str):
    with open('../data2024/dict_patient_'+new_str+'.pkl', 'rb') as file:
        dict_patient = pickle.load(file)
    # "SS0101_0000157": [
    #     ["SS0101_0000157_10", "2", "71", "I20.0", "I50.9", "J45.9", "K21.0", "I63.9", "M19.9", "M81.9", "K76.0"],
    #     ["SS0101_0000157_9", "2", "70", "I10", "I70.9", "M81.9", "K76.0", "K29.8", "E78.5", "K29.4", "I25.1"]]
    return dict_patient

def read_x_vec_data(new_str):
    with open('../data2024/co_dis_list_'+new_str+'.pkl', 'rb') as file:
        feature_name = pickle.load(file)
    return feature_name

def construct_icd_list(network_path):
    import re

    network = pd.read_csv(network_path).values[:,1:]
    disease_icd = pd.read_csv(network_path)[['source','target']].values
    disease_icd = list(set(disease_icd.reshape(-1)))
    print("disease icd number :",len(disease_icd))
    icd_name = {}
    icd10_3 = pd.read_excel('../data/3_ICD-10.xls').values # read icd-10
    icd10_4 = pd.read_excel('../data/4_ICD-10.xls').values
    for line in icd10_3:
        if line[1] is np.nan:
            continue
        icd_name[line[0]] = line[1]
    for line in icd10_4:
        if line[1] is np.nan:
            continue
        if line[0][-1] == '0':
            icd_name[line[0][0:3]] = line[1]
        else:
            icd_name[line[0]] = line[1]
    dict_wordvec = {}

    for key in disease_icd:# disease_icd -> chinese
        if key[-1] == '0':
            key=key[0:3]
        if key in icd_name:
            dict_wordvec[key] = re.sub(r'\s','',icd_name[key])
        else:
            if key[-1]=='0' and key[0:3] in icd_name:
                dict_wordvec[key] = re.sub(r'\s','',icd_name[key[0:3]])
            else:
                dict_wordvec[key] = 'None'

    # read glove vec
    dict_glove = {}
    with open('../data/vectors.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(' ')
            key = line[0]
            value = line[1:]
            dict_glove[key] = np.array(value,dtype=np.float)
    dict_icd_wordvec = {}
    for key,value in dict_wordvec.items():
        # Generate icd features based on the Chinese characters in the dict, 200 dimensions
        i = 0
        vec = np.zeros(200)
        for s in value:
            if s in dict_glove:
                vec+=dict_glove[s]
                i+=1
        vec = vec/i
        dict_icd_wordvec[key] = list(vec)
    print("length dict_icd_wordvec:",len(dict_icd_wordvec))
    icd_list = []
    for key in dict_icd_wordvec:
        icd_list.append(key)
    icd_list.sort()  # sort
    dict_icd_num = {key:i for i,key in enumerate(icd_list)}
    return icd_list,dict_icd_num

def new_negative_disease(pre_dis_list):
    with open('../data2024/new_negative_disease.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        dis_list=[]
        for line in lines:
            line = line.strip('\n')
            dis_list.append(line)
        random.shuffle(dis_list)
        dis_list = list(set(dis_list) - set(pre_dis_list))
    return dis_list


def record_error_correction(record_list):
    record_dict = {}
    try:
        for record_ in record_list:
            record_dict[int(record_[0].split('_')[-1])] = record_[3:]
    except Exception as e:
        print('Error:', e,)
    finally:
        record_index_list = list(record_dict.keys())
        record_index_list.sort()
        new_record_list = [record_dict[item_] for item_ in record_index_list]
    return new_record_list


def bulid_patient_record_matrix5(dict_patient, icd_list, pre_disease_list,pre_result,file_patient_record_matrix_path,
                                 file_patient_result_matrix_path):

    patient_list = list(dict_patient.keys())
    patient_list.sort()
    find_record_list=[]
    dict_icd_num = {key: i for i, key in enumerate(icd_list)}
    key_num = 0

    df_record_list = []
    df_result_list = []
    total_record_size = 0
    for key in tqdm(dict_patient):
        key_num += 1
        matrix = np.zeros(len(icd_list)).tolist()
        record_list = dict_patient[key]
        record_list = record_error_correction(record_list)
        pre_disease_Seizure_index=-1  # first appearance->index
        #find pre disease index
        flag = False
        pre_disease_name = ''
        for record_index in range(len(record_list)):
            if find_co_dis(record_list[record_index],pre_disease_list) == True:
                pre_disease_Seizure_index = record_index
                flag= True
                pre_disease_name = find_co_dis2(record_list[record_index],pre_disease_list)
                break
        if flag == False:  # Unrelated
            pass
        # elif pre_disease_Seizure_index == 0:#  appearance in 1th
        #     pass
        elif pre_disease_Seizure_index > 3:
            index_4 = 0
            disease_statastic_dict = {}
            disease_score ={0:8,1:4,2:2,3:1}
            for record_index in range(pre_disease_Seizure_index-1,-1,-1):
                record_item = record_list[record_index]
                for dis_ in record_item:
                    if dis_ in disease_statastic_dict:
                        disease_statastic_dict[dis_] += disease_score[index_4]
                    else:
                        disease_statastic_dict[dis_] = disease_score[index_4]
                index_4 += 1
                if index_4 >3:  # contain 4 records
                    break
            total_record_size += index_4
            record_num = len(record_list)
            for icd_item in disease_statastic_dict:
                if icd_item in dict_icd_num:
                    matrix[dict_icd_num[icd_item]] = disease_statastic_dict[icd_item]


            df_result_list.append([pre_result])

            find_record_list.append(key)
            df_record_list.append(matrix+[pre_disease_name])

    icd_list = list(icd_list)+['dis']
    df_record = pd.DataFrame(df_record_list,columns = icd_list)
    df_result = pd.DataFrame(df_result_list)
    df_record.to_csv(file_patient_record_matrix_path,encoding='utf_8_sig')
    df_result.to_csv(file_patient_result_matrix_path,encoding='utf_8_sig')
    return df_record,df_result,find_record_list

def network_filter(pre_dis_str,df_lung_record,df_lung_result,df_dia_record1,df_dia_result1,
                   df_dia_record2,df_dia_result2,
                   df_dia_record3,df_dia_result3,
                   df_dia_record4,df_dia_result4,data_v):
    df_dia_record = pd.concat([df_dia_record1, df_dia_record2,df_dia_record3,df_dia_record4], axis=0)
    df_dia_result = pd.concat([df_dia_result1, df_dia_result2,df_dia_result3,df_dia_result4], axis=0)
    print('df_lung_record mean',df_lung_record.select_dtypes('number').mean(1).mean()*df_lung_record.shape[1])
    print('df_dia_record mean', df_dia_record.select_dtypes('number').mean(1).mean() * df_lung_record.shape[1])

    df_lung = pd.concat([df_lung_record, df_lung_result], axis = 1)
    df_dia = pd.concat([df_dia_record, df_dia_result], axis = 1)

    df_lung = df_lung.sample(frac = 1)
    df_dia = df_dia.sample(frac = 1)
    print(df_lung.shape)
    print(df_dia.shape)
    rate_v = '111'
    # num = 10000
    df_lung_part = df_lung
    df_dia_part = df_dia.head(df_lung.shape[0])
    print(f'df_lung_record mean_{rate_v} ', df_lung_part.select_dtypes('number').mean(1).mean() * df_lung_part.shape[1])
    print(f'df_dia_record mean_{rate_v} ', df_dia_part.select_dtypes('number').mean(1).mean() * df_lung_part.shape[1])
    df_train = pd.concat([df_lung_part, df_dia_part], axis = 0)
    df_train = df_train.sample(frac = 1)
    print(df_train.shape)
    final_file = f'../data2024/{pre_dis_str}_dia_x_train_{data_v}_{rate_v}.csv'
    print(final_file)
    df_train.to_csv(final_file, index = False, encoding = 'utf_8_sig')


def find_co_dis(record_list_1,pre_disease_list):
    for dis1 in pre_disease_list:
        for dis2 in record_list_1:
            if dis1 in dis2:
                return True
    return False
def find_co_dis2(record_list_1,pre_disease_list):
    str_list = []
    for dis1 in pre_disease_list:
        for dis2 in record_list_1:
            if dis1 in dis2:
                str_list.append(dis1)
    return '|'.join(str_list)

def build_features_data(pre_dis_list):
    start_time = time.time()
    new_str = '_'.join(pre_dis_list).replace('.', '')
    dict_patient=patient(new_str)
    feature_name = read_x_vec_data(new_str)

    feature_name_ran = new_negative_disease(pre_dis_list)
    data_v = '20240615'

    end_time_1 = time.time()
    cost_time_1 =(end_time_1 - start_time) / 60
    print("Reading file takes %d minutes" % cost_time_1)
    df_pre_record,df_pre_result,pre_record_list = bulid_patient_record_matrix5(dict_patient, feature_name,
                                                                                pre_dis_list, 1,
                                                                                f'../result2024/pre_matrix_record_{new_str}_{data_v}.csv'
                                                                                ,f'../result2024/pre_matrix_result_{new_str}_{data_v}.csv')

    end_time_2 = time.time()
    cost_time_2 =(end_time_2 - end_time_1) / 60
    print("The predicted data processing time: %d minutes" % cost_time_2)
    df_ran_record1, df_ran_result1, ran_record_list1 = bulid_patient_record_matrix5(dict_patient, feature_name,
                                                                                    feature_name_ran[0:10], 0,
                                                                                    f'../result2024/ran_matrix_record_{new_str}_{data_v}.csv'
                                                                                    ,
                                                                                    f'../result2024/ran_matrix_result_{new_str}_{data_v}.csv')
    df_ran_record2, df_ran_result2, ran_record_list2 = bulid_patient_record_matrix5(dict_patient, feature_name,
                                                                                 feature_name_ran[10:20], 0,
                                                                                 f'../result2024/ran_matrix_record_{new_str}_{data_v}.csv'
                                                                                 ,
                                                                                 f'../result2024/ran_matrix_result_{new_str}_{data_v}.csv')
    df_ran_record3, df_ran_result3, ran_record_list3 = bulid_patient_record_matrix5(dict_patient, feature_name,
                                                                                 feature_name_ran[20:30], 0,
                                                                                 f'../result2024/ran_matrix_record_{new_str}_{data_v}.csv'
                                                                                 ,
                                                                                 f'../result2024/ran_matrix_result_{new_str}_{data_v}.csv')
    df_ran_record4, df_ran_result4, ran_record_list4 = bulid_patient_record_matrix5(dict_patient, feature_name,
                                                                                 feature_name_ran[30:40], 0,
                                                                                 f'../result2024/ran_matrix_record_{new_str}_{data_v}.csv'
                                                                                 ,
                                                                                 f'../result2024/ran_matrix_result_{new_str}_{data_v}.csv')
    end_time_3 = time.time()
    cost_time_3 = (end_time_3 - end_time_2) / 60
    print("The processing time for negative sample data : %d minutes" % cost_time_3)

    network_filter(new_str,df_pre_record, df_pre_result, df_ran_record1, df_ran_result1,df_ran_record2, df_ran_result2,
                   df_ran_record3, df_ran_result3,
                   df_ran_record4, df_ran_result4,data_v)


build_features_data(['N17.9'])


