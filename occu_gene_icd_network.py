import pandas as pd
import numpy as np
import re
import pickle
import os
from tqdm import tqdm

def read_x_vec_data(new_str):
    with open('../data2024/co_dis_list_' + new_str + '.pkl', 'rb') as file:
        feature_name = pickle.load(file)
    return feature_name


#根据网络构建邻接矩阵和特征矩阵
def construct_adj_wordvec(feature_name,network_path,xfeature_path,adj_path):#构建网络_network
    #'../data_gcn/total_network.csv'
    network = pd.read_csv(network_path).values[:,1:]
    # disease_icd = pd.read_csv(network_path)[['source','target']].values
    # disease_icd = pd.read_csv(network_path)[['icd1', 'icd2']].values
    # disease_icd = list(set(disease_icd.reshape(-1)))

    disease_icd = feature_name.copy()
    print("disease icd number :",len(disease_icd))

    icd_name = {}
    icd10_3 = pd.read_excel('../data/3位代码类目表（ICD-10）.xls').values
    icd10_4 = pd.read_excel('../data/4位代码亚目表（ICD-10）.xls').values
    for line in icd10_3:
        if line[1] is np.nan:
            continue
        icd_name[line[0]] = line[1]
    for line in icd10_4:
        if line[1] is np.nan:
            continue
        icd_name[line[0]] = line[1]
    dict_wordvec = {}

    for key in tqdm(disease_icd):
        if key in icd_name:
            dict_wordvec[key] = re.sub(r'\s','',icd_name[key])
        else:
            if key[-1]=='0' and key[0:3] in icd_name:
                # print(icd_name[key[0:3]])
                dict_wordvec[key] = re.sub(r'\s','',icd_name[key[0:3]])
            else:
                dict_wordvec[key] = '无'
    #读取glove词向
    dict_glove = {}
    with open('../data/vectors.txt','r',encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(' ')
            key = line[0]
            value = line[1:]
            dict_glove[key] = np.array(value,dtype=float)
    dict_icd_wordvec = {}
    for key,value in dict_wordvec.items():
        i = 0
        vec = np.zeros(200)
        for s in value:
            if s == '':
                vec += np.random.rand(1, 200)
                i += 1
            elif s in dict_glove:
                vec+=dict_glove[s]
                i+=1
        vec = vec/i
        dict_icd_wordvec[key] = list(vec)
    print("length dict_icd_wordvec:",len(dict_icd_wordvec))
    #构建特征矩阵，和邻接矩阵
    # icd_list = []
    # for key in dict_icd_wordvec:
    #     icd_list.append(key)
    # icd_list.sort()  # 按字典顺序升序排列
    icd_list = feature_name.copy()
    dict_icd_num = {key:i for i,key in enumerate(icd_list)}
    adj = np.zeros((len(icd_list),len(icd_list)),dtype=float)
    for line in network:
        id1 = dict_icd_num[line[0]]
        id2 = dict_icd_num[line[1]]
        value = line[2]
        adj[id1][id2] = 1
        adj[id2][id1] = 1
    df = pd.DataFrame(adj,columns=icd_list)
    df.to_csv(adj_path, index=False)
    features = []
    for key,line in dict_icd_wordvec.items():
        features.append(list(line))
    feature_name = ['f'+str(i) for i in range(200)]
    df2 = pd.DataFrame(features,columns=feature_name)
    df2.to_csv(xfeature_path, index=False)


def network_filter(network_path,feature_name,out_path):
    network = pd.read_csv(network_path).values[:,1:]
    new_network = []
    for line in tqdm(network):
        if line[0] in feature_name and line[1] in feature_name:
            new_network.append(line)
    print("node number:",len(new_network))
    df = pd.DataFrame(new_network,columns=['source','target','weight'])
    df.to_csv(out_path)


def statistic_weight(network_path):
    '''统计网络属性'''
    adj = pd.read_csv(network_path).values[:,:]
    print("adj shape:",adj.shape)
    print("adj shape:",np.sum(adj))


def icd_statistic(path):
    network = pd.read_csv(path,engine='python').values[:,:]
    print("网络中边数量",network.shape)#
    node_number = set(network[:,0:2].reshape(-1))
    print("node number:",len(node_number))


def network_merge(path1,path2,path3,path4):
    feature_name = pd.read_csv(path1).columns.values[:]
    co_network = pd.read_csv(path1).values[:,:]
    gene_network = pd.read_csv(path2).values[:,:]
    icd_network = pd.read_csv(path3).values[:,:]

    total_network = co_network+gene_network+icd_network
    total_network[total_network!=0] = 1 #将整个网络中归一化

    df = pd.DataFrame(total_network,columns=feature_name)
    df.to_csv(path4, index=False)


def start(pre_dis_list):
    print('start !')
    bp = '../data2024/'
    new_str = '_'.join(pre_dis_list).replace('.', '')
    file_path = bp + new_str + '/'
    if not os.path.exists(file_path):
        os.mkdir(file_path)
    icd_list = read_x_vec_data(new_str)
    basic_icd_net_path = '../data/gcn_icd_network.csv'
    basic_gene_net_path = '../data/gene_network.csv'
    basic_co_net_path = '../data/total_network.csv'

    new_icd_net_path  = file_path + new_str + '_icd' + '_network_' + '20240617.csv'
    new_gene_net_path = file_path + new_str + '_gene' + '_network_' + '20240617.csv'
    new_co_net_path = file_path + new_str + '_co' + '_network_' + '20240617.csv'

    path1 = file_path + new_str + '_adj_matrix_co_net.csv'
    path2 = file_path + new_str + '_adj_matrix_gene_net.csv'
    path3 = file_path + new_str + '_adj_matrix_icd_net.csv'
    path4 = file_path + new_str + '_merge_network.csv'

    network_filter(basic_icd_net_path, icd_list, new_icd_net_path)
    network_filter(basic_gene_net_path, icd_list, new_gene_net_path)
    network_filter(basic_co_net_path, icd_list, new_co_net_path)
    print('network_filter over !')
    print('construct_adj_wordvec start !')
    construct_adj_wordvec(icd_list, new_co_net_path,
                          file_path + new_str + '_features_co_net.csv',
                          path1)
    construct_adj_wordvec(icd_list, new_gene_net_path,
                          file_path + new_str + '_features_gene_net.csv',
                          path2)
    construct_adj_wordvec(icd_list, new_icd_net_path,
                          file_path + new_str + '_features_icd_net.csv',
                          path3)
    print('construct_adj_wordvec over !')
    print('statistic_weight start !')
    statistic_weight(path1) #统计
    statistic_weight(path3)
    statistic_weight(path2)

    icd_statistic(new_icd_net_path) #统计
    icd_statistic(new_gene_net_path)
    icd_statistic(new_co_net_path)








'''
build_features_data(['N17.9'])
'''
