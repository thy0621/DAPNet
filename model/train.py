import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from model import GraphConvolution_DAPNet
import scipy.io as scio
from sklearn.metrics import roc_auc_score
import datetime
import os
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def read_data():
    pd_data = pd.read_csv('../data2/train_demo.csv')
    x_vec = pd_data.values[:, :]
    X_train, X_test, y_train, y_test = train_test_split(x_vec[:, 1:3052], x_vec[:, 3052:3053], test_size = 0.2,
                                                        shuffle = False)
    dis_list = pd_data.columns.values.tolist()[1:3052]
    print(sum(y_train))
    print(sum(y_test))
    return X_train, X_test, y_train, y_test, dis_list


def get_batch(x_train, y_train,batch_size):
    n = int(len(x_train) / batch_size)
    for i in range(n):
        end = min((i + 1) * batch_size, len(x_train))
        batch_x = x_train[i * batch_size:end]
        batch_y = y_train[i * batch_size:end]
        yield batch_x, batch_y, i


def evaluate(y_test, y_pre, name, thre):
    auc_socre = roc_auc_score(y_test,y_pre)
    y_pre = np.rint(np.array(y_pre) + thre)
    y_test = y_test.reshape(-1)
    y_pre = y_pre.reshape(-1)
    y_test_id = np.where(y_test > 0.8)[0]
    y_pre_id = np.where(y_pre > 0.8)[0]
    print("jiaoji:", len(set(y_test_id) & set(y_pre_id)))
    print("len y test id:", len(y_test_id))
    print("len y pre id:", len(y_pre_id))
    pre = len(set(y_test_id) & set(y_pre_id)) / len(y_pre_id)
    recall = len(set(y_test_id) & set(y_pre_id)) / len(y_test_id)
    f1 = (2 * pre * recall) / (pre + recall)
    print("%s model,precision: %0.4f,recall: %0.4f,F1 score: %0.4f, AUC score: %0.4f" % (name, pre, recall, f1, auc_socre))
    with open(result_path, 'a', encoding = 'utf-8') as f_r:
        f_r.write("jiaoji: " + str(len(set(y_test_id) & set(y_pre_id))) + '\n')
        f_r.write("len y test id: " + str(len(set(y_test_id) & set(y_pre_id))) + '\n')
        f_r.write("len y pre id: " + str(len(y_pre_id)) + '\n')
        f_r.write("%s model,precision: %0.4f,recall: %0.4f,F1 score: %0.4f, AUC score: %0.4f" % (name, pre, recall, f1, auc_socre) + '\n')



def evaluate_mid(y_test, y_pre, name, thre):
    auc_socre = roc_auc_score(y_test,y_pre)
    y_pre = np.rint(np.array(y_pre) + thre)
    y_test = y_test.reshape(-1)
    y_pre = y_pre.reshape(-1)
    y_test_id = np.where(y_test > 0.8)[0]
    y_pre_id = np.where(y_pre > 0.8)[0]
    pre = len(set(y_test_id) & set(y_pre_id)) / len(y_pre_id)
    recall = len(set(y_test_id) & set(y_pre_id)) / len(y_test_id)
    f1 = (2 * pre * recall) / (pre + recall)


    return pre, recall, f1, auc_socre

def train():
    x_train, x_test, y_train, y_test, dis_list = read_data()
    print("y train shape:",y_train.shape)
    print("x train shape:",x_train.shape)

    # This section is only an example and only inputs one network. In the paper, the adjacency matrices of three networks are input.
    adj_co = pd.read_csv('../data/adj_matrix.csv').values[:, 1:]
    adj_gene = pd.read_csv('../data/adj_matrix.csv').values[:, 1:]
    adj_icd = pd.read_csv('../data/adj_matrix.csv').values[:, 1:]

    # This section is only an example and only inputs one network. In the paper, the embedding of three networks are input.
    features_co = np.load('../data/Graph_embeddingfull.npy')
    features_gene = np.load('../data/Graph_embeddingfull.npy')
    features_icd = np.load('../data/Graph_embeddingfull.npy')


    with open(result_path, 'a', encoding = 'utf-8') as f_r:
        f_r.write("adj co shape:"+str(adj_co.shape)+ '\n')
        f_r.write("gene icd shape:" + str(adj_gene.shape) + '\n')
        f_r.write("icd icd shape:" + str(adj_icd.shape) + '\n')

    model =GraphConvolution_DAPNet(3051, len(adj_co), 32,
                                 features_co,adj_co,features_gene,adj_gene,features_icd,adj_icd)# 第一个参数为特征维度大小，第二个参数为词向量维度


    plt_train_loss=[]

    best_epoch ,best_pre, best_recall, best_f1, best_auc = 0, 0, 0, 0, 0

    for epoch in range(1000):
        epoch_loss = []
        for batch_x, batch_y, i in get_batch(x_train, y_train,256):
            feed_dict = {model.input_x: batch_x, model.input_y: batch_y, model.tf_is_training: True}
            _, loss = model.sess.run([model.train_loss, model.cross_entropy],feed_dict=feed_dict)
            epoch_loss.append(loss)
        print("{} epoch,epoch_loss:{}".format(epoch, np.mean(epoch_loss)))
        plt_train_loss.append(epoch_loss)
        with open(result_path, 'a', encoding='utf-8') as f_r:
            f_r.write("{} epoch,epoch_loss:{}".format(epoch, np.mean(epoch_loss))+'\n')

        result = model.sess.run(model.prediction,
                                feed_dict={model.input_x: x_test, model.input_y: y_test, model.tf_is_training: False})
        pre_, recall_, f1_, auc_socre_ = evaluate_mid(y_test, result, model_name, 0.0)

        if f1_ > best_f1:
            scio.savemat(bp + model_name + '.mat', {'ypre': result , 'y_test': y_test, 'best_epoch':epoch})
            best_epoch = epoch + 1
            best_pre = pre_
            best_recall = recall_
            best_f1 = f1_
            best_auc = auc_socre_
            print('best_epoch =', best_epoch, 'best_pre = ', '%.4f' % best_pre,
                  'best_recall = ','%.4f' % best_recall,
                  'best_f1 = ', '%.4f' % best_f1, 'best_auc = ', '%.4f' % best_auc)

        if epoch > best_epoch+300: break

    DPnet_result = scio.loadmat(bp + model_name + '.mat')['ypre'].reshape(-1)

    flag_ypre = 0
    y_test_list, result_list = [] , []
    for i in range(len(DPnet_result)):
        if flag_ypre % 10000 == 0:
            print(DPnet_result[i], y_test[i])
        result_list.append(DPnet_result[i])
        y_test_list.append(y_test[i][0])
        flag_ypre += 1

    evaluate(y_test, DPnet_result, model_name, 0.0)
    pd_pre = pd.DataFrame(x_test, columns=dis_list)
    pd_pre['pre'] = result_list
    pd_pre['result_check'] = y_test_list
    pd_pre.to_csv(bp + model_name + '_pre' + '.csv', index = False)

    fig = plt.figure()
    plt.plot(range(len(plt_train_loss)), plt_train_loss, 'b-', label='loss')
    plt.ylim(0, 1)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.savefig(bp + model_name + ".png", dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    bp = bp_base ='../result/'
    model_name = 'J002_3'+'cnn_gcl_mlp'
    result_path= bp + 'result_'+model_name+'.txt'
    start_time = datetime.datetime.now()
    with open(result_path, 'a', encoding = 'utf-8') as f_r:
        f_r.write(str(datetime.datetime.now()) + '\n')
        f_r.write("***************" + model_name + " new model******************" + '\n')
    train()
    end_time = datetime.datetime.now()
    delta = end_time - start_time
    with open(result_path, 'a', encoding = 'utf-8') as f_r:
        f_r.write("*************** cost time: " + str(delta) + " ***********************" + '\n')
        f_r.write("***************" + model_name + " end ***********************" + '\n'+ '\n')
    print(delta)



