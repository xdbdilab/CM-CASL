from sklearn.svm import SVC
import pandas as pd
import pickle
import numpy as np
import os
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import time
from sklearn.metrics import accuracy_score
import math


class svm_ssl(object):
    def __init__(self, split_number_u, system, s_number):
        self.split_number_u = self.split_number_l = split_number_u
        self.system = system
        self.s_number = 10 

        self.test_number = 50
        self.flag = True
        self.SVM1 = SVC(random_state=0, C=1.0, kernel='rbf', class_weight='balanced', probability=True)
        self.SVM2 = SVC(random_state=0, C=1.0, kernel='rbf', class_weight='balanced', probability=True)
        # self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
        #                                '{}/svm_ssl_model.pickle'.format(system))
        # self.scalar_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
        #                                       '{}/svm_ssl_scalar.pickle'.format(system))

        self.test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           '{}/{}_test_data.csv'.format(system, system))
        self.data_L_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        '{}/{}/{}_{}_data_L.csv'.format(system, s_number, system, self.split_number_u))

        self.data_U_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        '{}/{}/{}_{}_data_U.csv'.format(system, s_number, system, self.split_number_u))

    def data_in(self, data):
        columns = list(data.columns)
        data = data.drop('label', 1)
        # print(data)
        return data

    def fit_svm1(self, data, model):
        Y = data.iloc[:, -1].values.astype(int)
        if len(np.unique(Y)) == 1:
            data = pd.concat((data, data.iloc[-1:, :]), axis=0)
            Y = data.iloc[:, -1].values.astype(int)
            Y[-1] = abs(Y[-1] - 1)
        X = self.data_in(data)
        X = X.values
        scalar = StandardScaler()
        X = scalar.fit_transform(X)
        model = model.fit(X, Y)
        return model, scalar

    def fit_svm2(self, data, model):
        Y = data.iloc[:, -1].values.astype(int)
        if len(np.unique(Y)) == 1:
            data = pd.concat((data, data.iloc[-1:, :]), axis=0)
            Y = data.iloc[:, -1].values.astype(int)
            Y[-1] = abs(Y[-1] - 1)

        X = self.data_in(data)
        X = X.values
        scalar = StandardScaler()
        X = scalar.fit_transform(X)
        model = model.fit(X, Y)
        return model, scalar

    def one_train(self, svm1, svm2, data_L, data_U, data_S):
        # permutation = np.random.permutation(len(data_L))

        data_L1 = data_L.iloc[:int(len(data_L) / 2), :]
        data_L2 = data_L.iloc[int(len(data_L) / 2):, :]
        svm1_train_data = pd.concat((data_L1, data_S), axis=0)
        svm1_train_data = svm1_train_data.reset_index(drop=True)

        svm1, scalar1 = self.fit_svm1(svm1_train_data, svm1)
        svm2, scalar2 = self.fit_svm2(data_L2, svm2)

        U_and_S_data = pd.concat((data_U, data_S), axis=0)
        U_and_S_data = U_and_S_data.reset_index(drop=True)

        test_data_process = self.data_in(U_and_S_data)
        test_data_process1 = scalar1.transform(test_data_process)
        test_data_process2 = scalar2.transform(test_data_process)

        predict1 = svm1.predict(test_data_process1)
        predict2 = svm2.predict(test_data_process2)
        probility2 = svm1.predict_proba(test_data_process1)
        probility2 = np.array([max(i) for i in probility2])

        index_new_S = []
        i = 0
        for x, y in zip(list(predict1), list(predict2)):
            if x == y:
                index_new_S.append(i)
            i += 1

        data_S = U_and_S_data.iloc[index_new_S, :] 
        probility2 = probility2[index_new_S]

        # print(data_S)
        predict_values = predict1[index_new_S]
        true_value = list(data_S.iloc[:, -1])

        data_S = data_S.iloc[:, :-1]
        data_S['label'] = np.array(predict_values)

        data_S = data_S.reset_index(drop=True)
        if len(data_S) > self.s_number and self.flag:
            # index = np.argpartition(probility2, len(probility2) - self.s_number)
            # index = index[-self.s_number:]
            # list_choose_s = index
            index = np.argsort(probility2)
            list_choose_s = index[-self.s_number:]
        else:
            list_choose_s = range(len(data_S))

        data_S = data_S.iloc[list_choose_s, :]
        data_S = data_S.reset_index(drop=True)

        dengyu = sum(predict_values[list_choose_s] == np.array(true_value)[list_choose_s])
        index_new_S = [index_new_S[i] for i in list_choose_s] 
        data_U = U_and_S_data.drop(index=index_new_S) 
        data_U = data_U.reset_index(drop=True)
        data_L = data_L.reset_index(drop=True)

        return svm1, svm2, data_L, data_U, data_S, scalar1

    def train(self):
        data_L = pd.read_csv(self.data_L_path)
        # data_L = data_L.sample(frac=split_number).reset_index(drop=True)
        data_U = pd.read_csv(self.data_U_path)
        # data_U = data_U.sample(frac=split_number).reset_index(drop=True)

        final_data_l = data_L.iloc[:, :]

        test_data = pd.read_csv(self.test_data_path)

        data_S = pd.DataFrame(columns=data_L.columns)
        epoch = 0

        while (not data_U.empty) and epoch < 10:
            self.SVM1, self.SVM2, data_L, data_U, data_S, scalar1 = self.one_train(self.SVM1, self.SVM2, data_L, data_U,
                                                                                   data_S)
            epoch += 1
            self.test(self.SVM1, test_data, scalar1)

        svm1_train_data = pd.concat((data_L, data_S), axis=0)
        svm1_train_data = svm1_train_data.reset_index(drop=True)

        svm1, scalar = self.fit_svm1(svm1_train_data, self.SVM1)

        accuracy, rank_avg, rd_loss_max, rd_loss_min = self.test(svm1, test_data, scalar)
        # return svm1, test_data, scalar
        return accuracy, rank_avg, rd_loss_max, rd_loss_min

    def rank_test(self, predict, true_y, test_number):
        predict_rank = {}
        true_rank = {}
        for i in range(test_number):
            predict_rank[i] = [0, 0]
            true_rank[i] = [0, 0]

        number_index = 0
        for i in range(test_number):
            for j in range(i + 1, test_number):
                if predict[number_index] == 1:
                    predict_rank[i][1] += 1
                    predict_rank[j][0] += 1
                else:
                    predict_rank[i][0] += 1
                    predict_rank[j][1] += 1

                if true_y[number_index] == 1:
                    true_rank[i][1] += 1
                    true_rank[j][0] += 1
                else:
                    true_rank[i][0] += 1
                    true_rank[j][1] += 1
                number_index += 1

        rank_avg = 0
        rd_loss_max = 0
        rd_loss_min = 0

        for i in range(len(predict_rank)):
            rank_avg += abs(predict_rank[i][0] - true_rank[i][0])
        rank_avg = rank_avg / len(predict_rank)

        for i in range(len(predict_rank)):
            if true_rank[i][1] == 0:
                rd_loss_max = abs(predict_rank[i][0] - true_rank[i][0])
                break

        for i in range(len(predict_rank)):
            if true_rank[i][0] == 0:
                rd_loss_min = abs(predict_rank[i][0] - true_rank[i][0])
                break

        return rank_avg, rd_loss_max, rd_loss_min

    def test(self, model, testdata, scalar):
        y = testdata.iloc[:, -1].values.astype(int)

        testdataset = self.data_in(testdata)
        testdataset = testdataset.reset_index(drop=True)

        X = scalar.transform(testdataset.values)
        y_pred = model.predict(X)
        rank_avg, rd_loss_max, rd_loss_min = self.rank_test(y_pred, y, self.test_number)
        return accuracy_score(y, y_pred), rank_avg, rd_loss_max, rd_loss_min


def system_samplesize(sys_name):

    N_train_all = np.multiply(6, [1, 2, 3])

    return N_train_all


def out_dict_info():
    systerm_list = ['hadoopsort', 'hadoopterasort', 'hadoopwordcount', 'mysql',
                    'redis', 'sparksort', 'sparkterasort', 'sparkwordcount', 'sqlite', 'tomcat', 'x264']

    s_number_list = [2, 3, 4]

    class multidict(dict):
        def __getitem__(self, item):
            try:
                return dict.__getitem__(self, item)
            except KeyError:
                value = self[item] = type(self)()
                return value

    dict_info = multidict()

    for s_number in s_number_list:
        for system in systerm_list:
            number_list = system_samplesize(system)
            for number in number_list:

                if number % s_number != 0:
                    half = math.floor(number / s_number)
                    Q_number = (number - math.floor(number / s_number)) * 20
                else:
                    half = number / s_number
                    Q_number = math.ceil(number / s_number) * (s_number - 1) * 20

                N = (half * (half - 1) // 2) + Q_number
                first = math.ceil(math.sqrt(N * 2))
                if first > number:
                    add_ = number - half
                    sub_ = (number) * (number - 1) // 2
                else:
                    if first * (first - 1) // 2 < N:
                        first += 1

                        add_ = first - half
                        if first * (first - 1) // 2 == N:
                            sub_ = first * (first - 1) // 2
                        else:
                            sub_ = first * (first - 1) // 2 - N
                            sub_ = - sub_
                    elif first * (first - 1) // 2 == N:
                        add_ = first - half
                        sub_ = first * (first - 1) // 2
                    else:
                        add_ = first - half
                        if first * (first - 1) // 2 == N:
                            sub_ = first * (first - 1) // 2
                        else:
                            sub_ = first * (first - 1) // 2 - N
                            sub_ = - sub_
                dict_info[s_number][system][Q_number][number] = [sub_, add_]
    return dict_info

if __name__ == '__main__':

    dict_data = out_dict_info()
    systerm_list = ['hadoopsort', 'hadoopterasort', 'hadoopwordcount', 'sparksort', 'sparkterasort', 'sparkwordcount',
                    'mysql', 'redis',
                    'x264', 'tomcat', 'sqlite']
    s_number_list = [2, 3, 4]

    columns = ['Systerm', 'Split_NUMBER', 'NUMBER', 'accuracy', 'rank_avg',
               'rd_loss_max', 'rd_loss_min']
    out = pd.DataFrame(columns=columns)
    out.to_csv('svm_ssl.csv', index=False)

    for system in systerm_list:
        for s_number in s_number_list:
            number_list = system_samplesize(system)
            for number in number_list:
                if number==6:
                    continue
                print("---{}---{}----{}-".format(system, s_number, number))
                accuracy, rank_avg, rd_loss_max, rd_loss_min = svm_ssl(number, system, s_number).train()
                row = [
                    [system, s_number, number, accuracy, rank_avg, rd_loss_max,
                     rd_loss_min]]
                out = pd.DataFrame(row)
                out.to_csv('svm_ssl.csv', index=False, mode='a+', header=None)
                print("----------------------")
