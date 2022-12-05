from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
import pandas as pd
import pickle
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score
import math


class svm_our(object):
    def __init__(self, split_number_u, system, q_number, SPLIT_NUMBER):
        self.split_number_u = self.split_number_l = split_number_u 
        self.system = system 

        self.SVM = SVC(random_state=0, C=1.0, kernel='rbf', class_weight='balanced')
        self.test_number = 50
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '{}/svm_model.pickle'.format(system))
        self.scalar_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              '{}/svm_scalar.pickle'.format(system))

        self.test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           '{}/{}_test_data.csv'.format(system, system))

        self.svm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     '{}/{}/{}/{}_svm_{}_data_U.csv'.format(system, SPLIT_NUMBER, q_number, system,
                                                                            self.split_number_u))

    def data_in(self, data):
        columns = list(data.columns)
        data = data.drop('label', 1)
        # print(data)
        return data

    def fit_svm1(self, data, model):
        Y = data.iloc[:, -1].values.astype(int)
        X = self.data_in(data)

        SCALAR = StandardScaler()
        X = SCALAR.fit_transform(X)

        scores = cross_val_score(model, X, Y, cv=3, scoring='accuracy')
        model = model.fit(X, Y)

        y_pred = model.predict(X)
        return model, SCALAR

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

    def train(self):
        data_L = pd.read_csv(self.svm_path)

        svm1, SCALAR = self.fit_svm1(data_L, self.SVM)
        accuracy, rank_avg, rd_loss_max, rd_loss_min = self.test(svm1, SCALAR)
        return accuracy, rank_avg, rd_loss_max, rd_loss_min
        # return svm1, SCALAR

    def test(self, model, SCALAR):
        testdataset = pd.read_csv(self.test_data_path)

        y = testdataset.iloc[:, -1].values.astype(int)

        testdataset = self.data_in(testdataset)
        # testdataset = substract_(testdataset)
        X = testdataset.reset_index(drop=True)

        X = SCALAR.transform(X)
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

    columns = ['Systerm', 'Split_NUMER', 'expert_num', 'NUMBER', 'accuracy', 'rank_avg', 'rd_loss_max', 'rd_loss_min']
    out = pd.DataFrame(columns=columns)
    out.to_csv('svm.csv', index=False)

    for system in systerm_list:
        for s_number in s_number_list:
                number_list = system_samplesize(system)
                for number in number_list:

                    if number==6:
                        continue

                    q_number = (number - math.floor(number / s_number)) * 20 

                    print("---{}---{}---{}--{}-".format(system, s_number, q_number, number))
                    accuracy, rank_avg, rd_loss_max, rd_loss_min = svm_our(number, system, q_number, s_number).train()
                    row = [[system, s_number, accuracy, q_number, number, rank_avg, rd_loss_max, rd_loss_min]]
                    out = pd.DataFrame(row)
                    out.to_csv('svm.csv', index=False, mode='a+', header=None)
                    print("----------------------")
