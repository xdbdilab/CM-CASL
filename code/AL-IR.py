from sklearn.svm import SVC
import pandas as pd
import pickle
import numpy as np
import os
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
import random
import math

from sklearn.cluster import KMeans


class svm_al(object):
    def __init__(self, split_number_u, system, expert_exact_percent, q_number, s_number):
        self.split_number_u = self.split_number_l = split_number_u 
        self.system = system  

        self.RSTRICT_NUMBER = q_number 
        self.expert_exact_percent = expert_exact_percent

        self.Q_number = 10 
        self.test_number = 50
        self.cluster_number = 20
        # self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
        #                                '{}/svm_al_model.pickle'.format(system))
        # self.scalar_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
        #                                       '{}/svm_al_scalar.pickle'.format(system))
        self.data_L_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                        '{}/{}/{}_{}_data_L.csv'.format(system, s_number, system, self.split_number_u))

        self.data_U_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                        '{}/{}/{}_{}_data_U.csv'.format(system, s_number, system, self.split_number_u))

        self.test_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                           '{}/{}_test_data.csv'.format(system, system))
        self.SVM1 = SVC(random_state=0, C=1.0, kernel='rbf', class_weight='balanced', probability=True)

    def data_in(self, data, ):
        columns = list(data.columns)
        data = data.drop('label', 1)
        # print(data)
        return data

    def fit_svm1(self, data, model):
        Y = data.iloc[:, -1].values.astype(int)
        if len(np.unique(Y)) == 1:
            if np.unique(Y)[0] == 1:
                Y[-1] = 0
            else:
                Y[-1] = 1

        X = self.data_in(data)
        # X = data
        X = X.values
        scalar = StandardScaler()
        X = scalar.fit_transform(X)
        model = model.fit(X, Y)
        return model, scalar

    def Euclidean(self, point1, clusters):
        dist = filter(lambda x: np.linalg.norm(point1 - x), clusters)
        return dist

    def train_cluster(self, data_u):
        cluster_model = KMeans(n_clusters=self.cluster_number).fit(data_u)
        clusters = cluster_model.cluster_centers_
        probility = filter(lambda x: self.Euclidean(x, clusters), data_u)
        probility = np.array([min(i) for i in probility])
        if len(probility) <= self.cluster_number:
            index = range(len(probility))
        else:
            index = np.argpartition(probility, self.cluster_number)

            index = index[:self.cluster_number]
        return index

    def one_train(self, svm1, data_L, data_U, expert_exact_percent):
        svm1, scalar1 = self.fit_svm1(data_L, svm1)
        data_U_process = self.data_in(data_U)

        data_U_process = scalar1.transform(data_U_process)
        index_20 = self.train_cluster(data_U_process)
        data_cluster_20 = data_U_process[index_20]

        probility = svm1.predict_proba(data_cluster_20)
        probility = np.array([max(i) for i in probility])
        if len(probility) <= self.Q_number:
            index = range(len(probility))
        else:
            index = np.argpartition(probility, self.Q_number)

            index = index[:self.Q_number]

        Q_dataset = data_U.iloc[index_20[index], :]
        Q_dataset = Q_dataset.reset_index(drop=True)
        # print(Q_dataset)
        true_y_0 = np.sum(Q_dataset.iloc[:, -1] == 0)
        true_y_1 = np.sum(Q_dataset.iloc[:, -1] == 1)

        true_number = len(Q_dataset)
        for i in range(len(Q_dataset)):
            random_number = random.uniform(0, 1)
            if random_number > expert_exact_percent:
                Q_dataset.iloc[i, -1] = abs(1 - Q_dataset.iloc[i, -1])
                true_number -= 1

        predict_label = svm1.predict(Q_dataset.iloc[:, :-1].values)
        # print(predict_label)
        predict_y_0 = np.sum(predict_label == 0)
        predict_y_1 = np.sum(predict_label == 1)

        cha = np.sum(predict_label == Q_dataset.iloc[:, -1])
            expert_exact_percent,
            true_number / len(Q_dataset),
            np.sum(Q_dataset.iloc[:, -1] == 0),
            np.sum(Q_dataset.iloc[:, -1] == 1),
            true_y_0,
            true_y_1, predict_y_0, predict_y_1, cha
            ))
        # print()

        data_L = pd.concat((data_L, Q_dataset), axis=0)  
        data_U = data_U.drop(index=index_20[index])
        data_U = data_U.reset_index(drop=True)
        data_L = data_L.reset_index(drop=True)

        return svm1, data_L, data_U, scalar1

    def train(self):
        data_L = pd.read_csv(self.data_L_path)
        # data_L = data_L.sample(frac=split_number).reset_index(drop=True)

        data_U = pd.read_csv(self.data_U_path)
        # data_U = data_U.sample(frac=split_number).reset_index(drop=True)

        test_data = pd.read_csv(self.test_data_path)

        restrict_number = 0

        epoch = 0

        while restrict_number < self.RSTRICT_NUMBER and not data_U.empty

            self.SVM1, data_L, data_U, scalar1 = self.one_train(self.SVM1, data_L, data_U, self.expert_exact_percent)
            restrict_number += self.Q_number

            epoch += 1
            self.test(self.SVM1, test_data, scalar1)
        self.SVM1, scalar1 = self.fit_svm1(data_L, self.SVM1)
            accuracy_score_, rank_avg, rd_loss_max, rd_loss_min = self.test(self.SVM1, test_data, scalar1)
            return accuracy_score_, rank_avg, rd_loss_max, rd_loss_min

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

    expert_exact_percent_list = [1, 0.9, 0.8, 0.7]

    columns = ['expert_exact_percent', 'Systerm', 'Split_NUMBER', 'expert_num', 'NUMBER', 'accuracy', 'rank_avg',
               'rd_loss_max', 'rd_loss_min']
    out = pd.DataFrame(columns=columns)
    out.to_csv('svm_al.csv', index=False)

    for expert_exact_percent in expert_exact_percent_list:
        for system in systerm_list:
            for s_number in s_number_list:
                    number_list = system_samplesize(system)
                    for number in number_list:
                        if number == 6:
                            continue
                        q_number = (number - math.floor(number / s_number)) * 20  
                        accuracy, rank_avg, rd_loss_max, rd_loss_min = svm_al(number, system, expert_exact_percent,
                                                                              q_number, s_number).train()

                        row = [
                            [expert_exact_percent, system, s_number, q_number, number, accuracy, rank_avg, rd_loss_max,
                             rd_loss_min]]
                        out = pd.DataFrame(row)
                        out.to_csv('svm_al_cluster.csv', index=False, mode='a+', header=None)
                        print("----------------------")
