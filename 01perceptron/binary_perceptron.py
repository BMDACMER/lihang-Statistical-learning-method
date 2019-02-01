import numpy as np
import pandas as pd
import time

from sklearn.cross_validation import  train_test_split
from sklearn.metrics import accuracy_score

class Perceptron(object):

    def __init__(self):
        self.learning_step = 0.00001
        self.max_iteration = 5000

    def predict_(self, x):
        wx = np.sum(self.w[j] * x[j] for j in range(len(self.w)))
        return int( wx > 0)

    def train(self, features, labels):
        # 初始化w   注意：这里将b也加进去w里了
        self.w = [0.001] * (len(features[0]) + 1)

        correct_count = 0
        iter = 0

        while iter < self.max_iteration:
            # 在python中的random.randint(a,b)用于生成一个指定范围内的整数。
            # 其中参数a是下限，参数b是上限，生成的随机数n: a <= n <= b。
            index = np.random.randint(0, len(labels) - 1)
            x = list(features[index])
            # 将bais加入到 x 矩阵中去
            x.append(1.0)
            # labels 取值为0和1    执行以下操作将labels转换为-1 或者 1 的取值
            # 此目的有助于预测准确性的提高
            y = 2 * labels[index] - 1
#            wx = np.sum(self.w * x)
            wx = np.sum([self.w[j] * x[j] for j in range(len(self.w))])

            if wx * y > 0:
                correct_count += 1
                if correct_count > self.max_iteration:
                    break
                continue
            # 错误分类，则更新
            for i in range(len(self.w)):
                self.w[i] += self.learning_step * (y * x[i])


    def predict(self, features):
        labels = []
        for feature in features:
            x = list(feature)
            x.append(1)
            labels.append(self.predict_(x))
        return labels

if __name__ == '__main__':

    print("start read data...")
    start = time.time()
    raw_data = pd.read_csv('../data/train_binary.csv', header=0)
    data = raw_data.values

    imgs = data[0::, 1::]
    labels = data[::, 0]

    # train_set : test_set = 2 : 1
    train_features, test_features, train_labels, test_labels = train_test_split(
        imgs, labels, test_size=0.33, random_state=1314
    )

    end = time.time()
    print("read data cost", end - start, " second")

    print("start training")
    p = Perceptron()
    p.train(train_features, train_labels)
    end2 = time.time()
    print("training cost ",end2 - end, " second" )

    print("start predicting")
    test_predict = p.predict(test_features)
    end3 = time.time()
    print("predicting cost ", end3 - end2, " second")

    score = accuracy_score(test_labels, test_predict)
    print("The accruacy score is ", score)

