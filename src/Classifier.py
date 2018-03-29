import json
import random
import math
#import numpy as np

class Classifier(object):

    def __init__(self):
        pass

    # train the classifier with training set (X_train, y_train)
    # where X_train is a N x M matrix, N = number of trainning data, M = number of feature
    # y_train a N-dimention vector, denoting the correct label
    def train(self, X_train, y_train):
        pass

    # predicting labels
    # X_test: N x M matrix
    # return: a N-dimention vector
    def predict(self, X_test):
        pass

class NaiveBayesianClassifier(Classifier):

    # default: Laplace smoothing
    # lev_of_fea: the i-th feature should take intergal value between 0 and lev_of_fea[i]
    def init(self):
         # prior distribution of label 
        self.prior = [0.] * self.num_of_label
        # conditional distribution of each feature given label
        # cond[c][i][k] = p(x_i = k | y = c)
        self.cond = []
        for j in range(self.num_of_label):
            self.cond.append([])
            for i in self.lev_of_fea:
                self.cond[j].append(([0.] * (i + 1)))
        print self.cond

    def __init__(self, num_of_label, lev_of_fea, alpha = 1.): 
        self.num_of_label = num_of_label
        self.lev_of_fea = lev_of_fea
        self.alpha = alpha

        self.prior = None
        self.cond = None
       
    def train(self, X_train, y_train):
        # to obtain prior and cond
        N = len(X_train)
        M = len(X_train[0])
        self.init()

        for i in range(len(y_train)):
            self.prior[y_train[i]] += 1
        s = len(y_train) * 1.
        for i in range(self.num_of_label):
            self.prior[i] = (self.prior[i] + self.alpha) / (s + self.num_of_label * self.alpha)

        for i in range(len(X_train)):
            for j in range(len(X_train[i])):
                self.cond[y_train[i]][j][X_train[i][j]] += 1.

        for i in range(self.num_of_label):
            for j in range(M):
                s = 0.
                for k in range(self.lev_of_fea[j]):
                    s += self.cond[i][j][k]
                for k in range(self.lev_of_fea[j]):
                    self.cond[i][j][k] = (self.cond[i][j][k] + self.alpha) / (s + self.lev_of_fea[j] * self.alpha)


    # get the unnormalized log propability log( p(y = c | x) )
    def get_log_prob(self, x, c):
        p = math.log(self.prior[c])
        for i in range(len(x)): 
            p += math.log(self.cond[c][i][x[i]])
        return p
    
    def predict(self, X_test):
        y_test = []
        for i in range(len(X_test)):
            v = - 0x7fffffff
            c = 0
            for j in range(self.num_of_label):
                v_ = self.get_log_prob(X_test[i], j)
                if v_ > v:
                    v = v_
                    c = j
            y_test.append(c)
        return y_test
    
X_train = [ [0, 0], [0, 1], [1, 0], [1, 1] ]
y_train = [0, 0, 1, 1]
print X_train
print y_train
X_test = X_train
classifier = NaiveBayesianClassifier(4, [2, 2])
classifier.train(X_train, y_train)
print classifier.predict(X_test)



