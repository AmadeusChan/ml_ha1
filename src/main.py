import numpy as np
import random
from Classifier import Classifier, NaiveBayesianClassifier

class KFold(object):

    # please make sure that cv >= |X_train|
    def __init__(self, X, y, cv):
        self.X = np.copy(X)
        self.y = np.copy(y)
        self.cv = cv
        self.now = 0 
        N = len(self.X)
        index = range(N)
        random.shuffle(index)
        self.X = self.X[index]
        self.y = self.y[index]

    def next_fold(self):
        size = len(self.X) / self.cv
        begin = self.now
        end = self.now + size
        self.now += size
        if self.now >= len(self.X):
            self.now = 0
        X_train = self.X[begin: end]
        y_train = self.y[begin: end]
        X_valid = np.apend(self.X[0: begin], self.X[end: len(self.X)], axis = 0)
        y_valid = np.append(self.y[0: begin], self.y[end: len(self.X)], axis = 0)
        return X_train, y_train, X_valid, y_valid

def load_data():
    pass

# to calculate
# - accuracy
# - recall
# - precision
# - F1-score
# - true negative rate
def evaluate(y_, y):
    tp = tn = fp = fn = 0
    for i in range(len(y_)):
        if y == 1:
            if y_ == 1:
                tp += 1
            else: 
                fn += 1
        else:
            if y_ == 1:
                fp += 1
            else:
                tn += 1
    accuracy = (tp + tn) * 1. / (tp + tn + fp + fn)
    recall = tp * 1. / (tp + fn)
    precision = tp * 1. / (tp + fp)
    f1 = 2. * precision * recall / (precision + recall)
    tnr = tn * 1. / (tn + fp)
    return accuracy, recall, precision, f1, tnr


cv = 11
X, y, X_test, y_test, lev_of_fea = loda_data()
fold = KFold(X, y, cv)
y_test_ = np.zeros(shape = [len(y_test)])
for i in range(cv):
    X_train, y_train, X_valid, y_valid = fold.next_fold()
    NaiveBayesianClassifier classifier = NaiveBayesianClassifier(2, lev_of_fea)
    classifier.train(X_train, y_train)
    y_test_ = y_test_ + np.asarray(classifier.predict(X_test))
    y_valid_ = np.asarray(classifier.predict(X_valid))
    acc, rec, prec, f1, tnr = evaluate(y_valid_, y_valid)
    print "Fold %s:     accuracy: %s    recall: %s      precision: %s       F1-score: %s        true negative rate: %s" % (i, acc, rec, prec, f1, tnr)

# majority vote
for i in range(len(y_test_)):
    if y_test_[i] > cv - y_test_[i]:
        y_test_[i] = 1
    else:
        y_test_[i] = 0

acc, rec, prec, f1, tnr = evaluate(y_test_, y_test)
print "test:     accuracy: %s    recall: %s      precision: %s       F1-score: %s        true negative rate: %s" % (acc, rec, prec, f1, tnr)
