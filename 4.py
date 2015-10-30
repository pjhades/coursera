#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import itertools
import numpy as np
from scipy import linalg

datafile_train = '/Users/pjhades/code/lab/ml/train.dat'
datafile_test = '/Users/pjhades/code/lab/ml/test.dat'
datafiles = [datafile_train, datafile_test]

def read_data(filename):
    xs = []
    ys = []
    with open(filename, 'r') as infile:
        for line in infile:
            fields = [float(x) for x in line.strip().split()]
            xs.append(fields[:-1])
            ys.append(fields[-1])
    return np.array(xs), np.array(ys).reshape(len(ys), 1)


def add_weight0(X):
    N = X.shape[0]
    ones = np.ones((N, 1))
    return np.hstack([ones, X])


def q5():
    answers = [
        ('sqrt(sqrt(3)+4)', np.sqrt(np.sqrt(3.) + 4.)),
        ('sqrt(sqrt(3)-1)', np.sqrt(np.sqrt(3.) - 1.)),
        ('sqrt(9+4sqrt(6))', np.sqrt(9. + 4.*np.sqrt(6.))),
        ('sqrt(9-sqrt(6))', np.sqrt(9. - np.sqrt(6.))),
    ]
    for answer in answers:
        k = answer[1]
        print(answer[0], 1./((k+1)**2) + 1./((k-1)**2) == 1./8.)


# q13-15
# r: the regularization factor
def q13_train(X, y, r=0):
    dim = X.shape[1]
    w = linalg.inv(X.T.dot(X) + r*np.eye(dim)).dot(X.T.dot(y))
    return w


def q13_test(w, datasets):
    errors = []
    for dataset in datasets:
        X, y = dataset
        N, dim = X.shape
        y_hat = np.sign(X.dot(w))
        error = 0
        for i in range(N):
            if y_hat[i][0] != y[i][0]:
                error += 1
        errors.append(error*1. / N)

    return errors


def q14_15():
    X_train, y_train = read_data(datafile_train)
    X_test, y_test = read_data(datafile_test)
    X_train = add_weight0(X_train)
    X_test = add_weight0(X_test)
    for r in range(2, -11, -1):
        w = q13_train(X_train, y_train, np.power(10., r))
        errors = q13_test(w, [(X_train, y_train), (X_test, y_test)])
        print('{0:3} {1} {2}'.format(r, *errors))


def q16_17_18():
    X, y = read_data(datafile_train)
    X = add_weight0(X)
    N, dim = X.shape

    X_train, X_val = X[:120, :], X[120:, :]
    y_train, y_val = y[:120, :], y[120:, :]

    X_test, y_test = read_data(datafile_test)
    X_test = add_weight0(X_test)

    best_r = -100
    best_w = None
    eval_min = 1.
    for r in range(2, -11, -1):
        w = q13_train(X_train, y_train, np.power(10., r))
        errors = q13_test(w, [(X_train, y_train),
                              (X_val, y_val),
                              (X_test, y_test)])
        if errors[1] < eval_min:
            eval_min = errors[1]
            best_r = r 
            best_w = w
        elif errors[1] == eval_min and r > best_r:
            best_r = r
            best_w = w
        print('{0:3} {1} {2} {3}'.format(r, *errors))

    w_final = q13_train(X, y, np.power(10., best_r))
    errors = q13_test(w_final, [(X, y), (X_test, y_test)])
    print(*errors)


def q19_20():
    X, y = read_data(datafile_train)
    X = add_weight0(X)
    N, dim = X.shape

    n_fold = 5
    each = int((N + n_fold - 1) / n_fold)

    ecv_min = 1.
    best_r = 0
    for r in range(2, -11, -1):
        ecv = 0
        for i in range(n_fold):
            idx = list(itertools.chain(range(0, i*each), range((i+1)*each, N)))
            X_train = X.take(idx, axis=0)
            y_train = y.take(idx, axis=0)

            idx = list(range(i*each, (i+1)*each))
            X_val = X.take(idx, axis=0)
            y_val = y.take(idx, axis=0)

            w = q13_train(X_train, y_train, np.power(10., r))
            ecv += q13_test(w, [(X_val, y_val)])[0]

        if ecv*1. / n_fold < ecv_min:
            ecv_min = ecv*1. / n_fold
            best_r = r

        print('{0:3} {1}'.format(r, ecv*1. / n_fold))
    
    w_final = q13_train(X, y, np.power(10., best_r))

    X_test, y_test = read_data(datafile_test)
    X_test = add_weight0(X_test)
    errors = q13_test(w_final, [(X, y), (X_test, y_test)])
    print(*errors)


if __name__ == '__main__':
    q19_20()

