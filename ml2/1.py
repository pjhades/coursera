#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import numpy as np
import cvxopt as co
import sklearn.svm
from math import sqrt
from scipy import linalg
from sklearn.preprocessing import Binarizer
from sklearn.cross_validation import train_test_split

# fuck cvxopt's hard-to-manipulate matrix
def q3_4():
    def k(x1, x2):
        return (1 + x1.dot(x2))**2.

    Xs = np.array([[1, 0], [0, 1], [0, -1], [-1, 0], [0, 2], [0, -2], [-2, 0]])
    ys = np.array([[-1], [-1], [-1], [1], [1], [1], [1]])
    N = Xs.shape[0]

    xy = []
    for i in range(N):
        for j in range(N):
            xy.append(1. * ys[i][0] * ys[j][0] * k(Xs[i], Xs[j]))

    X = co.matrix(Xs, tc='d')
    y = co.matrix(ys, tc='d')

    # degree-2 coefficients
    Q = co.matrix(xy, size=(N, N))
    # degree-1 coefficients
    q = co.matrix(-1., size=(N, 1))
    # inequality coefficients
    G = co.spmatrix(-1., range(N), range(N))
    # inequality bounds
    h = co.matrix(0., size=(N, 1))
    # equality coefficients
    A = y.T
    # equality value
    b = co.matrix(0.)

    res = co.solvers.qp(Q, q, G, h, A, b)
    alphas = np.array(res['x'])
    print('Q3: alpha\n{0}\n'.format(alphas))

    # for Q4, we compute the weights after feature
    # transformation, and choose the nearest match

    # index of the first non-zero alpha
    s = np.nonzero(alphas > 1e-2)[0][0]

    # get b after transformation
    b = ys[s][0]
    for i in range(N):
        if alphas[i][0] < 1e-2:
            continue
        b -= ys[i][0] * alphas[i][0] * k(Xs[i], Xs[s])

    # transformed features
    Zs = []
    for x in Xs:
        Zs.append([1, sqrt(2.)*x[0], sqrt(2.)*x[1], x[0]**2., x[1]**2.])
    Zs = np.array(Zs)

    # get w after transformation
    w = np.zeros(Zs[0].shape)
    for i in range(N):
        w += ys[i][0] * alphas[i][0] * Zs[i]
    w[0] += b

    q4_answers = [
        ('1/9 * (8x_1^2 - 16x_1 + 6x_2^2 + 15) = 0', [15./9, -16./9/sqrt(2.), 0, 8./9, 6./9]),
        ('1/9 * (8x_1^2 - 16x_1 + 6x_2^2 - 15) = 0', [-15./9, -16./9/sqrt(2.), 0, 8./9, 6./9]),
        ('1/9 * (8x_2^2 - 16x_2 + 6x_1^2 - 15) = 0', [-15./9, 0, -16./9/sqrt(2.), 6./9, 8./9]),
        ('1/9 * (8x_2^2 - 16x_2 + 6x_1^2 + 15) = 0', [15./9, 0, -16./9/sqrt(2.), 6./9, 8./9]),
    ]
    print('Q4: hyperplane in Z space:\n{0}'.format(w))
    print('answers:')
    for ans in q4_answers:
        print('{0}:   {1}'.format(*ans))


def load_data(filepath):
    X = []
    y = []
    with open(filepath) as f:
        for line in f:
            fields = [float(x) for x in line.strip().split()]
            y.append(fields[0])
            X.append(fields[1:])
    return np.array(X), np.array(y)


def set_binlabel(y, thres):
    ret = y.copy()
    for i in range(y.shape[0]):
        ret[i] = -1 if y[i] != thres else 1

    return ret


def q15():
    X_train, y_train = load_data('/Users/pjhades/code/lab/ml/train.dat')
    y = set_binlabel(y_train, 0)

    svm = sklearn.svm.SVC(C=0.01, kernel='linear') 
    svm.fit(X_train, y)
    print(linalg.norm(svm.coef_))


def get_error(svm, X, y):
    err = 0
    N = y.shape[0]
    for i in range(N):
        if y[i] != svm.predict(X[i])[0]:
            err += 1
    return err*1. / N


def q16_17():
    X_train, y_train = load_data('/Users/pjhades/code/lab/ml/train.dat')

    for goal in [0, 2, 4, 6, 8]:
        y = set_binlabel(y_train, goal)
        svm = sklearn.svm.SVC(C=0.01, kernel='poly', degree=2, coef0=1)
        svm.fit(X_train, y)
        ein = get_error(svm, X_train, y)
        print('{0} vs not {0}, ein={1}'.format(goal, ein), end=', ')
        # FIXME fuck this, don't know why
        print('sum of alphas={0}'.format(np.sum(np.abs(svm.dual_coef_))))


def q18():
    X_train, y_train = load_data('/Users/pjhades/code/lab/ml/train.dat')
    X_test, y_test = load_data('/Users/pjhades/code/lab/ml/test.dat')

    y_train = set_binlabel(y_train, 0)
    y_test = set_binlabel(y_test, 0)

    for C in [0.001, 0.01, 0.1, 1, 10]:
        svm = sklearn.svm.SVC(C=C, kernel='rbf', gamma=100)
        svm.fit(X_train, y_train)

        print('C={0}'.format(C))
        print('# support vectors =', np.sum(svm.n_support_))
        print('Eout =', get_error(svm, X_test, y_test))


def q19():
    X_train, y_train = load_data('/Users/pjhades/code/lab/ml/train.dat')
    X_test, y_test = load_data('/Users/pjhades/code/lab/ml/test.dat')

    y_train = set_binlabel(y_train, 0)
    y_test = set_binlabel(y_test, 0)

    for gamma in [10000, 1000, 1, 10, 100]:
        svm = sklearn.svm.SVC(C=0.1, kernel='rbf', gamma=gamma)
        svm.fit(X_train, y_train)
        print('gamma={0:<10}, Eout={1}'.format(gamma, get_error(svm, X_test, y_test)))


def q20():
    X, y = load_data('/Users/pjhades/code/lab/ml/train.dat')
    y = set_binlabel(y, 0)

    # init hit counts
    gammas = [1, 10, 100, 1000, 10000]
    hits = {}
    for gamma in gammas:
        hits[gamma] = 0

    repeat = 100
    for round in range(repeat):
        print('round {0}/{1}'.format(round, repeat), end=', ')

        err_min = 1
        gamma_min = max(gammas) + 1

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=1000)
        for gamma in gammas:
            svm = sklearn.svm.SVC(C=0.1, kernel='rbf', gamma=gamma)
            svm.fit(X_train, y_train)
            err = get_error(svm, X_val, y_val)
            if err < err_min or (err == err_min and gamma < gamma_min):
                err_min = err
                gamma_min = gamma
        hits[gamma_min] += 1
        print('gamma={0}'.format(gamma_min))

    for gamma in gammas:
        print('{0} hits {1} times'.format(gamma, hits[gamma]))


if __name__ == '__main__':
    q16_17()

