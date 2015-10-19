#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg
from sklearn import linear_model

# q6-10
# original function
def E(u, v):
    return np.exp(u) + np.exp(2*v) + np.exp(u*v) + \
           np.power(u, 2) - 2*u*v + 2*np.power(v, 2) - 3*u - 2*v


# order-2 Taylor
def E2(delta_u, delta_v, buu, bvv, buv, bu, bv, b):
    return buu*np.power(delta_u, 2) + bvv*np.power(delta_v, 2) + \
           buv*delta_u*delta_v + bu*delta_u + bv*delta_v + b


# order-1 derivatives
def d_u(u, v):
    return np.exp(u) + v*np.exp(u*v) + 2*u - 2*v - 3
def d_v(u, v):
    return 2*np.exp(2*v) + u*np.exp(u*v) - 2*u + 4*v - 2


# order-2 derivatives
def d_uu(u, v):
    return np.exp(u) + np.exp(u*v)*np.power(v, 2) + 2
def d_uv(u, v):
    return np.exp(u*v) + u*v*np.exp(u*v) - 2
def d_vu(u, v):
    return np.exp(u*v) + u*v*np.exp(u*v) - 2
def d_vv(u, v):
    return 4*np.exp(2*v) + np.exp(u*v)*np.power(u, 2) + 4


def q7(u0, v0, eta, n_iter):
    u, v = u0, v0
    while n_iter > 0:
        u, v = u - eta * d_u(u, v), v - eta * d_v(u, v)
        print('({0}, {1}), {2}'.format(u, v, E(u, v)))
        n_iter -= 1
    return E(u, v), u, v


def q10(u0, v0, n_iter):
    x = np.array([u0, v0])
    while n_iter > 0:
        # Hessian matrix
        H = np.array([[d_uu(*x), d_uv(*x)],
                      [d_vu(*x), d_vv(*x)]])
        nabla = np.array([d_u(*x), d_v(*x)])
        # Newton
        x = x - linalg.inv(H).dot(nabla)
        print('({0}, {1}), {2}'.format(x[0], x[1], E(*x)))
        n_iter -= 1
    return E(*x)


# q13-15
def q13_gendata(N):
    xs = []
    ys = []
    for i in range(N):
        x1 = np.random.uniform(-1, 1)
        x2 = np.random.uniform(-1, 1)
        y = np.sign(x1*x1 + x2*x2 - 0.6)
        xs.append([x1, x2])
        ys.append(y)
    # noise
    for i in np.random.choice(range(N), int(0.1*N)):
        ys[i] = -ys[i]
    return np.array(xs), np.array(ys).reshape(N, 1)


def q13_test(repeat):
    ein_all = 0
    for r in range(repeat): 
        N = 1000

        X, y = q13_gendata(N)
        one = np.ones((N, 1))
        X = np.hstack([one, X])

        w = linalg.pinv(X).dot(y)
        y_hat = np.sign(X.dot(w))

        error = 0
        for i in range(N):
            if y_hat[i][0] != y[i][0]:
                error += 1
        ein = error*1. / N
        print('round {0}, ein = {1}'.format(r, ein))
        ein_all += ein

    print('avg ein:', ein_all*1. / repeat)


def q14_feature_trans(X):
    xs = []
    for x in X:
        xs.append([1, x[0], x[1], x[0]*x[1], x[0]**2, x[1]**2])
    return np.array(xs)


def q14(repeat):
    answers = [
            # [hypothesis, # agrees]
            [[-1, -0.05, 0.08, 0.13, 1.5, 1.5], 0],
            [[-1, -0.05, 0.08, 0.13, 15, 1.5], 0],
            [[-1, -1.5, 0.08, 0.13, 0.05, 1.5], 0],
            [[-1, -0.05, 0.08, 0.13, 1.5, 15], 0],
            [[-1, -1.5, 0.08, 0.13, 0.05, 0.05], 0],
    ]

    for r in range(repeat): 
        N = 1000
        X, y = q13_gendata(N)
        X = q14_feature_trans(X)
        wlin = linalg.pinv(X).dot(y)
        y_hat = np.sign(X.dot(wlin))

        for i in range(len(answers)):
            dim = len(answers[i][0])
            w = np.array(answers[i][0]).reshape(dim, 1)
            pred = np.sign(X.dot(w))
            for j in range(N):
                if y_hat[j][0] == pred[j][0]:
                    answers[i][1] += 1
        print('round {0}:\nwlin={1}'.format(r, wlin))
        for answer in answers:
            print(answer)
        
    print('final agrees:')
    for answer in answers:
        print(answer)


def q15(repeat):
    eout_all = 0
    N = 1000

    X, y = q13_gendata(N)
    X = q14_feature_trans(X)
    w = linalg.pinv(X).dot(y)

    for round in range(repeat): 
        X_test, y_test = q13_gendata(N)
        X_test = q14_feature_trans(X_test)
        y_hat = np.sign(X_test.dot(w))

        error = 0
        for i in range(N):
            if y_hat[i][0] != y_test[i][0]:
                error += 1
        eout = error*1. / N
        print('round {0}, eout = {1}'.format(round, eout))
        eout_all += eout

    print('avg eout:', eout_all*1. / repeat)


# q18-20
def theta(x):
    return 1. / (1 + np.exp(-x))


def read_data(filename):
    xs = []
    ys = []
    with open(filename, 'r') as infile:
        for line in infile:
            fields = [float(x) for x in line.strip().split()]
            xs.append(fields[:-1])
            ys.append(fields[-1])
    return np.array(xs), np.array(ys).reshape(len(ys), 1)


def gradient_ein(w, X, y):
    N, dim = X.shape
    sum = np.zeros((dim, 1))
    for i in range(N):
        x = X[i].reshape((dim, 1))
        sum += -y[i][0] * x * theta(-y[i][0] * w.T.dot(x))
    return sum*1. / N


def logistic_train(X, y, eta, n_iter):
    dim = X.shape[1]
    w = np.zeros((dim, 1))
    for round in range(n_iter):
        w -= eta * gradient_ein(w, X, y)
        print('round {0}, w = {1}'.format(round, w.T))
    return w


def logistic_SGD_train(X, y, eta, n_iter):
    N, dim = X.shape
    w = np.zeros((dim, 1))
    i = 0
    for round in range(n_iter):
        x = X[i].reshape((dim, 1))
        w += eta * theta(-y[i][0] * w.T.dot(x)) * y[i][0] * x
        i += 1
        if i >= N:
            i = 0
        print('round {0}, w = {1}'.format(round, w.T))
    return w


def logistic_test(eta, n_iter, model_train):
    X, y = read_data('/Users/pjhades/code/lab/ml/train.dat')
    N = X.shape[0]
    ones = np.ones((N, 1))
    X = np.hstack([ones, X])
    w = model_train(X, y, eta, n_iter)

    X_test, y_test = read_data('/Users/pjhades/code/lab/ml/test.dat')
    N = X_test.shape[0]
    ones = np.ones((N, 1))
    X_test = np.hstack([ones, X_test])
    predict = np.sign(X_test.dot(w))

    error = 0
    for i in range(N):
        if predict[i][0] != y_test[i][0]:
            error += 1
    print('eout = {0}'.format(error*1. / N))


if __name__ == '__main__':
    q15(1000)
    #logistic_test(0.001, 2000, logistic_SGD_train)

