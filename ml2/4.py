#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import heapq
import numpy as np
from scipy import linalg
from multiprocessing import Pool

def q4():
    n = 36
    dp = [[-1] * (n+1) for i in range(n+1)]
    dp[n][10] = 0
    path = [[-1] * (n+1) for i in range(n+1)]
    path[n][10] = 0
    for i in range(n-1, -1, -1):
        for j in range(1, n+1):
            best = -1
            if i + j > n:
                continue
            if i + j == n:
                best = dp[n][10] + 10*j
                last = 10
            for k in range(1, n-i-j+1):
                if dp[i+j][k] != -1 and dp[i+j][k] + k*j > best:
                    best = dp[i+j][k] + k*j
                    last = k
            dp[i][j] = best
            path[i][j] = last
    answer = 0
    last = 0
    for i in range(1, n+1):
        if dp[0][i] > answer:
            answer = dp[0][i]
            last = i
    print(answer + last, last)
    
    print('each layer: ', end=' ')
    left = 0
    while left < n:
        print(last, end=' ')
        k = left
        left += last
        last = path[k][last]
    print()


def load_data(filepath):
    X = []
    y = []
    with open(filepath, 'r') as f:
        for line in f:
            fields = [float(x) for x in line.strip().split()]
            X.append(fields[:-1])
            y.append(fields[-1])
    return np.array(X), np.array(y)


def get_error(X, y, f):
    N = X.shape[0]
    err = 0.
    for i in range(N):
        if f(X[i]) != y[i]:
            err += 1
    return err*1. / N


def d_tanh(x):
    #return np.cosh(x) ** -2.
    return 1. - np.tanh(x)**2


def make_nnet_hypo(topo, weight):
    def _f(x):
        for layer in range(1, len(topo)):
            x = np.vstack([1, x.reshape(topo[layer - 1], 1)])
            x = np.tanh(weight[layer].T.dot(x))
        return np.sign(x.ravel()[0])
    return _f


def nnet_train(X, y, hidden, n_iter, eta, r):
    N, d = X.shape
    topo = [d] + hidden
    L = len(topo)

    weight = [None] * L
    x = [None] * L
    score = [None] * L
    delta = [None] * L

    # init weights
    for layer in range(1, L):
        weight[layer] = np.random.uniform(-r, r, (topo[layer - 1] + 1, topo[layer]))

    # SGD
    for it in range(n_iter):
        idx = np.random.randint(N)
        x[0] = np.hstack([1, X[idx]])

        # forward
        for layer in range(1, L):
            score[layer] = weight[layer].T.dot(x[layer - 1])
            x[layer] = np.hstack([1, np.tanh(score[layer])])

        # backward
        delta[L - 1] = 2. * (np.tanh(score[L - 1]) - y[idx]) * d_tanh(score[L - 1])

        for layer in range(L - 2, 0, -1):
            delta[layer] = np.dot(weight[layer + 1][1:,:], delta[layer + 1]) * d_tanh(score[layer])

        for layer in range(L - 1, 0, -1):
            weight[layer] -= eta * np.outer(x[layer - 1], delta[layer])

    return make_nnet_hypo(topo, weight)


def nnet_param_test(param):
    X_train, y_train, X_test, y_test, hidden, n_iter, eta, r = param
    n_repeat = 500
    eout_all = 0.
    for repeat in range(n_repeat):
        print('topo={0}, n_iter={1}, eta={2}, r={3}, round {4}/{5}'.format(
              hidden, n_iter, eta, r, repeat, n_repeat))
        f = nnet_train(X_train, y_train, hidden, n_iter, eta, r)
        eout_all += get_error(X_test, y_test, f)
    return (r, eout_all / n_repeat)


def make_knn_hypo(X, y, k):
    assert(X.shape[0] >= k and y.shape[0] >= k)

    def _f(x):
        h = []
        for i in range(k):
            h.append((-linalg.norm(x - X[i]), y[i]))
        heapq.heapify(h)

        for i in range(k, X.shape[0]):
            dist = linalg.norm(x - X[i])
            dist_max= -heapq.nsmallest(1, h)[0][0]
            if dist < dist_max:
                heapq.heapreplace(h, (-dist, y[i]))

        score = 0.
        for nb in h:
            score += nb[1]
        return np.sign(score)

    return _f


def kmeans_train(X, k):
    N = X.shape[0]
    centers = X[np.random.choice(N, k, replace=False)]
    cluster = [-1] * N
    done = False
    while not done:
        done = True
        for i in range(N):
            dist_min = linalg.norm(X[i] - centers[0])
            idx = 0
            for j in range(k):
                dist = linalg.norm(X[i] - centers[j])
                if dist < dist_min:
                    dist_min = dist
                    idx = j
            if cluster[i] != idx:
                done = False
            cluster[i] = idx

        sums = [0] * k
        size = [0] * k
        for i in range(N):
            sums[cluster[i]] += X[i]
            size[cluster[i]] += 1
        for i in range(k):
            centers[i] = sums[i] / size[i]

    ein = 0.0
    for i in range(N):
        ein += linalg.norm(X[i] - centers[cluster[i]]) ** 2
    return ein / N


if __name__ == '__main__':
    # q11-14
    X_train, y_train = load_data('/Users/pjhades/code/lab/ml/nn_train.dat')
    X_test, y_test = load_data('/Users/pjhades/code/lab/ml/nn_test.dat')

    # n_iter = 50000
    # eta = 0.1
    # #r = 0.1
    # hidden = [3, 1]
    # pool = Pool(processes=5)
    # args = [(X_train, y_train, X_test, y_test,
    #          hidden, n_iter, eta, r) for r in [0, 0.1, 10, 100, 1000]]
    # for res in pool.map(nnet_param_test, args, chunksize=1):
    #     print(res)

    n_iter = 50000
    eta = 0.01
    r = 0.1
    n_repeat = 500
    eout_all = 0
    for repeat in range(n_repeat):
        print('round {0}/{1}'.format(repeat + 1, n_repeat))
        f = nnet_train(X_train, y_train, [8, 3, 1], n_iter, eta, r)
        eout_all += get_error(X_test, y_test, f)
    print('eout:', eout_all / n_repeat)


    # q15-18
    # X_train, y_train = load_data('/Users/pjhades/code/lab/ml/knn_train.dat')
    # X_test, y_test = load_data('/Users/pjhades/code/lab/ml/knn_test.dat')
    # print('ein:', get_error(X_train, y_train, make_knn_hypo(X_train, y_train, 5)))
    # print('eout:', get_error(X_test, y_test, make_knn_hypo(X_train, y_train, 5)))


    # q19-20
    # X = []
    # with open('/Users/pjhades/code/lab/ml/kmeans_train.dat') as f:
    #     for line in f:
    #         X.append([float(v) for v in line.strip().split()])
    # X = np.array(X)
    # n_repeat = 500
    # ein_all = 0.
    # for repeat in range(n_repeat):
    #     print('round {0}/{1}'.format(repeat + 1, n_repeat))
    #     ein_all += kmeans_train(X, 10)
    # print(ein_all / n_repeat)

