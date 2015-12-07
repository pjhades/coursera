#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy as sp
import numpy as np
from scipy import linalg

from sklearn import kernel_ridge

def load_data(filepath):
    X = []
    y = []
    with open(filepath, 'r') as f:
        for line in f:
            fields = [float(x) for x in line.strip().split()]
            X.append(fields[:-1])
            y.append(fields[-1])
    return np.array(X), np.array(y)


def split_data(X, y, n_train):
    X_train, y_train = X[:n_train, :], y[:n_train]
    X_test, y_test = X[n_train:, :], y[n_train:]
    return X_train, y_train, X_test, y_test


def get_error(X, y, f):
    N = X.shape[0]
    err = 0.
    for i in range(N):
        if f(X[i]) != y[i]:
            err += 1
    return err*1. / N


def decision_stump_train(X, y, weight):
    N = X.shape[0]
    err_min = N

    # try each feature
    for k in range(X.shape[1]):
        data = np.array(sorted(list(zip(X[:, k], y, weight))))
        xs = data[:, 0]
        ys = data[:, 1]
        us = data[:, 2]

        # predict all +
        err = np.sum(us * (ys == -1))
        if err < err_min:
            err_min = err
            s, dim, theta = 1., k, xs.min() - 1.

        # predict all -
        err = np.sum(us * (ys == 1))
        if err < err_min:
            err_min = err
            s, dim, theta = -1., k, xs.min() - 1.

        # try each stump
        for i in range(N - 1):
            yl, yr = ys[:i+1], ys[i+1:]
            ul, ur = us[:i+1], us[i+1:]

            # predict left - right +
            err = np.sum(ul * (yl == 1)) + np.sum(ur * (yr == -1))
            if err < err_min:
                err_min = err
                s, dim, theta = 1., k, (xs[i] + xs[i+1]) * 0.5

            # predict left + right -
            err = np.sum(ul * (yl == -1)) + np.sum(ur * (yr == 1))
            if err < err_min:
                err_min = err
                s, dim, theta = -1., k, (xs[i] + xs[i+1]) * 0.5

    return s, dim, theta, err_min*1. / np.sum(weight)


def make_stump(s, dim, theta):
    def _stump(x):
        return s * np.sign(x[dim] - theta)
    return _stump


def adaboost_train(X, y, n_iter):
    N = X.shape[0]

    stump = []
    vote = []
    weight = np.array([1./N] * N)

    epsilon_min = 1.

    for t in range(n_iter):
        # q14-15
        print('#{0} sum_u: {1:f}'.format(t + 1, np.sum(weight)), end=' ')

        s, dim, theta, err = decision_stump_train(X, y, weight)
        func = make_stump(s, dim, theta)

        # q16
        if err < epsilon_min:
            epsilon_min = err

        factor = np.sqrt((1. - err) / err)
        for i in range(N):
            if func(X[i]) != y[i]:
                weight[i] *= factor
            else:
                weight[i] /= factor

        stump.append(func)
        alpha = np.log(factor)
        vote.append(alpha)

        print('s: {0}, i: {1}, theta: {2:f}, ein: {3:f}'.format(s, dim, theta,
              get_error(X, y, func)))

    def _decision(x):
        score = 0.
        for (a, f) in zip(vote, stump):
            score += a * f(x)
        return np.sign(score)

    # q16
    print('min epsilon: {0:f}'.format(epsilon_min))

    return _decision


def make_rbf(gamma):
    def _rbf(x1, x2):
        return np.exp(-gamma * np.power(linalg.norm(x1 - x2), 2.))
    return _rbf


def kernel_ridge_train(X, y, gamma, lambd):
    kernel = make_rbf(gamma)
    N = X_train.shape[0]

    K = []
    for i in range(N):
        for j in range(N):
            K.append(kernel(X[i], X[j]))
    K = np.array(K).reshape(N, N)
    beta = linalg.inv(lambd * np.identity(N) + K).dot(y)

    return beta


def kernel_ridge_test(beta, X_train, y_train, X_test, y_test, gamma):
    kernel = make_rbf(gamma)
    N = X_train.shape[0]
    errors = []

    for (X, y) in [(X_train, y_train), (X_test, y_test)]:
        M = X.shape[0]
        err = 0.
        for i in range(M):
            score = 0.
            for j in range(N):
                score += beta[j] * kernel(X_train[j], X[i])
            if y[i] != np.sign(score):
                err += 1
        errors.append(err*1. / M)
    return errors


def kernel_ridge_train_cheat(X, y, gamma, lambd):
    kr = kernel_ridge.KernelRidge(alpha=lambd*.5, kernel='rbf', gamma=gamma)
    kr.fit(X, y)
    return kr


def kernel_ridge_test_cheat(kr, X_train, y_train, X_test, y_test):
    errors = []

    for (X, y) in [(X_train, y_train), (X_test, y_test)]:
        M = X.shape[0]
        err = 0.
        for i in range(M):
            if y[i] != np.sign(kr.predict(X[i])):
                err += 1
        errors.append(err*1. / M)
    return errors


if __name__ == '__main__':
    #X_train, y_train = load_data('/Users/pjhades/code/lab/ml/train.dat')
    #g = adaboost_train(X_train, y_train, 300)
    ## q12-13
    #print('ein: {0:f}'.format(get_error(X_train, y_train, g)))

    ## q17-18
    #X_test, y_test = load_data('/Users/pjhades/code/lab/ml/test.dat')
    #print('eout: {0:f}'.format(get_error(X_test, y_test, g)))

    # q19-20
    X, y = load_data('/Users/pjhades/code/lab/ml/kernel_ridge.dat')
    X_train, y_train, X_test, y_test = split_data(X, y, 400)

    for gamma in [32., 2., 0.125]:
        for lambd in [0.001, 1., 1000.]:
            print('gamma: {0}, lambda: {1}'.format(gamma, lambd), end='  ')

            beta = kernel_ridge_train(X_train, y_train, gamma, lambd)
            ein, eout = kernel_ridge_test(beta, X_train, y_train, X_test, y_test, gamma)

            #kr = kernel_ridge_train_cheat(X_train, y_train, gamma, lambd)
            #ein, eout = kernel_ridge_test_cheat(kr, X_train, y_train, X_test, y_test)

            print('ein: {0}, eout: {1}'.format(ein, eout))

