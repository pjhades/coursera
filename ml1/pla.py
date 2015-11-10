#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import cross_validation
from matplotlib import colors

def gen_data(dim=2, size=100):
    return datasets.make_classification(n_samples=size, n_features=dim,
                                        n_informative=dim, n_redundant=0,
                                        n_classes=2, n_clusters_per_class=1,
                                        random_state=1, hypercube=False,
                                        class_sep=30., flip_y=0)


def split_data(X, y, ratio=.3):
    return cross_validation.train_test_split(X, y, test_size=ratio)


# normal perceptron
# X: points to be classified
# y: labels of 1 or 0
# rate: learning rate
# random: if python list, visit samples in the given order
#         if True, visit samples purely randomly
#         if False, visit samples sequentially
# rs: if numpy.random.RandomState instance, random number generator
#     if int, seed for RandomState
#     if None, create a new RandomState
# verbose: display each iteration
def pla_train(X, y, rate=1.0, random=False, rs=None, verbose=False):
    n_samples, n_features = X.shape 

    w = np.zeros(n_features + 1)
    n_correct = 0
    n_iter = 1
    n_update = 0
    coeff = [-1, 1]
    i = 0

    if rs is None:
        rs = np.random.RandomState()
    if type(rs) is int:
        rs = np.random.RandomState(rs)

    if type(random) is bool and random:
        i = rs.randint(0, n_samples)

    while n_correct != n_samples:
        if type(random) is list:
            idx = random[i]
        else:
            idx = i

        x = np.hstack([np.array(1.0), X[idx]])
        res = np.sign(w.dot(x))
        if res != coeff[y[idx]]:
            if verbose:
                print('#iter{0}, sample {1} wrong, '.format(n_iter, idx), end='')
            w += rate * coeff[y[idx]] * x
            if verbose:
                print('updated w={0}'.format(w))
            n_correct = 0
            n_update += 1
        else:
            if verbose:
                print('#iter{0}, sample {1} correct'.format(n_iter, idx))
            n_correct += 1

        if type(random) is bool and random:
            i = rs.randint(0, n_samples)
        else:
            i += 1
            if i >= n_samples:
                i = 0
        n_iter += 1

    if verbose:
        print('learned w={0}'.format(w))
        print('n_iter={0}, n_update={1}'.format(n_iter, n_update))

    return w


# perceptron with pocket 
# X: points to be classified
# y: labels of 1 or 0
# rate: learning rate
# update_all: number of updates
# random: if python list, visit samples in the given order;
#         if False, visit samples sequentially;
#         if True, visit samples purely randomly
# rs: if numpy.random.RandomState instance, random number generator
#     if int, seed for RandomState
#     if None, create a new RandomState
# verbose: display each iteration
def pla_pocket_train(X, y, rate=1.0, update_all=100, random=False, rs=None,
                     verbose=False):
    n_samples, n_features = X.shape 

    w = np.zeros(n_features + 1)
    pocket = w
    n_pocket_all = 0
    n_pocket_run = 0
    n_cur_all = 0
    n_cur_run = 0
    n_iter = 1
    n_update = 0
    coeff = [-1, 1]
    i = 0

    if rs is None:
        rs = np.random.RandomState()
    if type(rs) is int:
        rs = np.random.RandomState(rs)

    if type(random) is bool and random:
        i = rs.randint(0, n_samples)

    while n_update < update_all:
        if type(random) is list:
            idx = random[i]
        else:
            idx = i

        x = np.hstack([np.array(1.0), X[idx]])
        res = np.sign(w.dot(x))
        if res != coeff[y[idx]]:
            if verbose:
                print('#iter{0}, sample {1} wrong, '.format(n_iter, idx), end='')
            w += rate * coeff[y[idx]] * x
            if verbose:
                print('updated w={0}'.format(w))
            n_cur_run = 0
            n_update += 1
        else:
            if verbose:
                print('#iter{0}, sample {1} correct'.format(n_iter, idx))
            n_cur_run += 1
            if n_cur_run > n_pocket_run:
                n_cur_all = 0
                # compute how many samples we do correctly
                for k in range(n_samples):
                    vx = np.hstack([np.array(1.0), X[k]])
                    vy = np.sign(w.dot(vx))
                    if vy == coeff[y[k]]:
                        n_cur_all += 1
                if n_cur_all > n_pocket_all:
                    pocket = w
                    n_pocket_run = n_cur_run
                    n_pocket_all = n_cur_all
                if n_cur_all == n_samples:
                    if verbose:
                        print('all samples are correct')
                    break

        if type(random) is bool and random: 
            i = rs.randint(0, n_samples)
        else:
            i += 1
            if i >= n_samples:
                i = 0
        n_iter += 1

    if verbose:
        print('learned w={0}'.format(w))
        print('n_iter={0}, n_update={1}'.format(n_iter, n_update))

    return w


def main():
    X, y = gen_data(size=5000)
    X_train, X_test, y_train, y_test = split_data(X, y)
    #w = pla_train(X_train, y_train)
    w = pla_pocket_train(X_train, y_train, random=True, rs=10, verbose=True)

    cmap = colors.ListedColormap(['#ff0000', '#0000ff'])

    xs = np.linspace(X[:, 0].min() - 5., X[:, 0].max() + 5., 1000)
    ys = np.linspace(X[:, 1].min() - 5., X[:, 1].max() + 5., 1000)
    xs, ys = np.meshgrid(xs, ys)

    def func(xs, ys):
        mat = np.vstack([np.ones(xs.ravel().shape[0]), xs.ravel(), ys.ravel()])
        return np.sign(w.dot(mat)).reshape(xs.shape)

    plt.contourf(xs, ys, func(xs, ys), alpha=0.4)

    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap)
    plt.scatter(X_test[:, 0], X_test[:, 1], marker='^', c=y_test,cmap=cmap, alpha=0.6)

    plt.show()


if __name__ == '__main__':
    main()
