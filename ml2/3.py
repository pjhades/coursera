#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

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


def gini(y):
    s = 0.
    for label in set(y):
        s += (np.sum(y == label) * 1. / y.size) ** 2
    return 1 - s


def cart_find_split(X, y):
    N = X.shape[0]
    imp_min = N
    pos = None
    dim = None
    thres = None

    # try each feature
    for k in range(X.shape[1]):
        data = np.array(sorted(list(zip(X[:, k], y))))
        xs = data[:, 0]
        ys = data[:, 1]

        # try each stump
        for i in range(N - 1):
            yl = ys[:i+1]
            yr = ys[i+1:]
            imp = gini(yl) * yl.size + gini(yr) * yr.size
            if imp < imp_min:
                imp_min = imp
                thres = (xs[i] + xs[i+1]) * 0.5
                pos = i
                dim = k 

    assert((imp_min != N) and (pos is not None) and
           (dim is not None) and (thres is not None))

    return pos, thres, dim
    

def make_leaf(label):
    def _func(x):
        return label
    return _func


def make_internal(fun_l, fun_r, thres, dim):
    def _func(x):
        if x[dim] < thres:
            return fun_l(x)
        return fun_r(x)
    return _func


def do_cart_train(X, y, depth, max_depth=None):
    assert(y.size > 0)

    N, d = X.shape

    if depth == max_depth:
        majority = 0
        leaf_label = 0
        for label in set(y):
            count = np.sum(y == label)
            if count > majority:
                majority = count
                leaf_label = label
        return make_leaf(leaf_label)

    if np.sum(y == y[0]) == y.size:
        return make_leaf(y[0])

    # learn branch
    pos, thres, dim = cart_find_split(X, y)

    # split
    xy = np.hstack([X, y.reshape(y.shape[0], 1)])
    xy = xy[xy[:, dim].argsort()]
    X_sub = xy[:, :d]
    y_sub = xy[:, d:].reshape(N)
    func_left  = do_cart_train(X_sub[:pos+1], y_sub[:pos+1], depth+1, max_depth)
    func_right = do_cart_train(X_sub[pos+1:], y_sub[pos+1:], depth+1, max_depth)

    return make_internal(func_left, func_right, thres, dim)


def cart_train(X, y, max_depth=None):
    return do_cart_train(X, y, 0, max_depth)


ein_all = 0.


def random_forest_train(X, y, n_tree, n_sample):
    N, d = X.shape
    forest = []
    for t in range(n_tree):
        idx_bag = np.random.choice(np.array(range(N)), n_sample)
        X_bag = X[idx_bag]
        y_bag = y[idx_bag]
        tree = cart_train(X_bag, y_bag, max_depth=1)
        #q16
        global ein_all
        ein_all += get_error(X, y, tree)
        forest.append(tree)

    def _decision(x):
        score = 0
        for tree in forest:
            score += tree(x)
        return np.sign(score)

    return _decision


if __name__ == '__main__':
    X_train, y_train = load_data('/Users/pjhades/code/lab/ml/train.dat')
    X_test, y_test = load_data('/Users/pjhades/code/lab/ml/test.dat')

    # q13-15
    #func = cart_train(X_train, y_train)
    #print(get_error(X_test, y_test, func))

    # q16-20
    n_tree = 300
    n_sample = X_train.shape[0]
    repeat = 100
    ein_rf_all = 0.
    eout_rf_all = 0.
    for r in range(repeat):
        print('round', r+1)
        rf = random_forest_train(X_train, y_train, n_tree, n_sample)
        ein_rf_all += get_error(X_train, y_train, rf)
        eout_rf_all += get_error(X_test, y_test, rf)
    print('avg ein of each tree:', ein_all / (n_tree * repeat))
    print('avg ein of random forest:', ein_rf_all / repeat)
    print('avg eout of random forest:', eout_rf_all / repeat)
