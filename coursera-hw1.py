#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pla
import numpy as np

def read_data(filename):
    with open(filename, 'r') as infile:
        X = []
        y = []
        for line in infile:
            fields = line.strip().split()
            X.append([float(x) for x in fields[:-1]])
            y.append(0 if fields[-1] == '-1' else 1)

    return np.array(X), np.array(y)


if __name__ == '__main__':
    # question 15
    #X, y = read_data('data/quiz1-15-train.dat')
    #pla.pla_train(X, y, verbose=True, random=False)


    # question 16-17
    #X, y = read_data('data/quiz1-15-train.dat')
    #n = 2000
    #update_total = 0

    #for i in range(n):
    #    print('round {0}/{1}'.format(i, n))
    #    # set a different seed for each round
    #    rs = np.random.RandomState(i + 1)
    #    # get a visit order
    #    visit_seq = list(rs.permutation(X.shape[0]))
    #    update_total += pla.pla_train(X, y, rate=0.5, random=visit_seq)[-1]

    #print(update_total*1.0 / n)


    # question 18-20
    X_train, y_train = read_data('data/quiz1-18-train.dat')
    X_test, y_test = read_data('data/quiz1-18-test.dat')
    coeff = [-1, 1]

    n = 2000
    err_total = 0
    for i in range(n):
        print('round {0}/{1} '.format(i, n), end='')
        w = pla.pla_pocket_train(X_train, y_train, random=True, rs=i+1,
                                 update_all=50)
        err = 0
        for x, y in zip(X_test, y_test):
            x = np.hstack([np.array(1.0), x])
            res = np.sign(w.dot(x))
            if res != coeff[y]:
                err += 1
        print('error={0:f}'.format(err*1.0 / X_test.shape[0]))
        err_total += err*1.0 / X_test.shape[0]
    
    print(err_total / n)
        
