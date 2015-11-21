#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cvxopt as co
from math import sqrt
from scipy import linalg

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




if __name__ == '__main__':
    q3_4()

