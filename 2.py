#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import functools
import numpy as np
import matplotlib.pyplot as plt

# q3
def estimate_vc_sample_size(N, dvc, epsilon, delta):
    while True:
        estimate = 8.0/np.power(epsilon, 2) * np.log(4.0 * np.power(2.0*N, dvc) / delta)
        if np.abs(N - estimate) < 1e-8:
            return estimate
        N = estimate


# q4
def compare_bounds(N, dvc, delta):
    # growth function
    def m(n):
        return np.power(n, dvc)

    xs = np.linspace(1, N*1.5, 1000)
    ys = np.linspace(0, 100., 1000)
    x, y = np.meshgrid(xs, ys)

    contours = [
        ('original vc', plt.contour(x, y, np.sqrt(8/x * np.log(4*m(2*x) / delta)) - y, [0], colors='r')),
        ('rademancher', plt.contour(x, y, np.sqrt(2*np.log(2*x*m(x)) / x) + np.sqrt(2/x * np.log(1./delta)) + 1./x - y, [0], colors='g')),
        ('parrondo', plt.contour(x, y, np.sqrt(1./x * (2*y + np.log(6*m(2*x) / delta))) - y, [0], colors='b')),
        # here taking the power of 100 results in overflow, so change it manually
        ('devroye', plt.contour(x, y, np.sqrt(1./(2*x) * (4*y*(1+y) + 2.*dvc * np.log(np.power(4, 1./(2.*dvc))*x / np.power(delta, 1./(2.*dvc))))) - y, [0], colors='magenta')),
        ('variant vc', plt.contour(x, y, np.sqrt(16./x * np.log(2*m(x) / np.sqrt(delta))) - y, [0], colors='orange')),
    ]

    for cont in contours:
        for line in cont[1].collections:
            line.set_label(cont[0])

    plt.legend()
    plt.grid()
    plt.show()


# q16-20
def generate_data(N):
    data = []
    for i in range(N):
        x = np.random.uniform(-1., 1.)
        y = np.sign(x)
        if np.random.uniform(0, 1.) <= 0.2:
            y = -y
        data.append((x, y))
    data.sort(key=lambda x:x[0])
    data = np.array(data)
    return data[:, 0], data[:, 1]


def decision_stump_train(x, y):
    N = x.size

    n_pos = functools.reduce(lambda s,e:s+(1 if e == 1 else 0), y, 0)
    n_neg = N - n_pos

    ein_all_pos = n_neg*1. / N
    ein_all_neg = n_pos*1. / N
    if ein_all_pos < ein_all_neg:
        ein_min = ein_all_pos
        theta = x.min()
        s = 1
    else:
        ein_min = ein_all_neg
        theta = x.max()
        s = -1

    # find all other positions and calculate Ein
    # for each i:
    #   s=+1: 0~i are -1, i+1~N-1 are +1
    #   s=-1: 0~i are +1, i+1~N-1 are -1
    for i in range(N - 1):
        for sign in [1, -1]:
            ein = (functools.reduce(lambda s,e:s+(1 if e == sign else 0), y[:i+1], 0) + \
                   functools.reduce(lambda s,e:s+(1 if e == -sign else 0), y[i+1:], 0))*1. / N
            if ein < ein_min or \
                (ein == ein_min and np.random.uniform(0, 1.) <= 0.5):
                ein_min = ein
                s = sign
                theta = (x[i] + x[i + 1]) / 2.

    print('ein_min={0}, theta={1}, s={2}'.format(ein_min, theta, s))
    return ein_min, s, theta


def decision_stump_test(repeat=5000):
    ein_sum = 0
    eout_sum = 0
    for i in range(repeat):
        ein, s, theta = decision_stump_train(*generate_data(20))
        eout = 0.5 + 0.3*s*(np.abs(theta) - 1)
        ein_sum += ein
        eout_sum += eout
    print('ein_avg={0}, eout_avg={1}'.format(ein_sum*1./repeat, eout_sum*1./repeat))


# train each dimension with 1D decision stump,
# find the best ein, s, theta, and the dimension
# on which we get those parameters
def multidim_decision_stump_train(xs, y):
    ein_min = 100
    dim = None
    for i in range(xs.shape[1]):
        x = xs[:, i]
        xy = [pair for pair in zip(x, y)]
        xy.sort(key=lambda pair:pair[0])
        xy = np.array(xy)
        ein, s_this, theta_this = decision_stump_train(xy[:, 0], xy[:, 1])
        if ein < ein_min or \
            (ein == ein_min and np.random.uniform(0, 1.) <= 0.5):
            ein_min = ein
            s = s_this
            theta = theta_this
            dim = i
    if dim is None:
        print('cannot find best parameter', file=sys.stderr)

    return ein_min, s, theta, dim


def read_data(filename):
    xs = []
    y = []
    with open(filename, 'r') as f:
        for line in f:
            fields = line.strip().split()
            xs.append([float(x) for x in fields[:-1]])
            y.append(int(fields[-1]))
    return np.array(xs), np.array(y)


def multidim_decision_stump_test(xs, y, s, theta, dim):
    x = xs[:, dim]
    labels = np.array([s * np.sign(xi - theta) for xi in x])
    print('eout={0}'.format(functools.reduce(lambda s,e:s + (1 if e[0] != e[1] else 0),
                                             zip(labels, y),
                                             0) * 1. / y.size))


if __name__ == '__main__':
    #decision_stump_test()

    xs, y = read_data('/Users/pjhades/code/lab/ml/train.dat')
    ein_min, s, theta, dim = multidim_decision_stump_train(xs, y)
    print('ein_min={0}, s={1}, theta={2}'.format(ein_min, s, theta))

    xs, y = read_data('/Users/pjhades/code/lab/ml/test.dat')
    multidim_decision_stump_test(xs, y, s, theta, dim)
