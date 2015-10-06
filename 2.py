#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# question 3
def estimate_vc_sample_size(N, dvc, epsilon, delta):
    while True:
        estimate = 8.0/np.power(epsilon, 2) * np.log(4.0 * np.power(2.0*N, dvc) / delta)
        if np.abs(N - estimate) < 1e-8:
            return estimate
        N = estimate


# question 4
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


if __name__ == '__main__':
    compare_bounds(10, 50, 0.05)
