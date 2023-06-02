import numpy as np
import math
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def f(t):
    # write target function HERE
    # this code is for problem 3.32
    return np.where((t/np.pi) % 2 < 1, 1, -1)


class FourierTransform:

    def __init__(self, f, t, n):
        self.f = f
        self.t = t
        self.n = n
        self.dt = t[1] - t[0]
        self.T = t[-1] + self.dt
        self.wr = 2 * math.pi/self.T
        self.a0 = 2/self.T * np.sum(self.f(self.t)) * self.dt
        self.an = []
        self.bn = []
        for i in range(self.n):
            ai = 2 / self.T * np.sum(self.f(self.t) * np.cos(i * self.wr * self.t)) * self.dt
            bi = 2 / self.T * np.sum(self.f(self.t) * np.sin(i * self.wr * self.t)) * self.dt
            self.an.append(ai)
            self.bn.append(bi)

    def params(self):
        return self.a0, self.an, self.bn

    def F(self):
        sigma = 0
        for i in range(self.n):
            sigma = sigma + self.an[i] * np.cos(i * self.wr * self.t)
            sigma = sigma + self.bn[i] * np.sin(i * self.wr * self.t)
        return self.a0/2 + sigma


# # recommend to set tplot 1 cycle, not necessary
# tplot = np.arange(0, math.pi * 2, 0.01)
# xplot = f(tplot)
# plt.plot(tplot, xplot, label='rectangular wave')
# ft = FourierTransform(f, tplot, 2)
# ft5 = FourierTransform(f, tplot, 10)
# ft100 = FourierTransform(f, tplot, 100)
# plt.plot(tplot, ft.F(), label='n=2')
# plt.plot(tplot, ft5.F(), label='n=10')
# plt.plot(tplot, ft100.F(), label='n=100')
# plt.title('rectangular wave 1 cycle')
# plt.legend()
# plt.show()
#
# a0, an, bn = ft100.params()
# print(f"a[0] = {a0:.2f}")
# for i in range(100):
#     print(f"a[{i+1}] = {an[i]:.2f} b[{i+1}] = {bn[i]:.2f}")

