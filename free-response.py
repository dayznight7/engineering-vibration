import numpy as np
import math
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


class SpringMassSystem:

    def __init__(self, m, k):
        self.m = m
        self.k = k
        self.wn = (k/m)**0.5
        self.A = 1
        self.phi = 0.5 * math.pi

    def initCondition(self, x0, v0):
        self.A = (self.wn**2 * x0**2 + v0**2)**0.5/self.wn
        self.phi = math.atan2(self.wn * x0, v0)

    def initCondition2(self, A, phi):
        self.A = A
        self.phi = phi

    def params(self):
        return self.m, self.k, self.wn, self.A, self.phi

    def x(self, t):
        return self.A * np.sin(self.wn * t + self.phi)

    def v(self, t):
        return self.wn * self.A * np.cos(self.wn * t + self.phi)

    def a(self, t):
        return -self.wn**2 * self.A * np.sin(self.wn * t + self.phi)


# # window 1.3
# # ma + kx = 0
# tplot = np.arange(0, 10, 0.01)
# km = SpringMassSystem(4, 36)
# km.initCondition(2, 1)
# xplot = km.x(tplot)
# vplot = km.v(tplot)
# aplot = km.a(tplot)
# plt.plot(tplot, xplot, label='x')
# plt.plot(tplot, vplot, label='v')
# plt.plot(tplot, aplot, label='a')
# plt.title('mx" + kx = 0')
# plt.legend()
# plt.show()


class ViscousDamping:

    def __init__(self, m, c, k):
        self.m = m
        self.c = c
        self.k = k
        self.wn = (k / m) ** 0.5
        self.zeta = 0.5 * c * (k * m)**-0.5


class Overdamped(ViscousDamping):

    def initCondition(self, x0, v0):
        self.a1 = (-v0 + (-self.zeta + (self.zeta**2 - 1)**0.5) * self.wn * x0) / (2 * self.wn * (self.zeta**2 - 1)**0.5)
        self.a2 = (v0 + (self.zeta + (self.zeta**2 - 1)**0.5) * self.wn * x0) / (2 * self.wn * (self.zeta**2 - 1)**0.5)

    def x(self, t):
        return self.a1 * np.exp((-self.zeta - (self.zeta**2 - 1)**0.5) * self.wn * t) + self.a2 * np.exp((-self.zeta + (self.zeta**2 - 1)**0.5) * self.wn * t)


# # fig 1.11
# # ma + cv + kx = 0, zeta > 1
# tplot = np.arange(0, 3, 0.01)
# kcm = Overdamped(100, 600, 225)   # Ccr = 2*sqrt(km) = 300
# kcm.initCondition(0.3, 0)
# xplot = kcm.x(tplot)
# kcm.initCondition(0, 1)
# xplot2 = kcm.x(tplot)
# kcm.initCondition(-0.3, 0)
# xplot3 = kcm.x(tplot)
# plt.plot(tplot, xplot, label='x0=0.3,  v0=0')
# plt.plot(tplot, xplot2, label='x0=0,     v0=1')
# plt.plot(tplot, xplot3, label='x0=-0.3, v0=0')
# plt.title('mx" + cx\' + kx = 0, zeta >1')
# plt.legend()
# plt.show()


class Criticallydamped(ViscousDamping):

    def initCondition(self, x0, v0):
        self.a1 = x0
        self.a2 = v0+self.wn * x0

    def x(self, t):
        return (self.a1 + self.a2 * t) * np.exp(-self.wn * t)


# # fig 1.12
# # ma + cv + kx = 0, zeta = 1
# tplot = np.arange(0, 3, 0.01)
# kcm = Criticallydamped(100, 300, 225)   # Ccr = 2*sqrt(km) = 300
# kcm.initCondition(0.4, 1)
# xplot = kcm.x(tplot)
# kcm.initCondition(0.4, 0)
# xplot2 = kcm.x(tplot)
# kcm.initCondition(0.4, -1)
# xplot3 = kcm.x(tplot)
# plt.plot(tplot, xplot, label='x0=0.4, v0=+1')
# plt.plot(tplot, xplot2, label='x0=0.4, v0=0')
# plt.plot(tplot, xplot3, label='x0=0.4, v0=-1')
# plt.title('mx" + cx\' + kx = 0, zeta =1')
# plt.legend()
# plt.show()


class Underdamped(ViscousDamping):

    def initCondition(self, x0, v0):
        self.wd = self.wn * (1- self.zeta**2)**0.5
        self.A = (((v0 + self.zeta * self.wn * x0)**2 + (x0*self.wd)**2) / (self.wd**0.5)) ** 0.5
        self.phi = math.atan2(x0 * self.wd, v0 + self.zeta * self.wn * x0)

    def changeParam(self, A, phi):
        self.wd = self.wn * (1- self.zeta**2)**0.5
        self.A = A
        self.phi = phi

    def x(self, t):
        return self.A * np.exp(-self.zeta * self.wn * t) * np.sin(self.wd * t + self.phi)


# # ex 1.3.2
# # ma + cv + kx = 0, zeta < 1
# # x" + (2 zeta wn)x' + (wn^2)x = 0
# tplot = np.arange(0, 0.14, 0.001)
# kcm = Underdamped(1, 2*125.66*0.224, 125.66**2)
# kcm.changeParam(0.005, 0)
# xplot = kcm.x(tplot)
# plt.plot(tplot, xplot)
# plt.title('mx" + cx\' + kx = 0, zeta <1')
# plt.show()

