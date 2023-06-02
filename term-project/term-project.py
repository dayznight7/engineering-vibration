import numpy as np
import math
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from datetime import datetime


def getSensorData(filename_a, filename_t):

    from datetime import datetime
    data = np.loadtxt(filename_a, delimiter=',')
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    timestamps = []
    with open(filename_t, 'r') as file:
        for line in file:
            timestamp_str = line.strip()  # 각 줄의 시간 값을 읽어옴
            timestamp = datetime.strptime(timestamp_str, "%d-%b-%Y %H:%M:%S.%f")  # 문자열을 datetime 객체로 변환
            timestamps.append(timestamp)

    start_time = timestamps[0]  # 시작 시간은 첫 번째 요소
    time_array = [(timestamp - start_time).total_seconds() for timestamp in timestamps]  # 시작 시간과의 시간 차이를 초 단위로 계산
    t = np.array((time_array))

    return x, y, z, t


def getData(filename):
    data = np.loadtxt(filename, delimiter=',')
    return data


def get_zeta_wn_from_mck(m,c,k):
    return c/(2*(k*m)**0.5), (k/m)**0.5


def IRF(zeta, wn, a, tau, t, dt):

    wd = (wn*(1-zeta**2)**0.5)
    return np.where(t-tau < 0, 0, a*dt/wd * np.exp(-zeta*wn*(t-tau)) * np.sin(wd*(t-tau)))


x, y, z, t = getSensorData('data.txt', 't.txt')
z = z - np.average(z)


# # step0 : x, y,z raw data
#
# x_max = np.max(np.abs(x))
# x_max_index = np.argmax(np.abs(x))
# print(x_max, x_max_index)
# # 3.888302 55443
# x_abs_mean = np.abs(x).mean()
# y_abs_mean = np.abs(y).mean()
# z_abs_mean = np.abs(z).mean()
# print(x_abs_mean, y_abs_mean, z_abs_mean)
# # x_abs_mean = 0.5749981817770655
# # y_abs_mean = 0.39136066780064305
# # z_abs_mean = 0.27057284127283654
#
# plt.subplot(3, 1, 1)
# plt.plot(t, x)
# plt.title('x-axis acceleration')
# plt.subplot(3, 1, 2)
# plt.plot(t, y)
# plt.title('y-axis acceleration')
# plt.subplot(3, 1, 3)
# plt.plot(t, z)
# plt.title('z-axis accelerationz')
# plt.show()


# # step 0.5: checking IRF
# zeta = 0.3
# wn = 10
# xp = np.zeros_like(t)
# dt = (t[-1] - t[0])/len(t)
# print(x[55443])
# xp += IRF(zeta, wn, x[55443], t[55443], t, dt)
# plt.plot(t, xp)
# plt.show()


# # step1 : a(t) -> x(t) zeta=0.7, wn=0.1~100
#
# zeta = 0.7
# wn = np.array([[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10,20,30,40,50,60,70,80,90,100]]).T
# t_tile = np.tile(t, (len(wn), 1))
# xp = np.zeros_like(t_tile)
# dt = (t[-1] - t[0])/len(t)
#
# for i in range(len(t)):
#     xp += IRF(zeta, wn, x[i], t[i], t_tile, dt)
#     print(i)
#
# for i in range(28):
#     np.savetxt('x_wn['+str(i) +'].txt', xp[i], delimiter='\n')


# # step1.5 : plot x(t)
# xp_z03wn01= getData('xp_z0.3wn0.1.txt')
# xp_z03wn1 = getData('xp_z0.3wn1.txt')
# xp_z03wn5 = getData('xp_z0.3wn5.txt')
# xp_z03wn10= getData('xp_z0.3wn10.txt')
# xp_z03wn20= getData('xp_z0.3wn20.txt')
# xp_z03wn50= getData('xp_z0.3wn50.txt')
# # plt.subplot(3, 2, 1)
# plt.plot(t, xp_z03wn01, label='wn=0.1')
# # plt.legend()
# # plt.subplot(3, 2, 2)
# plt.plot(t, xp_z03wn1, label='wn=1')
# # plt.legend()
# # plt.subplot(3, 2, 3)
# plt.plot(t, xp_z03wn5, label='wn=5')
# # plt.legend()
# # plt.subplot(3, 2, 4)
# plt.plot(t, xp_z03wn10, label='wn=10')
# plt.legend()
# # plt.subplot(3, 2, 5)
# plt.plot(t, xp_z03wn20, label='wn=20')
# # plt.legend()
# # plt.subplot(3, 2, 6)
# plt.plot(t, xp_z03wn50, label='wn=50')
# plt.legend()
# plt.show()


# # step2 : bring x(t) -> v(t) -> vrms
#
# z01_vrms = np.zeros(28)
# z03_vrms = np.zeros(28)
# z05_vrms = np.zeros(28)
# z07_vrms = np.zeros(28)
# for i in range(28):
#     xp01 = getData('z01/xp_z01_' + str(i) + '.txt')
#     xp03 = getData('z03/xp_z03_' + str(i) + '.txt')
#     xp05 = getData('z05/xp_z05_' + str(i) + '.txt')
#     xp07 = getData('z07/xp_z07_' + str(i) + '.txt')
#     dt = (t[-1] - t[0]) / len(t)
#     dx01 = np.diff(xp01)
#     dx03 = np.diff(xp03)
#     dx05 = np.diff(xp05)
#     dx07 = np.diff(xp07)
#     v01 = dx01 / dt
#     v03 = dx03 / dt
#     v05 = dx05 / dt
#     v07 = dx07 / dt
#     z01_vrms[i] = np.sqrt(np.mean(np.square(v01)))
#     z03_vrms[i] = np.sqrt(np.mean(np.square(v03)))
#     z05_vrms[i] = np.sqrt(np.mean(np.square(v05)))
#     z07_vrms[i] = np.sqrt(np.mean(np.square(v07)))
#
#
# np.savetxt('z01_vrms.txt', z01_vrms, delimiter='\n')
# np.savetxt('z03_vrms.txt', z03_vrms, delimiter='\n')
# np.savetxt('z05_vrms.txt', z05_vrms, delimiter='\n')
# np.savetxt('z07_vrms.txt', z07_vrms, delimiter='\n')


# # step2.5 : plot v(t)
# xp_z03wn01= getData('xp_z0.3wn0.1.txt')
# xp_z03wn1 = getData('xp_z0.3wn1.txt')
# xp_z03wn5 = getData('xp_z0.3wn5.txt')
# xp_z03wn10= getData('xp_z0.3wn10.txt')
# xp_z03wn20= getData('xp_z0.3wn20.txt')
# xp_z03wn50= getData('xp_z0.3wn50.txt')
# dt = (t[-1] - t[0]) / len(t)
# v01 = np.diff(xp_z03wn01)/dt
# v1 = np.diff(xp_z03wn1)/dt
# v5 = np.diff(xp_z03wn5)/dt
# v10 = np.diff(xp_z03wn10)/dt
# v20 = np.diff(xp_z03wn20)/dt
# v50 = np.diff(xp_z03wn50)/dt
# plt.subplot(3, 2, 1)
# plt.plot(t[1:],v01,label='wn=0.1')
# plt.legend()
# plt.subplot(3, 2, 2)
# plt.plot(t[1:],v1,label='wn=1')
# plt.legend()
# plt.subplot(3, 2, 3)
# plt.plot(t[1:],v5,label='wn=5')
# plt.legend()
# plt.subplot(3, 2, 4)
# plt.plot(t[1:],v10,label='wn=10')
# plt.legend()
# plt.subplot(3, 2, 5)
# plt.plot(t[1:],v20,label='wn=20')
# plt.legend()
# plt.subplot(3, 2, 6)
# plt.plot(t[1:],v50,label='wn=50')
# plt.legend()
# plt.show()



# step3 : x-axis: wn y-axis: vrms

z01_vrms = getData('z01_vrms.txt')
z03_vrms = getData('z03_vrms.txt')
z05_vrms = getData('z05_vrms.txt')
z07_vrms = getData('z07_vrms.txt')
wn = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10,20,30,40,50,60,70,80,90,100])
plt.plot(wn, z01_vrms, label='zeta = 0.1')
plt.plot(wn, z03_vrms, label='zeta = 0.3')
plt.plot(wn, z05_vrms, label='zeta = 0.5')
plt.plot(wn, z07_vrms, label='zeta = 0.7')
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
plt.xlabel('wn')
plt.ylabel('vrms')
plt.legend()
plt.title('Vrms of spring mass damper system in bus')
plt.show()

