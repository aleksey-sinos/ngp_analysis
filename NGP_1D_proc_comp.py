import numpy as np
from numpy import dot
from filterpy.monte_carlo import residual_resample, systematic_resample
from numpy.linalg import norm
from numpy.random import randn
import matplotlib.pyplot as plt
import matplotlib
from numpy.random import uniform
import scipy.stats
from filterpy.kalman import KalmanFilter
import filterpy as fp
from filterpy.common import Q_discrete_white_noise
from scipy import interpolate
from numpy.random import seed
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import seaborn as sns
from scipy import signal
from filterpy.common import dot3

ss = signal.StateSpace

mdls = {}


def InitModels(smpl, **kwargs):
    len = kwargs['len']
    dt = kwargs['dt']  # интервал решения задачи [с]
    time = smpl * dt  # время решения задачи [c]
    V = len / time  # скорость в [м/c]
    dgdl = kwargs['dgdl']
    sg_ga = kwargs['sg_ga']

    # M1:
    dgdt = dgdl * V  # градиент поля[мГал / с]
    tau_ga = sg_ga / dgdt  # интервал корреляции поля, соответствующий h [c]
    alpha = 1 / tau_ga  # величина, обратная интервалу корреляции поля [1/с]
    q_w = np.sqrt(2 * sg_ga ** 2 * alpha)  # СКО порождающего шума в модели поля
    F = np.array([[-alpha]])
    G = np.array([[q_w]])
    Fd, Gd = discm(F, G, dt, 20, 0)
    sys = ss(Fd, Gd, 1, 0, dt=dt)
    sys.P0 = sg_ga ** 2
    mdls['M1'] = sys

    # Jordan
    DZETA = (np.sqrt(5) - 1) / np.sqrt(5)

    beta = V * dgdl / sg_ga / np.sqrt(2)
    F = np.zeros([3, 3])
    G = np.zeros([3, 1])
    C = np.zeros([1, 3])
    C[0, 0] = -beta * DZETA
    C[0, 1] = 1

    F[0, 0] = -beta
    F[1, 1] = -beta
    F[2, 2] = -beta
    F[0, 1] = 1
    F[1, 2] = 1
    G[2, 0] = sg_ga * np.sqrt(10 * beta ** 3)

    Fd, Gd = discm(F, G, dt, 20, 0)
    sys = ss(Fd, Gd, C, 0, dt=dt)

    # получение установившейся матрицы ковариаций
    STAB_THRESHOLD = 0.01
    P0 = np.zeros([3, 3])
    for i in range(1, 100000):
        P0 = Fd @ P0 @ Fd.T + np.dot(Gd, Gd.T)
        if np.abs(np.sqrt(C @ P0 @ C.T) - sg_ga) / sg_ga > STAB_THRESHOLD:
            stab_flag = 0
        else:
            stab_flag += 1
            if stab_flag > 100:
                # print("Jordan mdl cov calculated, steps:", i)
                break
        if i == 99999:
            print("Jordan mdl cov failed")
    sys.P0 = P0
    mdls['Jordan'] = sys


def discm(A, B, dt, N, mode):
    n = np.size(A, 0)
    r = np.size(B, 1)
    Adt = A * dt
    F = np.eye(n)
    for i in range(N, 0, -1):
        F = np.eye(n) + (Adt / i).dot(F)
    if r:
        Gm = np.eye(n)
        for i in range(N, 0, -1):
            Gm = np.eye(n) + Adt.dot(Gm) / (i + 1)
        if mode == 0:
            G = dot(Gm, B) * np.sqrt(dt)
        else:
            G = Gm

    else:
        G = np.array([])

    return F, G


def GenerateProfile(type, smpl, **kwargs):
    # V = kwargs['V']  # скорость в [м/с]
    len = kwargs['len']
    dt = kwargs['dt']  # интервал решения задачи [с]
    time = smpl * dt  # время решения задачи [c]

    V = len / time  # скорость в [м/c]

    if type == 'linear':
        len = kwargs['len']
        dgdl = kwargs['dgdl']
        fld = np.empty([3, smpl])
        fld[0, :] = np.linspace(0, len, smpl, endpoint=False)
        fld[1, :] = np.linspace(0, len * dgdl, smpl, endpoint=False)
        fld[2, :] = np.ones(smpl) * dgdl

    if type == 'M1':
        dgdl = kwargs['dgdl']
        sg_ga = kwargs['sg_ga']
        dgdt = dgdl * V  # градиент поля[мГал / с]
        tau_ga = sg_ga / dgdt  # интервал корреляции поля, соответствующий h [c]
        alpha = 1 / tau_ga  # величина, обратная интервалу корреляции поля [1/с]
        q_w = np.sqrt(2 * sg_ga ** 2 * alpha)  # СКО порождающего шума в модели поля
        F = np.array([[-alpha]])
        G = np.array([[q_w]])
        Fd, Gd = discm(F, G, dt, 20, 0)
        fld = np.empty([3, smpl])
        fld[0, :] = np.linspace(0, len, smpl, endpoint=False)
        fld[1, 0] = np.dot(Gd, np.random.randn()).ravel()
        for i in range(1, smpl):
            n = np.random.randn(Gd.shape[1])
            fld[1, i] = np.dot(Fd, fld[1, i - 1]) + np.dot(Gd, n)
        fld[2, :] = np.gradient(fld[1, :])

    if type == 'M2':
        len = kwargs['len']
        dgdl = kwargs['dgdl']
        sg_ga = kwargs['sg_ga']
        dl = len / smpl  # количество метров на один отсчет [м/отсчет]
        dgds = dgdl * dl  # градиент поля в [мГал/м] переведенный через [м/отсчет]  в [мГал/отсчет]

        '''alpha = (dgds ** 2 / (sg_ga ** 2 * 2)) ** 0.5
        beta = dgds / sg_ga'''

        alpha = 0  # (dgds ** 2 / (sg_ga ** 2 * 2)) ** 0.5
        beta = dgds / sg_ga

        F = np.array([[0, 1], [-(alpha ** 2 + beta ** 2), -2 * alpha]])
        G = np.array([[0], [np.sqrt(4 * alpha * (sg_ga ** 2) * (alpha ** 2 + beta ** 2))]])
        Fd, Gd = discm(F, G, dt, 20, 0)
        fld = np.empty([3, smpl])
        fld[0, :] = np.linspace(0, len, smpl, endpoint=False)
        fld[1:3, 0] = np.dot(Gd, np.random.randn()).ravel()
        for i in range(1, smpl):
            n = np.random.randn(Gd.shape[1])
            fld[1:3, i] = np.dot(Fd, fld[1:3, i - 1]) + np.dot(Gd, n)
    if type == 'Jordan':
        dgdl = kwargs['dgdl']
        sg_ga = kwargs['sg_ga']
        Fd = mdls['Jordan'].A
        Gd = mdls['Jordan'].B
        C = mdls['Jordan'].C
        P0 = mdls['Jordan'].P0

        x = np.zeros([3, smpl])
        fld = np.zeros([3, smpl])
        fld[0, :] = np.linspace(0, len, smpl, endpoint=False)
        # x[1, 0] = np.dot(sg_ga, np.random.randn())
        x[:, 0] = np.random.multivariate_normal([0, 0, 0], P0)
        for i in range(1, smpl):
            n = np.random.randn(Gd.shape[1])
            x[:, i] = np.dot(Fd, x[:, i - 1]) + np.dot(Gd, n)
        fld[1, :] = np.dot(C, x)
        fld[2, :] = np.dot(np.gradient(fld[1, :]), smpl / len)

    if type == 'sin':
        len = kwargs['len']
        p = 300
        sg_ga = kwargs['sg_ga']
        fld = np.empty([3, smpl])
        fld[0, :] = np.linspace(0, len, smpl, endpoint=False)
        fld[1, :] = sg_ga * np.sin(fld[0, :] / p)
        fld[2, :] = 1 / p * sg_ga * np.cos(fld[0, :] / p)
        # fld[2, :] = np.dot(np.gradient(fld[1, :]), smpl / len)

    if 'offset' in kwargs:
        fld[1, :] = fld[1, :] + kwargs['offset']

    f = scipy.interpolate.interp1d(fld[0, :], fld, fill_value='extrapolate')
    return fld, f


def ModelMeasurements(fld, smpl, dt, r):
    # q_d = r / np.sqrt(dt)
    err = r * np.random.randn(smpl)
    mnt = fld[1, :] + err
    m = scipy.interpolate.interp1d(fld[0, :], mnt, fill_value='extrapolate')
    return mnt, m


class KF(KalmanFilter):
    def __init__(self, sys, P0, r):
        KalmanFilter.__init__(self, dim_x=sys.A.shape[0], dim_z=sys.C.shape[0])
        self.F = sys.A
        self.H = sys.C
        self.P = P0
        self.Q = dot(sys.B, sys.B.T)
        self.R = r ** 2
        self.x = np.zeros([sys.A.shape[0], 1])
        self.mean = []
        self.var = []

    def estimate(self):
        self.mean.extend([self.x])
        self.var.extend([self.P])
    def update_cov(self, R=None, H=None):
        """
        Add a new measurement (z) to the Kalman filter. If z is None, nothing
        is changed.

        Parameters
        ----------
        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.R will be used.

        H : np.array, or None
            Optionally provide H to override the measurement function for this
            one call, otherwise self.H will be used.
        """

        if R is None:
            R = self.R
        elif isscalar(R):
            R = eye(self.dim_z) * R

        # rename for readability and a tiny extra bit of speed
        if H is None:
            H = self.H
        P = self._P

        # S = HPH' + R
        # project system uncertainty into measurement space
        S = dot3(H, P, H.T) + R

        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        K = dot3(P, H.T, np.linalg.inv(S))

        # P = (I-KH)P(I-KH)' + KRK'
        I_KH = self._I - dot(K, H)
        self._P = dot3(I_KH, P, I_KH.T) + dot3(K, R, K.T)

        self._S = S
        self._K = K




class PF:
    def __init__(self, N):
        self.N = N
        self.particles = np.zeros(N)
        self.weights = np.zeros(N)
        self.mean = []
        self.var = []

    def create_gaussian_particles(self, mean, std):
        self.particles = mean + (randn(self.N) * std)
        self.weights.fill(1.)

    def update(self, mnt, r, ns_pos, map):
        map_value = map(ns_pos - self.particles)[1]
        self.weights *= scipy.stats.norm(map_value, r).pdf(mnt).ravel()
        # self.weights += 1.e-300  # avoid round-off to zero
        self.weights /= sum(self.weights)  # normalize

    def predict(self, std):
        self.particles += randn(self.N) * std

    def estimate(self):
        self.mean.extend([np.average(self.particles, weights=self.weights, axis=0)])
        self.var.extend([np.average((self.particles - self.mean[-1]) ** 2, weights=self.weights, axis=0)])

    def neff(self, weights):
        return 1. / np.sum(np.square(weights))

    def resample_from_index(self):
        indexes = systematic_resample(self.weights)
        self.particles[:] = self.particles[indexes]
        self.weights[:] = self.weights[indexes]
        self.weights.fill(1.0 / self.N)


def find_stab(data, th, th_ln):
    l = data.shape[0]
    # ddata = np.diff(data)
    st_index = -1
    for i in range(l):
        if (np.abs(data[i] - data[i + th_ln]) < th):
            st_index = i
            break
    return st_index

def CRLB(map):  # не используются измерения. Только ковариационный канал фильтра

    # kf_init
    F = np.array([1])
    G = np.array([0])
    sys = ss(F, G, 1, 0, dt=dt)
    sys.P0 = sg_tau ** 2
    P_crlb_buf = []

    mse_num = 1  # количество реализаций для CRLB
    for mse in range(mse_num):

        tau = sg_tau * randn()
        kf = KF(sys, sg_tau ** 2, r)
        kf.estimate()

        # todo взять из поля производную, посчитать точность в ekf, осреднить
        for i in range(mnt_num):  # длина реализации
            pos = start_pos + V * i
            ns_pos = pos + tau
            # kf.predict()
            kf.update_cov(H=map(pos)[2])
            kf.estimate()
        P_crlb_buf.append(kf.var)
    P_crlb_buf = np.array(P_crlb_buf)
    P_n_sum = np.sum(P_crlb_buf, axis=0) / mse_num
    return P_n_sum


def pf_sim(nav_path):
    err_n_buf, err_f_buf, err_s_buf, err_kf_buf = [], [], [], []
    P_n_buf, P_f_buf, P_s_buf, P_kf_buf = [], [], [], []
    tau_est_n_buf, tau_est_f_buf, tau_est_s_buf, tau_est_kf_buf = [], [], [], []

    mse_num = 100
    for mse in range(mse_num):
        print('Итерация', mse + 1)
        seed()
        mnt_v, mnt_interp = ModelMeasurements(map_v, smpl, dt, r)

        # Начальные условия для задачи навигации

        tau = sg_tau * randn()  # действительное значение tau

        print('Истинное значение tau', tau, "м.", 'Априорное СКО:', sg_tau, "м.")

        # Инициализация навигационных фильтров

        p_number = 1000
        pf_n = PF(p_number)
        pf_n.create_gaussian_particles(0, sg_tau)
        pf_n.estimate()

        pf_f = PF(p_number)
        pf_f.create_gaussian_particles(0, sg_tau)
        pf_f.estimate()

        pf_s = PF(p_number)
        pf_s.create_gaussian_particles(0, sg_tau)
        pf_s.estimate()

        # # kf_init
        # F = np.array([1])
        # G = np.array([0])
        # sys = ss(F, G, 1, 0, dt=dt)
        # sys.P0 = sg_tau ** 2
        # kf = KF(sys, sg_tau ** 2, r)
        # kf.estimate()

        print('Интервал измерений в новом подходе', V * dt, "м")

        # инициализация фильтра предварительной обработки

        unsample_f = 100  # [м]
        unsample_s = 50  # [м]

        P0 = mdl.P0
        prf = KF(mdl, P0, r)

        # фильтрация
        print('Предварительная фильтрация измерений...')
        (mu, cov, _, _) = prf.batch_filter(mnt_interp(nav_path))
        P_f = (mdl.C @ cov @ mdl.C.T).ravel()
        mnt_f_interp = scipy.interpolate.interp1d(nav_path, (mdl.C @ mu).ravel(), fill_value=0)
        P_f_interp = scipy.interpolate.interp1d(nav_path, P_f, fill_value=0)

        pr_f_err = (map_interp(nav_path)[1] - mnt_f_interp(nav_path))
        st_step_f = find_stab(P_f, 0.1, 50)
        print('Установившийся режим для предварительной фильтрации', st_step_f, 'шага - ', st_step_f * V * dt, 'м')
        print('Установившееся СКО ошибки оценивания поля:', np.sqrt(P_f[st_step_f]), 'мГал')
        print('Интервал измерений для предварительной фильтрации', unsample_f, "м")

        # сглажиавание
        print('Предварительное сглаживание измерений...')
        (x, P, K) = prf.rts_smoother(mu, cov)
        P_s = (mdl.C @ P @ mdl.C.T).ravel()
        mnt_s_interp = scipy.interpolate.interp1d(nav_path, (mdl.C @ x).ravel(), fill_value=0)
        P_s_interp = scipy.interpolate.interp1d(nav_path, P_s, fill_value=0)

        pr_s_err = (map_interp(nav_path)[1] - mnt_s_interp(nav_path))
        st_step_s = find_stab(P_s, 0.1, 50)
        print('Установившийся режим для предварительного сглаживания', st_step_s, 'шага - ', st_step_s * V * dt,
              'м')
        print('Установившееся СКО ошибки оценивания поля:', np.sqrt(P_s[st_step_s]), 'мГал')
        print('Интервал измерений для предварительной фильтрации', unsample_s, "м")

        # Моделирование работы алгоритмов оценивания

        st_state = True
        est_ready = True

        mnt_cnt_n, mnt_cnt_f, mnt_cnt_s = 0, 0, 0

        print('Оценивание ошибок НС...')
        for i in range(mnt_num):
            pos = start_pos + V * i
            ns_pos = pos + tau

            # формирование измерений на шаге
            mnt = mnt_interp(pos)
            mnt_f = mnt_f_interp(pos)
            mnt_s = mnt_s_interp(pos)

            if st_state == True:
                # фильтр без обработки
                mnt_cnt_n += 1
                pf_n.update(mnt, r, ns_pos, map_interp)
                pf_n.estimate()

                # EKF
                # ekf_mnt = map_interp(ns_pos)[1] - mnt
                # kf.predict()
                # kf.update(ekf_mnt, r ** 2, map_interp(ns_pos - kf.x)[2])
                # # kf.update(ekf_mnt, r**2, map_interp(pos)[2])
                # kf.estimate()

                # с фильтрацией
                if np.mod(pos - start_pos, unsample_f) == 0:
                    mnt_cnt_f += 1
                    pf_f.update(mnt_f, P_f_interp(pos), ns_pos, map_interp)
                    pf_f.estimate()
                # со сглаживанием
                if np.mod(pos - start_pos, unsample_s) == 0:
                    mnt_cnt_s += 1
                    pf_s.update(mnt_s, P_s_interp(pos), ns_pos, map_interp)
                    pf_s.estimate()
            if np.mod(i, 100) == 0:
                print('Шаг', i)
                # pf.predict(1)
                # if pf.neff(pf.weights) < pf.N / 2:
                #     pf.resample_from_index()
                # if np.mod(i, 20) == 0:
                #     data = np.array([pf_n.particles, pf_n.weights])
                #     data = data[:, np.argsort(data[0])]
                #     ax.plot(data[0], np.ones(p_number) * i, zs=data[1], zdir='z')

        err_n = (tau - np.array(pf_n.mean)) ** 2
        err_f = (tau - np.array(pf_f.mean)) ** 2
        err_s = (tau - np.array(pf_s.mean)) ** 2
        # err_kf = np.abs(tau - np.array(kf.mean))

        err_n_buf.append(err_n)
        err_f_buf.append(err_f)
        err_s_buf.append(err_s)
        # err_kf_buf.append(err_kf)

        P_n_buf.append(pf_n.var)
        P_f_buf.append(pf_f.var)
        P_s_buf.append(pf_s.var)
        # P_kf_buf.append(kf.var)

        # tau_est_n_buf.append(pf_n.mean)
        # tau_est_f_buf.append(pf_f.mean)
        # tau_est_s_buf.append(pf_s.mean)

    # расчет средней ошибки
    err_n_buf = np.array(err_n_buf)
    err_f_buf = np.array(err_f_buf)
    err_s_buf = np.array(err_s_buf)
    err_kf_buf = np.array(err_kf_buf)

    err_n_sum = np.sum(err_n_buf, axis=0) / mse_num
    err_f_sum = np.sum(err_f_buf, axis=0) / mse_num
    err_s_sum = np.sum(err_s_buf, axis=0) / mse_num
    # err_kf_sum = np.sum(err_kf_buf, axis=0) / mse_num

    # расчет средней рассчетной дисперсии ошибки
    P_n_buf = np.array(P_n_buf)
    P_f_buf = np.array(P_f_buf)
    P_s_buf = np.array(P_s_buf)
    # P_kf_buf = np.array(P_kf_buf)

    P_n_sum = np.sum(P_n_buf, axis=0) / mse_num
    P_f_sum = np.sum(P_f_buf, axis=0) / mse_num
    P_s_sum = np.sum(P_s_buf, axis=0) / mse_num
    # P_kf_sum = np.sum(P_kf_buf, axis=0) / mse_num

    # # расчет средней оценки
    # tau_est_n_buf = np.array(tau_est_n_buf)
    # tau_est_f_buf = np.array(tau_est_f_buf)
    # tau_est_s_buf = np.array(tau_est_s_buf)
    #
    # tau_est_n_sum = np.sum(tau_est_n_buf, axis=0)/mse_num
    # tau_est_f_sum = np.sum(tau_est_f_buf, axis=0)/mse_num
    # tau_est_s_sum = np.sum(tau_est_s_buf, axis=0)/mse_num



    # print("Использованная для навигации длина траектории", V * mnt_num, "м")
    #
    # print('Новый подход: Оценка tau', pf_n.mean[-1], 'СКО tau', np.sqrt(pf_n.var[-1]), 'по', mnt_cnt_n, 'измерениям. Ошибка:',
    #       err_n[-1], 'м')
    # print('C предварительной фильтрацией: Оценка tau', pf_f.mean[-1], 'СКО tau', np.sqrt(pf_f.var[-1]), 'по', mnt_cnt_f,
    #       'измерениям. Ошибка:', err_f[-1], 'м')
    # print('C предварительным сглаживанием: Оценка tau', pf_s.mean[-1], 'СКО tau', np.sqrt(pf_s.var[-1]), 'по', mnt_cnt_s,
    #       'измерениям. Ошибка:', err_s[-1], 'м')

    print("Использованная для навигации длина траектории", V * mnt_num, "м")

    print('Новый подход: среднее СКО tau', np.sqrt(P_n_sum[-1]), 'по', mnt_cnt_n, 'измерениям. Средняя ошибка:',
          err_n_sum[-1], 'м')
    print('C предварительной фильтрацией: среднее СКО tau', np.sqrt(P_f_sum[-1]), 'по', mnt_cnt_f,
          'измерениям. Средняя ошибка:', err_f_sum[-1], 'м')
    print('C предварительным сглаживанием: среднее СКО tau', np.sqrt(P_s_sum[-1]), 'по', mnt_cnt_s,
          'измерениям. Средняя ошибка:', err_s_sum[-1], 'м')
    # print('EKF: среднее СКО tau', np.sqrt(P_kf_sum[-1]), 'по', mnt_cnt_n,
    #       'измерениям. Средняя ошибка:', err_kf_sum[-1], 'м')

    nav_path = np.arange(start_pos, start_pos + V * dt * mnt_cnt_n, V * dt)
    nav_path_P = np.arange(start_pos, start_pos + V * dt * mnt_cnt_n + 1, V * dt)
    nav_path_f = np.arange(start_pos, start_pos + unsample_f * mnt_cnt_f + 1, unsample_f)
    nav_path_s = np.arange(start_pos, start_pos + unsample_s * mnt_cnt_s + 1, unsample_s)

    # Графики поля
    matplotlib.rcParams.update({'font.size': 16})
    plt.figure(figsize=(16, 9))
    ax = plt.gca()
    # plt.subplot(211)
    real = ax.plot(map_v[0, :], map_v[1, :] - 1000, linewidth=1)
    mnt = ax.plot(map_v[0, :], mnt_v - 1000, alpha=0.5, marker='.', markersize=3, color='C1', linestyle='None')

    filt = ax.plot(nav_path, mnt_f_interp(nav_path) - 1000, alpha=0.8, linewidth=3, linestyle=':', color='C2')
    smooth = ax.plot(nav_path, mnt_s_interp(nav_path) - 1000, alpha=0.8, linestyle='-.', linewidth=3, color='C3')
    st = ax.plot(start_pos, map_interp(start_pos)[1] - 1000, ">", markersize=10, color='C4')
    en = ax.plot(start_pos + V * dt * mnt_cnt_n, map_interp(start_pos + V * dt * mnt_cnt_n)[1] - 1000, "s",
                 markersize=10, color='C4')
    plt.xlim([2000, 9000])
    ax.legend(['Профиль дна',
               'Исходные измерения',
               'Предварительная фильтрация',
               'Предварительное слгаживание',
               'Начальная точка',
               'Конечная точка'
               ], loc=3, fontsize=16)
    plt.grid()
    plt.gca().set_xlabel('[М]', fontsize=16)
    plt.gca().set_ylabel('[М]', fontsize=16)

    # plt.subplot(212)
    # plt.plot(map_v[0, :], map_v[2, :])
    # plt.legend(['Производная поля'])
    # plt.gca().set_xlabel('[М]', fontsize=12)
    # plt.gca().set_ylabel('[ед.]/[М]', fontsize=12)

    # plt.figure()
    # plt.plot(nav_path, 3 * np.sqrt(P_f_interp(nav_path)))
    # plt.plot(nav_path, np.abs(pr_f_err))
    # plt.plot(nav_path, 3 * np.sqrt(P_s_interp(nav_path)))
    # plt.plot(nav_path, np.abs(pr_s_err))

    # plt.draw()
    # plt.pause(0.001)

    plt.figure(2)
    plt.plot(nav_path_P, np.sqrt(P_n_sum), color='C0')
    plt.plot(nav_path_f, np.sqrt(P_f_sum), color='C1')
    plt.plot(nav_path_s, np.sqrt(P_s_sum), color='C2')
    # plt.plot(nav_path, 3 * np.sqrt(P_kf_sum.ravel()), color='C3')

    plt.plot(nav_path_P, np.sqrt(err_n_sum), linestyle='--', color='C0')
    plt.plot(nav_path_f, np.sqrt(err_f_sum), linestyle='--', color='C1')
    plt.plot(nav_path_s, np.sqrt(err_s_sum), linestyle='--', color='C2')
    # plt.plot(nav_path, err_kf_sum.ravel(), linestyle='--', color='C3')

    plt.legend([' СКО расчетное (PF)',
                'СКО расчетное (фильтрация)',
                'СКО расчетное(сглаживание)',
                # ' СКО расчетное(EKF)',
                'СКО действительное (PF)',
                'СКО действительное (фильтрация)',
                'СКО действительное (сглаживание)',
                # 'Ошибка (EKF)'
                ], fontsize='12')
    plt.grid()
    plt.gca().set_xlabel('[М]', fontsize=12)
    plt.gca().set_ylabel('СКО [ед.]', fontsize=12)
    plt.show()



# Параметры моделирования
seed(10)  #4 #7 #10
len = 10000  # длинна траектории [м]
smpl = 2000  # количество отсчетов
dt = 1  # [с]

path = np.linspace(0, len, smpl, endpoint=False)
# dl = len / smpl  # пространственный интервал решения [м]
V = 5
r = 20  # корень из интенсивности шума измерений [мГал*с^-1]
sg_ga = 30  # СКО полезного сигнала
dgdl = 50 / 1000  # СКО производной полезного сигнала [мГал / м]
dgdt = dgdl * V  # СКО производной полезного сигнала [мГал / с]

sg_tau = 500  # СКО погрешности НС
start_pos = 3000  # действительное начальное местоположение
mnt_num = 1000
nav_path = np.linspace(start_pos, start_pos + mnt_num * V, mnt_num + 1, endpoint=False)

InitModels(smpl, sg_ga=sg_ga, dgdl=dgdl, len=len, dt=dt)

mdl = mdls['M1']

# Подготовка поля и его измерений
map_v, map_interp = GenerateProfile('M1', smpl, dgdl=dgdl, len=len, sg_ga=sg_ga, dt=dt, offset=0)
print("СКО поля:", sg_ga, "мГал.",
      "СКО производной поля:", dgdl, "мГал / м.",
      "СКО ошибки измерений:", r, "мГал.")

pf_sim(nav_path)
# P_crlb = CRLB(map_interp)

# plt.figure(2)
# plt.plot(nav_path, np.sqrt(P_crlb.ravel()), color='C1')
# plt.grid()
