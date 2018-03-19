import numpy as np
from numpy import dot
from filterpy.monte_carlo import residual_resample, systematic_resample
from numpy.linalg import norm
from numpy.random import randn
import matplotlib.pyplot as plt
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
        fld[2, :] = np.gradient(fld[1, :])

    if type == 'sin':
        len = kwargs['len']
        p = 1000
        fld = np.empty([3, smpl])
        fld[0, :] = np.linspace(0, len, smpl, endpoint=False)
        fld[1, :] = np.sin(fld[0, :] / p)
        fld[2, :] = 1 / p * np.cos(fld[0, :] / p)

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


class Pre_Filter(KalmanFilter):
    def __init__(self, sys, P0, r):
        KalmanFilter.__init__(self, dim_x=sys.A.shape[0], dim_z=sys.C.shape[0])
        self.F = sys.A
        self.H = sys.C
        self.P = P0
        self.Q = dot(sys.B, sys.B.T)
        self.R = r ** 2
        self.x = np.zeros([sys.A.shape[0], 1])


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


'''def run_pf(N, iters=0, pos=None, pos_err=None, init_err_std=None, sensor_std_err=None,
           mnt=None, map=None, do_plot=True, plot_particles=False):
    particles = create_gaussian_particles(mean=0, std=init_err_std, N=N)

    weights = np.zeros(N)
    weights.fill(1.)
    pos = pos
    ns_pos = pos + pos_err
    if plot_particles:
        alpha = .20
        if N > 5000:
            alpha *= np.sqrt(5000) / np.sqrt(N)
        plt.scatter(ns_pos - particles, np.zeros([N]),
                    alpha=alpha, color='g')
    plt.scatter(pos, 0, marker='+',
                color='k', s=180, lw=3)
    plt.scatter(ns_pos - np.mean(particles), 0, marker='s', color='r')
    xs = []

    for i in range(iters):
        pos += v * dt
        ns_pos = pos + pos_err

        zs = mnt(pos)

        predict(particles, std=0.1)

        # incorporate measurements
        update(particles, weights, z=zs, r=sensor_std_err, ns_pos=ns_pos, map=map)

        # resample if too few effective particles
        if neff(weights) < N / 2:
            indexes = systematic_resample(weights)
            resample_from_index(particles, weights, indexes)

        mu, var = estimate(particles, weights)
        xs.append(mu)
        if np.mod(i, 50) == 0:
            if plot_particles:
                plt.scatter(ns_pos - particles, np.ones([N]) * 0,
                            color='g', marker=',', s=5, alpha=0.01)
            p1 = plt.scatter(pos, 0, marker='+',
                             color='k', s=180, lw=3)
            p2 = plt.scatter(ns_pos - mu, 0, marker='s', color='r')

    # xs = np.array(xs)
    # plt.plot(xs[:, 0], xs[:, 1])
    plt.legend([p1, p2], ['Actual', 'PF'], loc=4, numpoints=1)
    print('final position error, std:\n\t', mu - pos_err, np.sqrt(var))
    plt.show()


def run_kf(iters=5000, pos=None, pos_err=None, init_err_std=None, sensor_std_err=1, mnt=None, map=None, do_plot=True):
    pos = pos
    ns_pos = pos + pos_err

    f = KalmanFilter(dim_x=1, dim_z=1)
    f.F = np.array([[1]])
    f.P = init_err_std ** 2
    f.Q = 0.01
    f.R = r ** 2
    f.x = np.array([[0]])
    x_est = np.empty([iters])
    x_est[0] = 0
    P_est = np.empty([iters])
    P_est[0] = init_err_std ** 2

    plt.scatter(pos, 0, marker='+', color='k', s=180, lw=3)
    plt.scatter(ns_pos - x_est[0], 0, marker='s', color='r')

    for i in range(1, iters):
        pos += v * dt
        ns_pos = pos + pos_err

        f.H = map(ns_pos - f.x)[2]
        # f.H = map(pos)[2]

        f.predict()
        mnt_v = map(ns_pos)[1] - mnt(pos)
        f.update(mnt_v)
        x_est[i] = f.x
        P_est[i] = f.P
        j = 500
        if np.mod(i, j) == 0:
            p1 = plt.scatter(pos, 0, marker='+', color='k', s=180, lw=3)
            p2 = plt.scatter(ns_pos - x_est[i], 0, marker='s', color='r')
            # xs = np.array(xs)  # plt.plot(xs[:, 0], xs[:, 1])
    plt.legend([p1, p2], ['Actual', 'KF'], loc=4, numpoints=1)
    plt.figure()
    plt.plot(x_est - pos_err)
    plt.figure()
    plt.plot(np.sqrt(P_est))
    print('final position error, std:\n\t', x_est[-1] - pos_err, np.sqrt(P_est[-1]))
    plt.show()
    return x_est, P_est
'''

# Параметры моделирования
seed(10)  #4 #7 #10
len = 10000  # длинна траектории [м]

dt = 1  # [с]
smpl = 2000
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
nav_path = np.linspace(start_pos, start_pos + mnt_num * V, mnt_num, endpoint=False)

# tau_ga = sg_ga / dgdt  # интервал корреляции поля, соответствующий h [c]
# alpha = 1 / tau_ga  # величина, обратная интервалу корреляции поля [1/с]
# q_w = np.sqrt(2 * sg_ga ** 2 * alpha)  # СКО порождающего шума в модели поля

InitModels(smpl, sg_ga=sg_ga, dgdl=dgdl, len=len, dt=dt)

mdl = mdls['M1']

# Подготовка поля и его измерений
map_v, map_interp = GenerateProfile('M1', smpl, dgdl=dgdl, len=len, sg_ga=sg_ga, dt=dt, offset=0)
print("СКО поля:", sg_ga, "мГал.",
      "СКО производной поля:", dgdl, "мГал / м.",
      "СКО ошибки измерений:", r, "мГал.")

err_n_buf, err_f_buf, err_s_buf = [], [], []
P_n_buf, P_f_buf, P_s_buf = [], [], []
tau_est_n_buf, tau_est_f_buf, tau_est_s_buf = [], [], []

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('dX')
ax.set_ylabel('Номер измерения')
ax.set_zlabel('Значение плотности')
ax.set_zlim(0, 0.5)

mse_num = 1
for mse in range(mse_num):
    print('Итерация', mse + 1)
    seed()
    mnt_v, mnt_interp = ModelMeasurements(map_v, smpl, dt, r)

    # Начальные условия для задачи навигации

    tau = sg_tau * randn()  # действительное значение tau

    print('Истинное значение tau', tau, "м.", 'Априорное СКО:', sg_tau, "м.")

    # Инициализация навигационных фильтров

    p_number = 5000
    pf_n = PF(p_number)
    pf_n.create_gaussian_particles(0, sg_tau)

    pf_f = PF(p_number)
    pf_f.create_gaussian_particles(0, sg_tau)

    pf_s = PF(p_number)
    pf_s.create_gaussian_particles(0, sg_tau)

    print('Интервал измерений в новом подходе', V * dt, "м")

    # инициализация фильтра предварительной обработки

    unsample_f = 100  # [м]
    unsample_s = 50  # [м]

    P0 = mdl.P0
    prf = Pre_Filter(mdl, P0, r)

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
    print('Установившийся режим для предварительного сглаживания', st_step_s, 'шага - ', st_step_s * V * dt, 'м')
    print('Установившееся СКО ошибки оценивания поля:', np.sqrt(P_s[st_step_s]), 'мГал')
    print('Интервал измерений для предварительной фильтрации', unsample_s, "м")

    '''
    for i in range(2000):
        pos += V * dt
        #ns_pos = pos + tau
        mnt = mnt_interp(pos)
    
        prf.predict()
        prf.update(mnt)
        est[i] = np.dot(mdls['Jordan'].C, prf.x)
    
    
    
    plt.figure()
    plt.plot(map_v[0, :], map_v[1, :])
    plt.plot(map_v[0, :], mnt_v, alpha=0.4)
    plt.plot(path, est, alpha=0.7)
    plt.legend(['Истинное значение', 'Исходные измерения', 'Оценка поля'])
    plt.gca().set_xlabel('[М]', fontsize=12)
    plt.gca().set_ylabel('[ед.]', fontsize=12)
    
    plt.draw()
    plt.pause(0.001)
    '''

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
        if np.mod(i, 20) == 0:
            data = np.array([pf_n.particles, pf_n.weights])
            data = data[:, np.argsort(data[0])]
            ax.plot(data[0], np.ones(p_number) * i, zs=data[1], zdir='z')

    err_n = np.abs(tau - np.array(pf_n.mean))
    err_f = np.abs(tau - np.array(pf_f.mean))
    err_s = np.abs(tau - np.array(pf_s.mean))

    err_n_buf.append(err_n)
    err_f_buf.append(err_f)
    err_s_buf.append(err_s)

    P_n_buf.append(pf_n.var)
    P_f_buf.append(pf_f.var)
    P_s_buf.append(pf_s.var)

    # tau_est_n_buf.append(pf_n.mean)
    # tau_est_f_buf.append(pf_f.mean)
    # tau_est_s_buf.append(pf_s.mean)

# расчет средней ошибки
err_n_buf = np.array(err_n_buf)
err_f_buf = np.array(err_f_buf)
err_s_buf = np.array(err_s_buf)

err_n_sum = np.sum(err_n_buf, axis=0) / mse_num
err_f_sum = np.sum(err_f_buf, axis=0) / mse_num
err_s_sum = np.sum(err_s_buf, axis=0) / mse_num

# расчет средней рассчетной дисперсии ошибки
P_n_buf = np.array(P_n_buf)
P_f_buf = np.array(P_f_buf)
P_s_buf = np.array(P_s_buf)

P_n_sum = np.sum(P_n_buf, axis=0) / mse_num
P_f_sum = np.sum(P_f_buf, axis=0) / mse_num
P_s_sum = np.sum(P_s_buf, axis=0) / mse_num

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

nav_path = np.arange(start_pos, start_pos + V * dt * mnt_cnt_n, V * dt)
nav_path_f = np.arange(start_pos, start_pos + unsample_f * mnt_cnt_f, unsample_f)
nav_path_s = np.arange(start_pos, start_pos + unsample_s * mnt_cnt_s, unsample_s)

plt.figure()
plt.plot(map_v[0, :], map_v[1, :])
plt.plot(map_v[0, :], mnt_v, alpha=0.4)
plt.plot(nav_path, mnt_f_interp(nav_path), alpha=0.7)
plt.plot(nav_path, mnt_s_interp(nav_path), alpha=0.7)
plt.legend(['Истинное значение', 'Исходные измерения', 'Фильтрация', 'Слгаживание'])
plt.gca().set_xlabel('[М]', fontsize=12)
plt.gca().set_ylabel('[ед.]', fontsize=12)

plt.figure()
plt.plot(nav_path, 3 * np.sqrt(P_f_interp(nav_path)))
plt.plot(nav_path, np.abs(pr_f_err))
plt.plot(nav_path, 3 * np.sqrt(P_s_interp(nav_path)))
plt.plot(nav_path, np.abs(pr_s_err))

plt.draw()
plt.pause(0.001)

plt.figure()
plt.plot(nav_path, 3 * np.sqrt(P_n_sum), color='C0')
plt.plot(nav_path_f, 3 * np.sqrt(P_f_sum), color='C1')
plt.plot(nav_path_s, 3 * np.sqrt(P_s_sum), color='C2')

plt.plot(nav_path, err_n_sum, linestyle='--', color='C0')
plt.plot(nav_path_f, err_f_sum, linestyle='--', color='C1')
plt.plot(nav_path_s, err_s_sum, linestyle='--', color='C2')

plt.legend(['3 СКО (без обработки)', '3 СКО (фильтрация)', '3 СКО (сглаживание)',
            'действительная ошибка (без обработки)',
            'действительная ошибка (фильтрация)',
            'действительная ошибка (сглаживание)'], fontsize='12')
plt.grid()
plt.gca().set_xlabel('[М]', fontsize=12)
plt.gca().set_ylabel('СКО [ед.]', fontsize=12)
plt.show()

# plt.figure()
# plt.plot(map_v[0, :], map_v[1, :])
# plt.plot(map_v[0, :], mnt_v, alpha=0.2)
#
# plt.figure()

# run_kf(iters=7000, pos=3000, pos_err=1000, init_err_std=500, map=map_interp, mnt=mnt_interp, sensor_std_err=r)
# # fld = generate_profile('M2', 10000, dgdl = 1/1000, len = 1000,sg_ga=10)
#
# run_pf(iters=7000, N=1000, pos=3000, pos_err=1000, init_err_std=500, map=map_interp, mnt=mnt_interp, sensor_std_err=r,
#        plot_particles=True)

'''plt.figure()
plt.plot(fld[0, :], fld[1, :])
plt.show()


mnt = model_measurements(fld[0, :], smpl, dt, r)
m = scipy.interpolate.interp1d(path, mnt, fill_value='extrapolate')
x_est, P_est = pre_filter(mnt.T, smpl, Fd, Gd, r)

pos_err = 2000


plt.figure()
plt.plot(path, mnt[0, :], alpha=0.7)
plt.plot(path, fld[0, :])

# plt.plot(path, x_est[0,:],alpha=0.8)
plt.grid()
# plt.figure()
# plt.plot(path, P_est[0,0,:])
plt.show()'''
