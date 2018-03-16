import numpy as np
from numpy import dot
from filterpy.monte_carlo import residual_resample, systematic_resample
from numpy.linalg import norm
from numpy.random import randn
import matplotlib.pyplot as plt
from numpy.random import uniform
import scipy.stats
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy import interpolate
from numpy.random import seed
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import seaborn as sns



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


def generate_profile(type, smpl, **kwargs):
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
        len = kwargs['len']
        dgdl = kwargs['dgdl']
        sg_ga = kwargs['sg_ga']
        dgdt = dgdl * V  # градиент поля[мГал / с]
        tau_ga = sg_ga / dgdt  # интервал корреляции поля, соответствующий h [c]
        alpha = 1 / tau_ga  # величина, обратная интервалу корреляции поля [1/с]
        q_w = np.sqrt(2 * sg_ga**2 * alpha)  # СКО порождающего шума в модели поля
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
        len = kwargs['len']
        dgdl = kwargs['dgdl']
        sg_ga = kwargs['sg_ga']
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
        x = np.zeros([3, smpl])
        fld = np.zeros([3, smpl])
        fld[0, :] = np.linspace(0, len, smpl, endpoint=False)
        x[1, 0] = np.dot(sg_ga, np.random.randn())
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

    f = scipy.interpolate.interp1d(fld[0, :], fld, fill_value='extrapolate')
    return fld, f


'''def get_map_value(dist):
    return f(dist)


def get_mnt_value(dist):
    return m(dist)'''


def model_measurements(fld, smpl, dt, r):

    # q_d = r / np.sqrt(dt)
    err = r * np.random.randn(smpl)
    mnt = fld[1, :] + err
    m = scipy.interpolate.interp1d(fld[0, :], mnt, fill_value='extrapolate')
    return mnt, m


class Pre_Filter:
    def __init__(self, Fd, Gd, sg_ga, r):
        self.f = KalmanFilter(dim_x=1, dim_z=1)
        self.f.F = Fd
        self.f.H = np.array([[1.]])
        self.f.P = np.array([[sg_ga**2]])
        self.f.Q = dot(Gd, Gd.T)
        self.f.R = r ** 2
        self.f.x = np.array([[0]])
        self.mean = np.array([0])
        self.var = np.array([sg_ga ** 2])
    def predict(self):
        self.f.predict()
    def update(self, mnt):
        self.f.update(mnt)
        self.mean = np.append(self.mean, self.f.x)
        self.var = np.append(self.var, self.f.P)




def pre_filter(mnt, smpl, Fd, Gd, r):
    f = KalmanFilter(dim_x=1, dim_z=1)
    f.fld = np.array([0., 0.])
    f.F = Fd
    f.H = np.array([[1., 0.]])
    f.P = np.array([[400., 0.],
                    [0., 25.]])
    f.Q = dot(Gd, Gd.T)
    f.R = r ** 2
    x_est = np.empty([2, smpl])
    P_est = np.empty([2, 2, smpl])
    for i in range(1, smpl + 1):
        f.predict()
        f.update(mnt[i - 1])
        x_est[:, i - 1] = f.fld
        P_est[:, :, i - 1] = f.P
    return x_est, P_est


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
        self.weights *= scipy.stats.norm(map_value, r).pdf(mnt.ravel())
        #self.weights += 1.e-300  # avoid round-off to zero
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


def run_pf(N, iters=0, pos=None, pos_err=None, init_err_std=None, sensor_std_err=None,
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

# Параметры
seed()
len = 10000  # длинна траектории [м]
dt = 1  # [с]
smpl = 10000
# dl = len / smpl  # пространственный интервал решения [м]
V = 5
r = 5  # корень из интенсивности шума измерений [мГал*с^-1]
sg_ga = 30
dgdl = 5 / 1000
dgdt = dgdl * V  # градиент поля[мГал / с]
tau_ga = sg_ga / dgdt  # интервал корреляции поля, соответствующий h [c]
alpha = 1 / tau_ga  # величина, обратная интервалу корреляции поля [1/с]
q_w = np.sqrt(2 * sg_ga ** 2 * alpha)  # СКО порождающего шума в модели поля

print('СКО поля:', sg_ga, 'мГал.', "Интервал корреляции поля:", tau_ga * V, "м.", "СКО ошибки измерений:", r, "мГал")

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_zlim(0, 0.1)

# Подготовка поля и его измерений
map_v, map_interp = generate_profile('Jordan', smpl, dgdl=dgdl, len=len, sg_ga=sg_ga, dt=dt)
# map_v, map_interp = generate_profile('Jordan', smpl, dgdl=dgdl, len=len, sg_ga=sg_ga, dt=dt)
mnt_v, mnt_interp = model_measurements(map_v, smpl, dt, r)


# map_v, map_interp = generate_profile('linear', smpl, dgdl=-5 / 1000, len=ln)
# map_v, map_interp = generate_profile('sin', smpl, dgdl=-5 / 1000, len=ln)
# plt.plot(map_v[0, :], map_v[1, :])
# plt.figure()
# plt.plot(map_v[0, :], map_v[2, :])
# plt.show()

# Начальные условия

sg_tau = 500  # СКО погрешности НС
pos = 3000
tau = sg_tau*randn()
p_number = 5000
print('Истинное значение tau', tau, "м.", 'Априорное СКО:', sg_tau, "м.")



# Инициализация фильтров
pf = PF(p_number)
pf.create_gaussian_particles(0, sg_tau)

pf2 = PF(p_number)
pf2.create_gaussian_particles(0, sg_tau)


# инициализация фильтра предварительной обработки


F = np.array([[-alpha]])
G = np.array([[q_w]])
Fd, Gd = discm(F, G, dt, 20, 0)
prf = Pre_Filter(Fd, Gd, sg_ga, r)

# рассчет СКО ошибки фильтрации

# рассчет интервала решения задачи



# Моделирование работы алгоритмов оценивания


mnt_num = 1000
st_state = False
unsample = 300  # [м]
start_pos = 0
for i in range(mnt_num):
    pos += V * dt
    ns_pos = pos + tau
    mnt = mnt_interp(pos)

    prf.predict()
    prf.update(mnt)

    if st_state is False and np.abs(prf.var[-1] - prf.var[-2]) < 1e-3:
        st_state = True
        start_pos = pos
        mnt_new = 0
        mnt_tr = 0
        print('Установившийся режим в традиционном подходе начиная с', i, 'шага - ', (start_pos), 'м')
        print('Установившееся СКО ошибки оценивания поля:', np.sqrt(prf.var[-1]), 'мГал')
        print('Интервал измерений в новом подходе', V * dt, "м")
        print('Интервал измерений в традиционном подходе', unsample, "м")
        print('Коэффициент корреляции помехи измерений', 4.842, "мГал - ", 4.842 / np.sqrt(prf.var[-1]) * 100, "%")

    if st_state == True:
        mnt_new += 1
        pf.update(mnt, r, ns_pos, map_interp)
        pf.estimate()
        if np.mod(pos - start_pos, unsample) == 0:
            mnt_tr += 1
            pf2.update(prf.mean[-1], np.sqrt(prf.var[-1]), ns_pos, map_interp)
            pf2.estimate()

    #pf.predict(1)
    # if pf.neff(pf.weights) < pf.N / 2:
    #     pf.resample_from_index()
    if np.mod(i, 20) == 0:
        data = np.array([pf.particles, pf.weights])
        data = data[:, np.argsort(data[0])]
        ax.plot(data[0], np.ones(p_number) * i, zs=data[1], zdir='z')

err_new = np.abs(tau - np.array(pf.mean))
err_tr = np.abs(tau - np.array(pf2.mean))

print("Использованная для навигации длина траектории", mnt_new * V * dt, "м")

print('Новый подход: Оценка tau', pf.mean[-1], 'СКО tau', np.sqrt(pf.var[-1]), 'по', mnt_new, 'измерениям. Ошибка:',
      err_new[-1], 'м')
print('Традиционный: Оценка tau', pf2.mean[-1], 'СКО tau', np.sqrt(pf2.var[-1]), 'по', mnt_tr, 'измерениям. Ошибка:',
      err_tr[-1], 'м')

nav_path = np.arange(start_pos, start_pos + V * dt * mnt_new, V * dt)
nav_path_tr = np.arange(start_pos, start_pos + unsample * mnt_tr, unsample)

plt.figure()
plt.plot(map_v[0, :], map_v[1, :])
plt.plot(map_v[0, :], mnt_v, alpha=0.4)
plt.plot(nav_path, prf.mean[mnt_num - mnt_new + 1:], alpha=0.7)
plt.legend(['Истинное значение', 'Исходные измерения', 'Оценка поля'])
plt.gca().set_xlabel('[М]', fontsize=12)
plt.gca().set_ylabel('[ед.]', fontsize=12)

plt.draw()
plt.pause(0.001)

plt.figure()
plt.plot(nav_path, 3 * np.sqrt(pf.var), linestyle=":")
plt.plot(nav_path, err_new, linestyle="-.")
plt.plot(nav_path_tr, 3 * np.sqrt(pf2.var), linestyle="-")
plt.plot(nav_path_tr, err_tr, linestyle="--")
plt.legend(['3 СКО (новый)', 'действительная ошибка (новый)', '3 СКО (старый)',
            'действительная ошибка (старый)'], fontsize='12')
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
