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
    if type == 'linear':
        len = kwargs['len']
        dgdl = kwargs['dgdl']
        fld = np.empty([3, smpl])
        fld[0, :] = np.linspace(0, len, smpl, endpoint=False)
        fld[1, :] = np.linspace(0, len * dgdl, smpl, endpoint=False)
        fld[2, :] = np.ones(smpl) * dgdl

    if type == 'M1':
        pass
    if type == 'M2':
        len = kwargs['len']
        dgdl = kwargs['dgdl']
        sg = kwargs['sg']
        dl = len / smpl  # количество метров на один отсчет [м/отсчет]
        dgds = dgdl * dl  # градиент поля в [мГал/м] переведенный через [м/отсчет]  в [мГал/отсчет]

        '''alfa = (dgds ** 2 / (sg ** 2 * 2)) ** 0.5
        beta = dgds / sg'''

        alfa = 0  # (dgds ** 2 / (sg ** 2 * 2)) ** 0.5
        beta = dgds / sg

        F = np.array([[0, 1], [-(alfa ** 2 + beta ** 2), -2 * alfa]])
        G = np.array([[0], [np.sqrt(4 * alfa * (sg ** 2) * (alfa ** 2 + beta ** 2))]])
        Fd, Gd = discm(F, G, dt, 20, 0)
        fld = np.empty([3, smpl])
        fld[0, :] = np.linspace(0, len, smpl, endpoint=False)
        fld[1:3, 0] = np.dot(Gd, np.random.randn()).ravel()
        for i in range(1, smpl):
            n = np.random.randn(Gd.shape[1])
            fld[1:3, i] = np.dot(Fd, fld[1:3, i - 1]) + np.dot(Gd, n)
    if type == 'GRDN':
        pass
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


def model_measurements(fld, smpl, dt, R):
    q = R  # корень из интенсивности шума измерений [мГал*с^-1]
    q_d = q / np.sqrt(dt)
    err = q_d * np.random.randn(smpl)
    mnt = fld[1, :] + err
    m = scipy.interpolate.interp1d(fld[0, :], mnt, fill_value='extrapolate')
    return mnt, m


def pre_filter(mnt, smpl, Fd, Gd, R):
    f = KalmanFilter(dim_x=2, dim_z=1)
    f.fld = np.array([0., 0.])
    f.F = Fd
    f.H = np.array([[1., 0.]])
    f.P = np.array([[400., 0.],
                    [0., 25.]])
    f.Q = dot(Gd, Gd.T)
    f.R = R ** 2
    x_est = np.empty([2, smpl])
    P_est = np.empty([2, 2, smpl])
    for i in range(1, smpl + 1):
        f.predict()
        f.update(mnt[i - 1])
        x_est[:, i - 1] = f.fld
        P_est[:, :, i - 1] = f.P
    return x_est, P_est


'''def create_uniform_particles(x_range, y_range, hdg_range, N):
    particles = np.empty((N, 3))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = uniform(hdg_range[0], hdg_range[1], size=N)
    particles[:, 2] %= 2 * np.pi
    return particles'''


def create_gaussian_particles(mean, std, N):
    # particles = np.empty((N, 1))
    particles = mean + (randn(N) * std)
    return particles


def predict(particles, std):
    N = len(particles)
    # update heading
    particles += randn(N) * std


def update(particles, weights, z, R, ns_pos, map):
    # distance = np.linalg.norm(particles[:, 0:2] - landmark, axis=1)
    map_value = map(ns_pos - particles)[1]
    weights *= scipy.stats.norm(map_value, R).pdf(z.ravel())

    weights += 1.e-300  # avoid round-off to zero
    weights /= sum(weights)  # normalize


def estimate(particles, weights):
    """returns mean and variance of the weighted particles"""

    ns_err = particles
    mean = np.average(ns_err, weights=weights, axis=0)
    var = np.average((ns_err - mean) ** 2, weights=weights, axis=0)
    return mean, var


def neff(weights):
    return 1. / np.sum(np.square(weights))


def resample_from_index(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights[:] = weights[indexes]
    weights.fill(1.0 / len(weights))


def run_pf(N, iters=5000, pos=None, pos_err=None, init_err_std=None, sensor_std_err=None,
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
        update(particles, weights, z=zs, R=sensor_std_err, ns_pos=ns_pos, map=map)

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
    f.R = R ** 2
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
seed(2)

ln = 50000
v = 5
dt = 1
smpl = int(ln / v / dt)
# np.random.seed(seed=None)


R = 10  # СКО измерений!
plt.figure()
map_v, map_interp = generate_profile('M2', smpl, dgdl=5 / 1000, len=ln, sg=30)
# map_v, map_interp = generate_profile('linear', smpl, dgdl=-5 / 1000, len=ln)
# map_v, map_interp = generate_profile('sin', smpl, dgdl=-5 / 1000, len=ln)
plt.plot(map_v[0, :], map_v[2, :])

mnt_v, mnt_interp = model_measurements(map_v, smpl, dt, R)
plt.figure()
plt.plot(map_v[0, :], map_v[1, :])
plt.plot(map_v[0, :], mnt_v, alpha=0.2)
run_kf(iters=7000, pos=3000, pos_err=1000, init_err_std=500, map=map_interp, mnt=mnt_interp, sensor_std_err=R)
# fld = generate_profile('M2', 10000, dgdl = 1/1000, len = 1000,sg=10)

run_pf(iters=7000, N=1000, pos=3000, pos_err=1000, init_err_std=500, map=map_interp, mnt=mnt_interp, sensor_std_err=R,
       plot_particles=True)

'''plt.figure()
plt.plot(fld[0, :], fld[1, :])
plt.show()


mnt = model_measurements(fld[0, :], smpl, dt, R)
m = scipy.interpolate.interp1d(path, mnt, fill_value='extrapolate')
x_est, P_est = pre_filter(mnt.T, smpl, Fd, Gd, R)

pos_err = 2000


plt.figure()
plt.plot(path, mnt[0, :], alpha=0.7)
plt.plot(path, fld[0, :])

# plt.plot(path, x_est[0,:],alpha=0.8)
plt.grid()
# plt.figure()
# plt.plot(path, P_est[0,0,:])
plt.show()'''
