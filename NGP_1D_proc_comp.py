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


'''def generate_field(smpl, Fd, Gd):
    fld = np.empty([2, smpl])
    fld[:, 0] = np.dot(Gd, np.random.randn()).ravel()
    for i in range(1, smpl):
        n = np.random.randn(Gd.shape[1])
        fld[:, i] = np.dot(Fd, fld[:, i - 1]) + np.dot(Gd, n)
    fld[0,:] = np.linspace(0, 250, smpl, endpoint=False)
    fld[1,:] = np.zeros(smpl)
    return fld'''


def generate_profile(type, smpl, **kwargs):
    if type == 'linear':
        len = kwargs['len']
        dgdl = kwargs['dgdl']
        fld = np.empty([2, smpl])
        fld[0, :] = np.linspace(0, len, smpl, endpoint=False)
        fld[1, :] = np.linspace(0, len * dgdl, smpl, endpoint=False)
    if type == 'M1':
        pass
    if type == 'M2':
        len = kwargs['len']
        dgdl = kwargs['dgdl']
        sg = kwargs['sg']
        dl = len / smpl
        dgds = dgdl * dl  # градиент   поля   в   мГал / м   переведенный   через    псевдоскорость    в    мГал / отсчет.
        alfa = (dgds ** 2 / (sg ** 2 * 2)) ** 0.5
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
    return fld


def get_map_value(dist):
    return f(dist)


def get_mnt_value(dist):
    return m(dist)


def model_measurements(fld, smpl, dt, R):
    q = R  # корень из интенсивности шума измерений [мГал*с^-1]
    q_d = q / np.sqrt(dt)
    err = q_d * np.random.randn(1, smpl)
    return fld + err


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


def update(particles, weights, z, R, ns_pos):
    # distance = np.linalg.norm(particles[:, 0:2] - landmark, axis=1)
    map_value = get_map_value(ns_pos - particles)
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


def run_pf1(N, iters=9000, sensor_std_err=1,
            do_plot=True, plot_particles=False,
            xlim=(0, 20), ylim=(0, 20),
            initial_x=None):
    plt.figure()
    plt.plot(path, fld[0, :])
    plt.plot(path, mnt[0, :], alpha=0.5)

    # create particles and weights
    if initial_x is not None:
        particles = create_gaussian_particles(
            mean=0, std=1000, N=N)

    weights = np.zeros(N)
    weights.fill(1.)
    pos = np.array([initial_x])
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
    # robot_pos = np.array([0., 0.])

    for i in range(iters):
        pos += v * dt
        ns_pos = pos + pos_err
        # distance from robot to each landmark
        zs = get_mnt_value(pos)

        predict(particles, std=0.1)

        # incorporate measurements
        update(particles, weights, z=zs, R=sensor_std_err, ns_pos=ns_pos)

        # resample if too few effective particles
        if neff(weights) < N / 2:
            indexes = systematic_resample(weights)
            resample_from_index(particles, weights, indexes)

        mu, var = estimate(particles, weights)
        xs.append(mu)
        if np.mod(i, 50) == 0:
            if plot_particles:
                plt.scatter(ns_pos - particles, np.ones([N]) * np.mod(i + 1, 1000) / 50,
                            color='g', marker=',', s=5, alpha=0.01)
            p1 = plt.scatter(pos, np.mod(i + 1, 1000) / 50, marker='+',
                             color='k', s=180, lw=3)
            p2 = plt.scatter(ns_pos - mu, np.mod(i + 1, 1000) / 50, marker='s', color='r')

    xs = np.array(xs)
    # plt.plot(xs[:, 0], xs[:, 1])
    plt.legend([p1, p2], ['Actual', 'PF'], loc=4, numpoints=1)
    print('final position error, std:\n\t', mu - pos_err, np.sqrt(var))
    plt.show()


from numpy.random import seed

seed(2)

ln = 50000
v = 5
dt = 1
smpl = int(ln / v / dt)
# np.random.seed(seed=None)


R = 100
fld = generate_profile('linear', 10000, dgdl=-1 / 1000, len=10000)
# fld = generate_profile('M2', 10000, dgdl = 1/1000, len = 1000,sg=10)
plt.figure()
plt.plot(fld[0, :], fld[1, :])
plt.show()

'''
path = np.linspace(0, ln, smpl, endpoint=False)
f = scipy.interpolate.interp1d(path, fld[0, :], fill_value='extrapolate')
mnt = model_measurements(fld[0, :], smpl, dt, R)
m = scipy.interpolate.interp1d(path, mnt, fill_value='extrapolate')
x_est, P_est = pre_filter(mnt.T, smpl, Fd, Gd, R)

pos_err = 2000

run_pf1(N=1000, plot_particles=True, initial_x=1000, sensor_std_err=R)
plt.figure()
plt.plot(path, mnt[0, :], alpha=0.7)
plt.plot(path, fld[0, :])

# plt.plot(path, x_est[0,:],alpha=0.8)
plt.grid()
# plt.figure()
# plt.plot(path, P_est[0,0,:])
plt.show()'''
