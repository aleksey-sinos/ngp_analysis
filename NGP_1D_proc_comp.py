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



def discm(A,B,dt,N,mode):
    n = np.size(A,0)
    r = np.size(B,1)
    Adt = A*dt
    F = np.eye(n)
    for i in range(N,0,-1):
       F = np.eye(n)+(Adt/i).dot(F)
    if r:
       Gm = np.eye(n)
       for i in range(N,0,-1):
           Gm = np.eye(n)+Adt.dot(Gm)/(i+1)
       if mode==0:
          G = dot(Gm,B)*np.sqrt(dt)
       else:
          G = Gm

    else:
       G=np.array([])

    return F,G

def generate_field(smpl, Fd, Gd):
    x = np.empty([2,smpl])
    x[:,0] = np.dot(Gd, np.random.randn()).ravel()
    for i in range(1,smpl):
        n = np.random.randn(Gd.shape[1])
        x[:, i] = np.dot(Fd, x[:, i-1]) + np.dot(Gd, n)
    return x

def model_measurements(fld,smpl,dt,R):
    q = R  # корень из интенсивности шума измерений [мГал*с^-1]
    q_d = q/np.sqrt(dt)
    err = q_d*np.random.randn(1,smpl)
    return fld+err

def pre_filter(mnt, smpl, Fd, Gd, R):
    f = KalmanFilter(dim_x=2, dim_z=1)
    f.x = np.array([0., 0.])
    f.F = Fd
    f.H = np.array([[1.,0.]])
    f.P = np.array([[400., 0.],
                    [0., 25.]])
    f.Q = dot(Gd,Gd.T)
    f.R = R**2
    x_est = np.empty([2,smpl])
    P_est = np.empty([2,2,smpl])
    for i in range(1,smpl+1):
        f.predict()
        f.update(mnt[i-1])
        x_est[:,i-1] = f.x
        P_est[:,:,i-1] = f.P
    return x_est, P_est

def create_uniform_particles(x_range, y_range, hdg_range, N):
    particles = np.empty((N, 3))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = uniform(hdg_range[0], hdg_range[1], size=N)
    particles[:, 2] %= 2 * np.pi
    return particles

def create_gaussian_particles(mean, std, N):
    particles = np.empty((N, 3))
    particles[:, 0] = mean[0] + (randn(N) * std[0])
    particles[:, 1] = mean[1] + (randn(N) * std[1])
    particles[:, 2] = mean[2] + (randn(N) * std[2])
    particles[:, 2] %= 2 * np.pi
    return particles



def predict(particles, u, std, dt=1.):
    """ move according to control input u (heading change, velocity)
    with noise Q (std heading change, std velocity)`"""

    N = len(particles)
    # update heading
    particles[:, 2] += u[0] + (randn(N) * std[0])
    particles[:, 2] %= 2 * np.pi

    # move in the (noisy) commanded direction
    dist = (u[1] * dt) + (randn(N) * std[1])
    particles[:, 0] += np.cos(particles[:, 2]) * dist
    particles[:, 1] += np.sin(particles[:, 2]) * dist

def update(particles, weights, z, R, landmarks):
    weights.fill(1.)
    for i, landmark in enumerate(landmarks):
        distance = np.linalg.norm(particles[:, 0:2] - landmark, axis=1)
        weights *= scipy.stats.norm(distance, R).pdf(z[i])

    weights += 1.e-300      # avoid round-off to zero
    weights /= sum(weights) # normalize

def estimate(particles, weights):
    """returns mean and variance of the weighted particles"""

    pos = particles[:, 0:2]
    mean = np.average(pos, weights=weights, axis=0)
    var  = np.average((pos - mean)**2, weights=weights, axis=0)
    return mean, var

def neff(weights):
    return 1. / np.sum(np.square(weights))

def resample_from_index(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights[:] = weights[indexes]
    weights.fill (1.0 / len(weights))

def run_pf1(N, iters=18, sensor_std_err=.1,
            do_plot=True, plot_particles=False,
            xlim=(0, 20), ylim=(0, 20),
            initial_x=None):
    landmarks = np.array([[-1, 2], [5, 10], [12, 14], [18, 21]])
    NL = len(landmarks)

    plt.figure()

    # create particles and weights
    if initial_x is not None:
        particles = create_gaussian_particles(
            mean=initial_x, std=(5, 5, np.pi / 4), N=N)
    else:
        particles = create_uniform_particles((0, 20), (0, 20), (0, 6.28), N)
    weights = np.zeros(N)

    if plot_particles:
        alpha = .20
        if N > 5000:
            alpha *= np.sqrt(5000) / np.sqrt(N)
        plt.scatter(particles[:, 0], particles[:, 1],
                    alpha=alpha, color='g')

    xs = []
    robot_pos = np.array([0., 0.])
    for x in range(iters):
        robot_pos += (1, 1)

        # distance from robot to each landmark
        zs = (norm(landmarks - robot_pos, axis=1) +
              (randn(NL) * sensor_std_err))

        # move diagonally forward to (x+1, x+1)
        predict(particles, u=(0.00, 1.414), std=(.2, .05))

        # incorporate measurements
        update(particles, weights, z=zs, R=sensor_std_err,
               landmarks=landmarks)

        # resample if too few effective particles
        if neff(weights) < N / 2:
            indexes = systematic_resample(weights)
            resample_from_index(particles, weights, indexes)

        mu, var = estimate(particles, weights)
        xs.append(mu)

        if plot_particles:
            plt.scatter(particles[:, 0], particles[:, 1],
                        color='k', marker=',', s=1)
        p1 = plt.scatter(robot_pos[0], robot_pos[1], marker='+',
                         color='k', s=180, lw=3)
        p2 = plt.scatter(mu[0], mu[1], marker='s', color='r')

    xs = np.array(xs)
    # plt.plot(xs[:, 0], xs[:, 1])
    plt.legend([p1, p2], ['Actual', 'PF'], loc=4, numpoints=1)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    print('final position error, variance:\n\t', mu - np.array([iters, iters]), var)
    plt.show()


from numpy.random import seed

seed(2)
#run_pf1(N=5000, plot_particles=True)
ln = 50000
v = 5
dt = 1
smpl = int(ln/v/dt)
#np.random.seed(seed=None)

sg = 20 #???
dgdl = 5 * v/1000 # градиент   поля   в   мГал / км   переведенная   через    скорость    в    мГал / с.
alfa = (dgdl ** 2 / (sg ** 2 * 2)) ** (0.5)
beta = dgdl/sg
F = np.array([[0,1],[-(alfa**2+beta**2),-2*alfa]])
G = np.array([[0],[np.sqrt(4*alfa*(sg**2)*(alfa**2+beta**2))]])
Fd, Gd = discm(F,G,dt,20,0)
R = 10
x = generate_field(smpl, Fd, Gd)
y = model_measurements(x[0,:],smpl,dt, R)
x_est, P_est = pre_filter(y.T, smpl, Fd, Gd, R)
path = np.linspace(0, ln/1000, smpl, endpoint=False)
plt.figure()
plt.plot(path, y[0,:],alpha=0.7)
plt.plot(path, x[0,:])
plt.plot(path, x_est[0,:],alpha=0.8)
plt.grid()
plt.figure()
plt.plot(path, P_est[0,0,:])
plt.show()

