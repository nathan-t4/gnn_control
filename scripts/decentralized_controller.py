import numpy as np
import matplotlib.pyplot as plt
from control import dlqr

"""
    Dynamic programming solutions for decentralized state-feedback LQG problems with communication delays
"""

""" System parameters """
m = 1500
dt = 0.1
T = 600
ts = int(T / dt)
C = 450
alpha = 50
A_11 = np.array([[1, dt], [0, 1]])
A_ii = np.array([[1, dt], [alpha * dt / m, 1]]) # i = 2,3,4,5
A_ij = np.array([[0, 0], [-alpha * dt / m, 0]]) # i = 2,3,4,5; j = i-1
B_ii = np.array([[0, 0], [0, dt / m]]) # i = 1,2,3,4,5

zeros = np.zeros((2,2))
A = np.block([[A_11, zeros, zeros, zeros, zeros], 
            [A_ij, A_ii, zeros, zeros, zeros],
            [zeros, A_ij, A_ii, zeros, zeros],
            [zeros, zeros, A_ij, A_ii, zeros],
            [zeros, zeros, zeros, A_ij, A_ii]])
B = np.block([[B_ii, zeros, zeros, zeros, zeros],
            [zeros, B_ii, zeros, zeros, zeros],
            [zeros, zeros, B_ii, zeros, zeros],
            [zeros, zeros, zeros, B_ii, zeros],
            [zeros, zeros, zeros, zeros, B_ii]])
Q = np.eye(10)
R = 0.1 * np.eye(10)

# assume p0 = [1,2,3,4,5] * 10000 and v0 = [0, 0, 0, 0, 0]
x0 = np.array([1, 0, 2, 0, 3, 0, 4, 0, 5, 0])

def platoon_dynamics():
    K, S, E = dlqr(A, B, Q, R) # Solve discrete-time LQR

    def dynamics_part_a(x):
        """ Simulate closed-loop dynamics for part a"""
        D = np.random.normal(loc=0, scale=10000, size=5)
        w = np.zeros((10,))
        w[::2] = dt * D / m
        return (A - B @ K) @ x + w # u = -K x 
    
    xs = [x0] 
    [xs.append(dynamics_part_a(xs[t-1])) for t in range(1, ts)]
    xs = np.array(xs).T
    times = np.linspace(0, T, ts)
    F = -K @ xs # this is \tilde{F}

    ps = xs[[0,2,4,6,8]]
    fig, (ax1, ax2) = plt.subplots(2)
    labels = [rf'$\tilde p^{i}$' for i in range(1, 6)]
    ax1.plot(times, ps.T, label=labels)
    ax1.set_xlabel('$time$ [$s$]')
    ax1.set_ylabel(r'$\tilde{p}$')
    ax1.legend()

    Fs = F[[1,3,5,7,9]]
    labels = [rf'$\tilde F^{i}$' for i in range(1, 6)]
    ax2.plot(times, Fs.T, label=labels)
    ax2.set_xlabel('$time$ [$s$]')
    ax2.set_ylabel(r'Control input $\tilde{F}^i$ [$N$]')
    ax2.legend()

    plt.show()

    sim_cost = 0
    for x, u in zip(xs.T, F.T):
        sim_cost += (1 / ts) * (x.T @ Q @ x + u.T @ R @ u)
    print('Simulated cost:', sim_cost)

    min_cost = 5 * np.trace(10000 * S)
    print('Minimum cost:', min_cost)

def platoon_comm_delay():
    K, S, E = dlqr(A, B, Q, R) # Solve discrete-time LQR
    """ Part (b.ii) code """
    def propogate_K(v, r, X_v):
        """ 
        Propogate solution X_v through information hierarchy graph:

        Calculate K_r, X_v given that nodes v -> r in the information hierarchy graph and X_v. 
        """
        v = np.array(v) - 1 # e.g. v = [3,4,5] -> v = [2,3,4]
        r = np.array(r) - 1
        v_transform = []
        for v_i in v:
            # e.g. v = [0,2] -> v_transform = [0,1,4,5]
            v_transform.append(2 * v_i)
            v_transform.append(2 * v_i + 1)
        r_transform = []
        for r_i in r:
            r_transform.append(2 * r_i)
            r_transform.append(2 * r_i + 1)

        print(v, v_transform, r, r_transform)
        
        A_vr = A[np.ix_(v_transform,r_transform)]
        B_vr = B[np.ix_(v_transform,r_transform)]
        R_rr = R[np.ix_(r_transform,r_transform)]
        Q_rr = Q[np.ix_(r_transform,r_transform)]

        X_r = Q_rr + A_vr.T @ X_v @ A_vr - A_vr.T @ X_v @ B_vr @ np.linalg.inv(R_rr + B_vr.T @ X_v @ B_vr) @ B_vr.T @ X_v @ A_vr
        K_r = np.linalg.inv(R_rr + B_vr.T @ X_v @ B_vr) @ B_vr.T @ X_v @ A_vr

        return K_r, X_r

    """ Calculate feedback gain matrices K_s """
    K_1 = propogate_K([1,2], [1], propogate_K([1,2,3],[1,2], propogate_K([1,2,3,4],[1,2,3], propogate_K([1,2,3,4,5],[1,2,3,4], S)[1])[1])[1])[0]

    K_2 = propogate_K([1,2,3], [2], propogate_K([1,2,3,4],[1,2,3], propogate_K([1,2,3,4,5], [1,2,3,4], S)[1])[1])[0]

    K_3 = propogate_K([2,3,4], [3], propogate_K([1,2,3,4,5], [2,3,4], S)[1])[0]

    K_4 = propogate_K([3,4,5], [4], propogate_K([2,3,4,5], [3,4,5], propogate_K([1,2,3,4,5], [2,3,4,5], S)[1])[1])[0]

    K_5 = propogate_K([4,5], [5], propogate_K([3,4,5], [4,5], propogate_K([2,3,4,5], [3,4,5], propogate_K([1,2,3,4,5], [2,3,4,5], S)[1])[1])[1])[0]

    Ks = [K_1, K_2, K_3, K_4, K_5]

    for i in range(5):
        print(f'K_{i}:', Ks[i])

    """ Part (b.iii) code """
    def dynamics_part_b(x):
        """ Simulate closed-loop dynamics of delayed system """
        D = np.random.normal(loc=0, scale=10000, size=10)
        u = np.zeros(10)
        for i in range(5):
            # Calculate u(t)
            xi = np.random.normal(loc=0, scale=10000)
            idx = [2 * i, 2 * i + 1]
            I_Vs = np.eye(10)[np.ix_(np.arange(10), idx)]
            K = Ks[i].sum(axis=1).reshape(2,1)
            u -= xi * (I_Vs @ K).squeeze()
        return (A @ x + B @ u + D), u

    # xi0 = 0
    u0 = np.zeros(10)

    xs = [x0] 
    us = [u0]
    for t in range(1, ts):
        xt, ut = dynamics_part_b(xs[t-1])
        xs.append(xt)
        us.append(ut)

    xs = np.array(xs).T
    us = np.array(us).T
    times = np.linspace(0, T, ts)

    ps = xs[[0,2,4,6,8]]
    fig, (ax1, ax2) = plt.subplots(2)
    labels = [rf'$\tilde p^{i}$' for i in range(1, 6)]
    ax1.plot(times, ps.T, label=labels)
    ax1.set_xlabel('$time$ [$s$]')
    ax1.set_ylabel(r'$\tilde{p}$')
    # ax1.set_ylim([-2.5e6, 2.5e6])
    ax1.legend()

    Fs = us[[1,3,5,7,9]]
    labels = [rf'$\tilde F^{i}$' for i in range(1, 6)]
    ax2.plot(times, Fs.T, label=labels)
    ax2.set_xlabel('$time$ [$s$]')
    ax2.set_ylabel(r'Control input $\tilde{F}^i$ [$N$]')
    ax2.legend()

    plt.show()

    J_delay = 0
    D = np.random.normal(loc=0, scale=10000, size=10)
    for i in range(5):
        J_delay += np.trace(D[i] * Ks[i])
    print('Minimum cost J delay:', J_delay)

    sim_cost = 0
    for x, u in zip(xs.T, us.T):
        sim_cost += (1 / ts) * (x.T @ Q @ x + u.T @ R @ u)
    print('Simulated cost:', sim_cost)

if __name__ == '__main__':
    # data = platoon_dynamics()
    platoon_comm_delay()