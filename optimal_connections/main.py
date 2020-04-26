import math
import numpy as np
import json
import matplotlib.pyplot as plt
import heapq

from sim_model import state_update

# Vehicle Model Params
class CarModel(object):
    def __init__(self, params):
        self.m = params['m']
        self.lf = params['lf']
        self.lr = params['lr']
        self.Iz = params['Iz']
        self.Df = params['Df']
        self.Cf = params['Cf']
        self.Bf = params['Bf']
        self.Dr = params['Dr']
        self.Cr = params['Cr']
        self.Br = params['Br']
        self.Cr2 = params['Cr2']
        self.Cr0 = params['Cr0']
        self.Cm2 = params['Cm2']
        self.Cm1 = params['Cm1']

with open("model.json") as f:
    params = json.load(f)
    car = CarModel(params)
# Dynamics
# (xnew, unew, Tnew) = Steer(znearest, z)
# state: [px, py, vx, vy, psi, w]
# psi = steer angle
# phi = heading direction
# input: [Fx, dpsi]

# def cost(xg, x, )

def dist_to_goal(x, xg):
    x = np.ndarray.flatten(x)
    xg = np.ndarray.flatten(xg)
    return (x[0]-xg[0])**2 + (x[1]-xg[1])**2

NUM_STATES = 6
x0 = np.zeros((NUM_STATES,))
xg = np.array([[5, 5, 0, 0, 0, math.pi/2, 0]]).T

d_range = np.arange(0.1, 1, 0.1)
u_range = np.arange(-math.pi/4, math.pi/4, 0.1)

alpha_d = 0.5
alpha_t = 0.5
scale = 0.1
N = 10
dt = 0.01
eps = 1e-3
time_axis = np.arange(dt, N, dt)
num_samples = int(N/dt)
K = 10
min_iters = 10
best_K_paths = []
for u in u_range:
    for d in d_range:
        x = x0
        total_cost = scale * dist_to_goal(x, xg)
        prev_dist = 0
        for iter in range(num_samples):
            new_x = state_update(x, [d, u], car, dt)
            # print("d: %.3f, steer: %.3f" % (d, u))
            # print(x)
            # print(new_x)
            # input()
            x = new_x
            new_dist = dist_to_goal(x, xg)
            if iter > 0:
                ddist = (prev_dist - new_dist) / dt
                # print(ddist)
                if ddist < eps: break

            prev_dist = new_dist
            total_cost += scale * new_dist
            
        # average scaled total cost of path
        if iter > min_iters:
            # print(iter, total_cost / iter)
            heapq.heappush(best_K_paths, (total_cost / iter, d, u, iter))
        if len(best_K_paths) > K:
            heapq._heappop_max(best_K_paths)
        
heapq.heapify(best_K_paths)
while len(best_K_paths) > 0:
    (_, d, u, max_iters) = heapq.heappop(best_K_paths)
    x_traj = np.zeros((NUM_STATES, max_iters))
    x = x0
    for i in range(max_iters):
        x = state_update(x, [d, u], car, dt)
        x_traj[:, i] = x

    # plot results y vs x
    plt.plot(x_traj[0,:], x_traj[1,:], 'r')
    plt.show()

    # plot results vs time
    # fig, [[ax1, ax2, ax3], [ax4, ax5, ax6]] = plt.subplots(2, 3)
    # time_axis = np.arange(0, max_iters)
    # ax1.plot(time_axis, x_traj[0,:])
    # ax1.set_title('X v.s time')
    # ax2.plot(time_axis, x_traj[1,:])
    # ax2.set_title('Y v.s time')
    # ax3.plot(time_axis, x_traj[4,:])
    # ax3.set_title('Psi v.s time')
    # ax4.plot(time_axis, x_traj[2,:])
    # ax4.set_title('Vx v.s time')
    # ax5.plot(time_axis, x_traj[3,:])
    # ax5.set_title('Vy v.s time')
    # ax6.plot(time_axis, x_traj[5,:])
    # ax6.set_title('omega v.s time')
    # plt.show()
        
