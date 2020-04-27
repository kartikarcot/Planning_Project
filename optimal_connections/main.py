import math
import numpy as np
import json
import matplotlib.pyplot as plt
import heapq

from sim_model import state_update

TWO_PI = 2 * math.pi

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
    # B, C, D = [10, 1.9, 1]
    # car.m = 1500 * 1000  # 1500kg
    # car.Bf = car.Br = B
    # car.Cf = car.Cr = C
    # car.Df = car.Dr = D
# Dynamics
# (xnew, unew, Tnew) = Steer(znearest, z)
# state: [px, py, vx, vy, psi, w]
# psi = steer angle
# phi = heading direction
# input: [Fx, dpsi]

# def cost(xg, x, )
print('hello')

def ang_dist(psi1, psi2):
    diff = psi1 - psi2
    # account for angle wraparound 0 to 2pi
    dist = min(abs(diff), TWO_PI - abs(diff))
    return dist

def dist_to_goal(x, xg, ang_scale = 1):
    # [px, py, vx, vy, psi, w] = x
    x = np.ndarray.flatten(x)
    xg = np.ndarray.flatten(xg)
    # print("xdist: %.3f, ydist: %.3f, angdist: %.3f" % (
    #     (x[0]-xg[0])**2, (x[1]-xg[1])**2, ang_scale*ang_dist(x[4], xg[4])
    # ))
    return (x[0]-xg[0])**2 + (x[1]-xg[1])**2 # + ang_scale*ang_dist(x[4], xg[4])

NUM_STATES = 6
x0 = np.zeros((NUM_STATES,))
xg = np.array([[3, 10, 0, 0, math.pi/4, 0]]).T

d_range = np.arange(0.1, 1, 0.1)
u_range = np.arange(-math.pi/4, math.pi/4, 0.1)

alpha_d = 0.5
alpha_t = 0.5
scale = 0.1
N = 10
dt = 0.01
eps = 1e-6
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
            # print(x)
            # print(new_x)
            x = new_x
            new_dist = dist_to_goal(x, xg)
            if iter > 0:
                ddist = (prev_dist - new_dist) / dt
                # print("d: %.3f, steer: %.3f, ddist: %.3f" % (d, u, ddist))
                # input()
                
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
        
print('hello')
heapq.heapify(best_K_paths)
while len(best_K_paths) > 0:
    (_, d, u, max_iters) = heapq.heappop(best_K_paths)
    x_traj = np.zeros((NUM_STATES, max_iters))
    rear_xtraj = np.zeros((2, max_iters))
    front_xtraj = np.zeros((2, max_iters))
    x = x0
    for i in range(max_iters):
        x = state_update(x, [d, u], car, dt)
        [cx, cy, vx, vy, psi, w] = x
        # psi += math.pi/2
        print(psi)
        # rear tire
        rear_xtraj[0, i] = cx - car.lr * math.cos(psi)
        rear_xtraj[1, i] = cy - car.lr * math.sin(psi)
        
        # front tire
        front_xtraj[0, i] = cx + car.lf * math.cos(psi)
        front_xtraj[1, i] = cy + car.lf * math.sin(psi)

        # CoM pose
        x_traj[:, i] = x

    # plot results y vs x
    # plt.plot(x_traj[0,:], x_traj[1,:], 'r')
    # plt.plot(front_xtraj[0,:], front_xtraj[1,:], 'g')
    # plt.plot(rear_xtraj[0,:], rear_xtraj[1,:], 'b')
    # plt.title("Position Y v.s X")
    # plt.legend(["center", "front", "rear"])
    # plt.show()

    # plot each car position as a line segment connecting rear to front
    for i in range(max_iters):
        plt.plot(
            [rear_xtraj[0,i], x_traj[0,i], front_xtraj[0,i]], 
            [rear_xtraj[1,i], x_traj[1,i], front_xtraj[1,i]])
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
        
