import numpy as np 
import control
from scipy.spatial.transform import Rotation
from simple_pid import PID
from collections import namedtuple
import json

import matplotlib.pyplot as plt

# import python_optimal_splines.OptimalSplineGen as OptimalSplineGen
from inverseDyn import inverse_dyn
from sim_model import state_update

"""
Failed attempt at feedback linearization for car control with LQR.
"""

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

# Global Constants
with open("model.json") as f:
    model = json.load(f)

with open("model.json") as f:
    params = json.load(f)
    car = CarModel(params)

# Flat Dynamics
FLAT_STATES = 6
FLAT_CTRLS = 3
A = np.array([
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0]
])
B = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 0],
    [0, 0, 1]
])
Q = np.eye(FLAT_STATES)
R = np.eye(FLAT_CTRLS) * 0.1

# Controls Calculation
dt = 0.01
# N = len(pts) // dt
T = 10  # seconds
N = int((T+dt) // dt)  # num samples

# PID Controller for setting torques
piddelta = PID(Kp=6, Ki=0, Kd=2)

# Optimal control law for flat system
K, S, E = control.lqr(A, B, Q, R)

x0 = np.ones((FLAT_STATES, )) * 0.1
x_traj = np.zeros((FLAT_STATES, N))
# get flightgoggles' odometry to get new state
# velocity given in body frame, need to change to world frame
x_traj[:, 0] = x0
xref = np.array([[3, 5, 0, 0, 0, 0]]).T
delta = 0
for i in range(1,N):
    t = i*dt
    x = np.reshape(x_traj[:,i-1], newshape=(FLAT_STATES, 1))
    u = -K*(x-xref)

    [duty_cycle, delta_des] = inverse_dyn(x, u, car, delta)

    ddelta = piddelta(delta - delta_des)
    delta += ddelta

    x = state_update(x, [duty_cycle, delta], car, dt)

    x_traj[:,i] = np.reshape(x, newshape=(FLAT_STATES,))

    # target_rot = Rotation.from_euler(seq="ZYX", angles=[psid, thetad, phid])

    # Pid to estimate torque moments

# Plot results
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
time_axis = np.arange(0, T, dt)
ax1.plot(time_axis, x_traj[0,:])
ax1.set_title('X v.s time')
ax2.plot(time_axis, x_traj[1,:])
ax2.set_title('Y v.s time')
ax3.plot(time_axis, x_traj[2,:])
ax3.set_title('Vx v.s time')
ax4.plot(time_axis, x_traj[3,:])
ax4.set_title('Vy v.s time')
plt.show()