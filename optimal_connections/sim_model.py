import numpy as np
import math

# Derivations from https://github.com/alexliniger/MPCC
def state_update(x, u, car, dt):
    [px, py, vx, vy, psi, w] = x
    [duty_cycle, delta] = u

    # slip ratio, force calculations
    alpha_f = -math.atan2(w*car.lf + vy, vx) + delta
    alpha_r = math.atan2(w*car.lr - vy, vx)
    Ffy = car.Df*math.sin(car.Cf*math.atan(car.Bf*alpha_f))
    Fry = car.Dr*math.sin(car.Cr*math.atan(car.Br*alpha_r))
    Frx = (car.Cm1 - car.Cm2*vx)*duty_cycle #- car.Cr - car.Cr2*vx**2

    # generate state update
    xdot = vx*math.cos(psi) - vy*math.sin(psi)
    ydot = vx*math.sin(psi) + vy*math.cos(psi)
    vxdot = (Frx - Ffy*math.sin(delta) + car.m*vy*w) / car.m
    vydot = (Fry - Ffy*math.cos(delta) - car.m*vx*w) / car.m
    psidot = w
    wdot = (Ffy*car.lf*math.cos(delta) - Fry*car.lr) / car.Iz
    dx = np.array([xdot, ydot, vxdot, vydot, psidot, wdot])
    return x + dx*dt
