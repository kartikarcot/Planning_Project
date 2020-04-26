import math
import numpy as np

def inverse_dyn(x, u, car, prev_delta):
    [px, py, vx, vy, psi, w] = np.ndarray.flatten(x)
    vxdot = float(u[0][0])
    vydot = float(u[1][0])
    wdot = float(u[2][0])
    alpha_f = -math.atan2(w*car.lf + vy, vx) + prev_delta
    alpha_r = math.atan2(w*car.lr - vy, vx)
    Ffy = car.Df*math.sin(car.Cf*math.atan(car.Bf*alpha_f))
    Fry = car.Dr*math.sin(car.Cr*math.atan(car.Br*alpha_r))

    delta_des = math.acos(
        (vydot*car.m - Fry + car.m*vx*w) / -Fry
    )
    Frx = vxdot*car.m + Ffy*math.sin(delta_des) - car.m*vy*w
    d_des = (Frx + car.Cr + car.Cd*vx**2) / (car.Cm1 - car.Cm2*vx)

    return [delta_des, d_des]