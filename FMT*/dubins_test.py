import numpy as np
from dubins import *
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool
from functools import partial

q0 = np.array([86.335,124,0])/160
q1 = np.array([140,140,0])/160
pts, _ = get_pts(q0,q1,0.25,0.006978)
# print(pts*160)
plt.figure()
plt.plot(pts[:,1]*160,pts[:,0]*160)
plt.axis('equal')
plt.show()
