import numpy as np

M = 268.73775590551185

f_l = M * 24.4936280811
cx_l, cy_l = 1510.08637472, 986.0160617
air_K_l = np.array([
    [f_l, 0, cx_l],
    [0, f_l, cy_l],
    [0, 0, 1]
])

f_r = M * 24.57431989
cx_r, cy_r = 1493.22488246, 996.53835531
air_K_r = np.array([
    [f_r, 0, cx_r],
    [0, f_r, cy_r],
    [0, 0, 1]
])

D_l=40.0
D_r=D_l


Rc = np.array([-0.65741284,  9.19567167, -3.20664961])
Tc =  np.array([-218.32875083,   -5.95499246,   20.59005131])

n_l =  np.array([ 0.01826953, -0.01171918,  0.99976441]) # (normalized)
n_r =  np.array([-0.02498697, -0.00409035,  0.99967941]) # (normalized)

THICKNESS = 13
ETA_GLASS = 1.492
ETA_WATER = 1.34