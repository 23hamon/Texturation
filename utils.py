import numpy as np
from calibration.utils import trace_refract_ray
import data.param_calib as param_calib

def r0_rd(Y) :
    """Donne r0 et rd pour un point Y = [a, b] dans l'image"""
    return trace_refract_ray(Y, param_calib.air_K_l, param_calib.D_l, param_calib.n_l, param_calib.THICKNESS, param_calib.ETA_GLASS, param_calib.ETA_WATER)

def distance_X_to_D_r0_rd(X, r0, rd):
    """
    Calcule d(X,D) ou X(x,y,z) est un point du mesh 3D, et D = {r0 + t * rd | d in IR} rayon lumineux parametre par r0 et rd
    X, r0 et rd sont des tableaux np de 3 floats
    """
    x,y,z = X[0], X[1], X[2]
    r0x, r0y, r0z = r0[0], r0[1], r0[2]
    rdx, rdy, rdz = rd[0], rd[1], rd[2]
    return np.sqrt(
        (rdz*(y-r0y) - rdy*(z-r0z))**2 +
        (rdx*(z-r0z) - rdz*(x-r0x))**2 +
        (rdy*(x-r0x) - rdx*(y-r0y))**2
    )

if __name__ == "__main__" :
    print(param_calib.air_K_l, param_calib.D_l, param_calib.n_l, param_calib.THICKNESS, param_calib.ETA_GLASS, param_calib.ETA_WATER)
