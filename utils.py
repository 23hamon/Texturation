import numpy as np
from calibration.utils import trace_refract_ray
import data.param_calib as param_calib

def r0_rd(Y, R=np.eye(3), t=np.zeros((3,))) :
    """Donne r0 et rd pour un point Y = [a, b] dans l'image"""
    t.reshape((3,))
    r0_cam, rd_cam = trace_refract_ray(Y, param_calib.air_K_l, param_calib.D_l, param_calib.n_l, param_calib.THICKNESS, param_calib.ETA_GLASS, param_calib.ETA_WATER)
    r0 = R.T @ r0_cam - R.T @ t
    rd = R.T @ rd_cam
    return r0, rd/np.linalg.norm(rd)


def distance_X_to_D_r0_rd(X, r0, rd):
    """
    Calcule d(X,D) ou X(x,y,z) est un point du mesh 3D, et D = {r0 + t * rd | d in IR} rayon lumineux parametre par r0 et rd
    X, r0 et rd sont des tableaux np de 3 floats
    """
    x,y,z = X[0], X[1], X[2]
    r0x, r0y, r0z = r0[0], r0[1], r0[2]
    rdx, rdy, rdz = rd[0], rd[1], rd[2]
    
    return np.array([
        rdz*(y-r0y) - rdy*(z-r0z),
        rdx*(z-r0z) - rdz*(x-r0x),
        rdy*(x-r0x) - rdx*(y-r0y)]).flatten()

if __name__ == "__main__" :
    print(param_calib.air_K_l, param_calib.D_l, param_calib.n_l, param_calib.THICKNESS, param_calib.ETA_GLASS, param_calib.ETA_WATER)

def closest_point_to_two_lines(ro1, rd1, ro2, rd2):
    b = ro2 - ro1

    d1_cross_d2 = np.cross(rd1, rd2)
    cross_norm2 = d1_cross_d2[0] * d1_cross_d2[0] + d1_cross_d2[1] * d1_cross_d2[1] + d1_cross_d2[2] * d1_cross_d2[2]

    t1 = np.linalg.det(np.array([
        [b[0], rd2[0], d1_cross_d2[0]],
        [b[1], rd2[1], d1_cross_d2[1]],
        [b[2], rd2[2], d1_cross_d2[2]]
    ])) / np.maximum(0.00001, cross_norm2)

    t2 = np.linalg.det(np.array([
        [b[0], rd1[0], d1_cross_d2[0]],
        [b[1], rd1[1], d1_cross_d2[1]],
        [b[2], rd1[2], d1_cross_d2[2]]
    ])) / np.maximum(0.00001, cross_norm2)

    p1 = ro1 + t1 * rd1
    p2 = ro2 + t2 * rd2

    return (p1 + p2) / 2.0, np.linalg.norm(p2 - p1)
