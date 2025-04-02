import param_calib
from calibration.utils import trace_refract_ray

def r0_rd(Y) :
    """Donne r0 et rd pour un point Y = [a, b] dans l'image"""
    return trace_refract_ray(Y, param_calib.air_K_l, param_calib.D_l, param_calib.n_l, param_calib.THICKNESS, param_calib.ETA_GLASS, param_calib.ETA_WATER)

