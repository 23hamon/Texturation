import numpy as np
from calibration.utils import trace_refract_ray
import data.param_calib as param_calib
import json
import cv2

def r0_rd(Y, R=np.eye(3), t=np.zeros((3,)), cam="l") :
    """
    Donne r0 et rd pour un point Y = [a, b] dans l'image
    /!\ -- R et t sont la matrice de rotation et le vecteur de translation 
    du repere monde vers le repere gauche
    """
    t.reshape((3,))
    if cam == "l" :
        r0_cam, rd_cam = trace_refract_ray(Y, param_calib.air_K_l, param_calib.D_l, param_calib.n_l, param_calib.THICKNESS, param_calib.ETA_GLASS, param_calib.ETA_WATER)
        # passage du repere gauche au repere monde
        r0 = R.T @ r0_cam - R.T @ t
        rd = R.T @ rd_cam
    elif cam == "r" :
        r0_cam, rd_cam = trace_refract_ray(Y, param_calib.air_K_r, param_calib.D_r, param_calib.n_r, param_calib.THICKNESS, param_calib.ETA_GLASS, param_calib.ETA_WATER)
        # passage du repere droit au repere monde
        R_DG = param_calib.RotationDroiteGauche
        t_DG = param_calib.TranslationDroiteGauche
        r0 = R.T @ (R_DG.T @ r0_cam - R_DG.T @ t_DG) - R.T @ t
        rd = R.T @ R_DG.T @ rd_cam
    else :
        raise ValueError(f"Incorrect value : {cam}. 'cam' must be 'l' or 'r'")
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

def get_image_data(image_id=26):
    """
    Renvoie (rot, t) ou rot est la matrice de rotation de la camera, et t son vecteur de translation
    """
    # position de l'image
    with open("data/absolute_transforms_full.json") as f :
        data = json.load(f)
        image_id=str(image_id)
        r = np.array(data["0"][image_id][0], dtype=np.float64)
        t = np.array(data["0"][image_id][1], dtype=np.float64)
    return (r, t)

if __name__ == "__main__" :
    print(param_calib.air_K_l, param_calib.D_l, param_calib.n_l, param_calib.THICKNESS, param_calib.ETA_GLASS, param_calib.ETA_WATER)

