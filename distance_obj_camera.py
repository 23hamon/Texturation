import cv2
from backprojection import back_projeter
import numpy as np
from data import param_calib
import trimesh
from utils import closest_point_to_two_lines
import os
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from utils import r0_rd

#récupérer la taille d'une image 
image_height, image_width, _ = cv2.imread("downsampled/scene_l_0026.jpeg").shape
print(image_height, image_width)


# -- fct auxiliaires --

def dist_obj_cam(X, R_cam, t_cam, R_cam_l_r, t_cam_l_r, image_height=2000, image_width=3000):
    '''Le R-cam eet t_cam de chaque vue a été calculé en amout'''

    ro_l, rd_l = r0_rd(X, R_cam, t_cam, cam="l")
    ro_r, rd_r = r0_rd(X, R_cam_l_r, t_cam_l_r, cam="r")

    p, _ = closest_point_to_two_lines(ro_l, rd_l, ro_r, rd_r)
    distance_l = ((p - ro_l) ** 2).sum() ** 0.5
    distance_r = ((p - ro_r) ** 2).sum() ** 0.5

    return p, distance_l, distance_r


# --- main ---

#pour l'instant, on ne tente que sur des distances pour des photos qui sont situées à l'origine. 
#On ne prendra pas de vues différentes, afin de ne pas avoir de mauvais calculs dus à la rotation de la caméra qu'on a mal (??) calculée
# view_number = 26

# R_cam = np.eye(3)
# t_cam = np.zeros((3,))

# R_cam_l_r = param_calib.RotationDroiteGauche
# t_cam_l_r = param_calib.TranslationDroiteGauche

# mesh_test = trimesh.load_mesh('fichiers_intermediaires/mesh_visible_low.ply')
# vertices = mesh_test.vertices
# print(dist_obj_cam(vertices[0], R_cam, t_cam, R_cam_l_r, t_cam_l_r))







