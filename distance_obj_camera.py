import cv2
from backprojection import back_projeter
import numpy as np
from data import param_calib
import trimesh
from utils import closest_point_to_two_lines
import os
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

mesh = trimesh.load_mesh('fichiers_intermediaires/mesh_visible_low.ply')


# -- fct auxiliaires --

def dist_cam_obj(X, view_number, R_cam, t_cam, R_cam_l__r, t_cam_l_r):

    image_path_l = f"downsampled/scene_l_00{str(view_number)}.jpeg"
    image_path_r = f"downsampled/scene_r_00{str(view_number)}.jpeg"
    image_l = cv2.imread(image_path_l)[..., ::-1]
    image_r = cv2.imread(image_path_r)[..., ::-1]

    if not os.path.exists(image_path_l):
        print(f"Erreur: Le fichier {image_path_l} est manquant.")
        return None
    if not os.path.exists(image_path_r):
        print(f"Erreur: Le fichier {image_path_r} est manquant.")
        return None
    
    height_l, width_l, _ = image_l.shape
    height_r, width_r, _ = image_r.shape

    #on trouve la matrice de rotation de la caméra droite par rapport à la caméra gauche
    R_cam_r = R_cam @ R_cam_l__r 
    t_cam_r = R_cam @ t_cam_l_r + t_cam #a voir si ca marche !!!!


    _, ro_l, rd_l = back_projeter( X, height_l, width_l, R_cam, t_cam)
    _, ro_r, rd_r = back_projeter( X, height_r, width_r, R_cam_r, t_cam_r)


    #on trouve les coordonnées du point
    p, _ = closest_point_to_two_lines(ro_l, rd_l, ro_r, rd_r)
    distance_l = np.linalg.norm(p - ro_l)
    distance_r = np.linalg.norm(p - ro_r)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(ro_l[0], ro_l[1], ro_l[2], rd_l[0], rd_l[1], rd_l[2], length=1, color='r', label="Rayon Caméra Gauche")
    ax.quiver(ro_r[0], ro_r[1], ro_r[2], rd_r[0], rd_r[1], rd_r[2], length=1, color='b', label="Rayon Caméra Droite")
    ax.scatter(p[0], p[1], p[2], color='g', s=100, label='Point d\'Intersection')
    ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2], triangles=mesh.faces, color='gray', alpha=0.5)
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


    return p, distance_l, distance_r


# --- main ---

view_number = 26
R_cam = np.eye(3)
t_cam = np.zeros((3,))
R_cam_l__r = param_calib.RotationDroiteGauche
t_cam_l_r = param_calib.TranslationDroiteGauche

mesh = trimesh.load_mesh('fichiers_intermediaires/mesh_visible_low.ply')
vertices = mesh.vertices

print(dist_cam_obj(vertices[0], view_number, R_cam, t_cam, R_cam_l__r, t_cam_l_r))







