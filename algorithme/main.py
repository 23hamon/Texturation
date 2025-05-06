import open3d as o3d
import cv2
import numpy as np

import ctypes
import multiprocessing
import gc

from calib_luca import n_l, n_r
from utils import get_image_data
from main_func import compute_edges
from FAST_build_MpjWpj import clean_and_build_Mpj_Wpj_cam
from FAST_build_Wpqjk import build_Wpqjk
from alpha_expansion import alpha_expansion, E

# hyperparametres
COS_THETA_MAX = -0.1   # cosinus de l'angle maximal tolere entre une face et une camera
                       # dans ]-1, 0]
FLOAT_INF = 1e20
N_INTEGRATION = 10  # nombre de points pour calculer l'integrale

# parametres 
N = 52
h, w = 2000, 3000

if __name__ == "__main__" :
    multiprocessing.set_start_method('forkserver')

    # ouverture des transformations image
    rot_images = []
    t_images = []
    for j in range(N) :
        rot, t = get_image_data(j+1)
        rot_images.append(rot)
        t_images.append(t)
    rot_images = np.array(rot_images)
    t_images = np.array(t_images)

    # ouverture des images
    Vjyxc_cam = np.zeros((N, h, w, 6), dtype=int)
    for cam_idx, cam in enumerate(["l", "r"]):
        image_path = f"downsampled/scene_{cam}_"
        for j in range(1, N + 1):
            img = cv2.cvtColor(cv2.imread(image_path + f"{j:04d}.jpeg"), cv2.COLOR_BGR2RGB)
            Vjyxc_cam[j - 1, ..., cam_idx * 3:(cam_idx + 1) * 3] = img
    print(f"Vjyxc_cam charge. Shape : ({Vjyxc_cam.shape})")

    # ETAPE 1 --- Ouverture et nettoyage du mesh, calcul de Mpj et Wpj
    original_mesh = o3d.io.read_triangle_mesh("ply/mesh_cailloux_luca_LOW.ply")
    print("Mesh ouvert. Calcul de Mpj_cam et Wpj_cam")
    mesh_clean, Mpj_cam, Wpj_cam = clean_and_build_Mpj_Wpj_cam(original_mesh, 
                                                               N, rot_images,
                                                               t_images, n_r, n_l, 
                                                               COS_THETA_MAX, FLOAT_INF)
    del original_mesh
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)

    np.save("tensors/Mpj_cam.npy", Mpj_cam)
    np.save("tensors/Wpj_cam.npy", Wpj_cam)
    o3d.io.write_triangle_mesh("ply/mesh_cailloux_luca_LOW_CLEAN.ply", mesh_clean)

    # ETAPE 2 -- Calcul du cout croise
    triangles = np.asarray(mesh_clean.triangles)
    vertices = np.asarray(mesh_clean.triangles)
    K = len(triangles)
    edges_set = compute_edges(triangles)
    print(f"Nombre d'edges : {len(edges_set)}")
    print("Construction du tenseur de cout croise Wpqjk_cam")
    Wpqjk_cam = build_Wpqjk(N, Vjyxc_cam, rot_images, t_images, vertices, edges_set, Mpj_cam, N_INTEGRATION)
    np.save('tensors/Wpqjk_cam.npy', Wpqjk_cam, allow_pickle=True)

    del Vjyxc_cam, vertices, triangles
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)


    # ETAPE 3 -- Alpha-expansion
    print("Optimisation du cout par alpha-expansion")
    M_final = alpha_expansion(N, K, edges_set, Mpj_cam, Wpj_cam, Wpqjk_cam, FLOAT_INF)
    print(M_final)
    print(f"E(M_final) = {E(M_final, K, edges_set, Mpj_cam, Wpj_cam, Wpqjk_cam, FLOAT_INF)}")
    np.save("tensors/M_final.npy", M_final)
