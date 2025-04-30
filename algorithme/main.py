import open3d as o3d
import cv2
import numpy as np
import multiprocessing


from clean_mesh import clean_mesh
from visible_faces_from_view import build_Mpj
from utils import get_image_data
from single_weight import build_Wpj
from cross_weight import build_Wpqjk
from main_func import compute_edges
from alpha_expansion import alpha_expansion

# hyperparametres
COS_THETA_MAX = 0   # cosinus de l'angle maximal tolere entre une face et une camera
                    # dans ]-1, 0]
CAM = "l"           # temporaire, a terme fera sur les deux cam
FLOAT_INF = 1e20
N_INTEGRATION = 10  # nombre de points pour calculer l'integrale


# parametres 
N = 52
image_path = f"downsampled/scene_{CAM}_"



if __name__ == "__main__" :
    multiprocessing.set_start_method('spawn')

    # ouverture des transformations image
    rot_images = []
    t_images = []
    for j in range(N) :
        rot, t = get_image_data(j+1)
        rot_images.append(rot)
        t_images.append(t)
    rot_images = np.array(rot_images)
    t_images = np.array(t_images)

    # ETAPE 1 --- Ouverture et nettoyage du mesh, calcul de Mpj
    original_mesh = o3d.io.read_triangle_mesh("ply/mesh_cailloux_luca_high.ply")
    print("Construction de la matrice originale des vues")
    mesh = clean_mesh(original_mesh, rot_images, t_images, COS_THETA_MAX)  
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    triangle_normals = np.asarray(mesh.triangle_normals)
    K = len(triangles)
    print("Construction de la nouvelle matrice des vues")
    Mpj = build_Mpj(mesh, rot_images, t_images, CAM, COS_THETA_MAX)

    # ETAPE 2 --- Calcul du cout individuel
    print("Construction de la matrice du cout par face et par vue")
    Wpj = build_Wpj(K, N, triangles, vertices, triangle_normals, Mpj, rot_images, t_images, CAM, FLOAT_INF)

    # ETAPE 3 -- Calcul du cout croise
    Vjyxc = [cv2.cvtColor(cv2.imread(image_path + f"{j:04d}.jpeg"), cv2.COLOR_BGR2RGB)
             for j in range(1, N + 1)] # ouverture des images
    Vjyxc = np.stack(Vjyxc, axis=0) # shape  (N, h, w, 3) = (52, 2000, 3000, 3)
    h, w = Vjyxc[0].shape[:2]
    edges_set = compute_edges(triangles)
    print("Construction du tenseur de cout croise")
    Wpqjk = build_Wpqjk(N, K, Vjyxc, CAM, rot_images, t_images, vertices, edges_set, Mpj, N_INTEGRATION)

    print("Optimisation du cout par alpha-expansion")
    M_final = alpha_expansion(N, K, edges_set, Mpj, Wpj, Wpqjk, FLOAT_INF)
    print(M_final)
    np.save("tensors/M_final.npy", M_final)