from backprojection import back_projeter
from utils import get_image_data, sqrt_newton
from main_func import compute_edges

import numpy as np
from numba import njit

import cv2
import open3d as o3d
from tqdm import tqdm
from collections import defaultdict

import time
import gc
from multiprocessing import Pool
from functools import partial
import ctypes

def invert_dict(d):
    new_d = defaultdict(list)
    for (p, q), jk_array in d.items():
        for jk in jk_array:
            key = tuple(jk)  # (j, k)
            new_d[key].append((p, q))
    return dict(new_d)


@njit(error_model="numpy", cache=True)
def trapezoidal_integration(x, y):
    """
    Calcule l'integrale de y par rapport a x en utilisant la methode des trapezes.
    x et y doivent être des tableaux de meme longueur, avec x trie dans l'ordre croissant.
    Hypothese : y > 0 partout (fonction strictement positive)
    """
    if len(x) != len(y):
        raise ValueError("x et y doivent avoir la meme longueur")
    n = len(x)
    integral = 0.0
    for i in range(n - 1):
        h = x[i + 1] - x[i]
        area = h * (y[i] + y[i + 1]) / 2
        integral += area
    return integral

@njit(error_model="numpy", cache=True)
def _distance_RGB(xRGB, yRGB) :
    """
    Calcule la distance euclidienne dans l'espace RGB 
    """
    return sqrt_newton(((xRGB-yRGB)**2).sum())


# /! POUR PLUS TARD
# plutot que de faire int(Y) on peut interpoler de maniere plus intelligente
@njit(error_model="numpy", cache=True)
def _weight_seam(Vjyxc_cam, N,
                 Yj_linspace, Yk_linspace, 
                 j, k, 
                 N_integration):
    """
    Donne le poids d'une couture
    **Input :**
        - Vjyxc_cam : np array de shape (N, h, w, 6) ou le dernier channel est RGBRGB 
                      pour l'image gauche et l'image droite
        - j, k entre 0 et 2N-1. [0, N[ represente l'image gauche, [N, 2N[ l'image droite 
    """
    slice_for_RGB_j = slice(0, 3) if j < N else slice(3, 6)
    slice_for_RGB_k = slice(0, 3) if k < N else slice(3, 6)
    real_j = j if j < N else j-N
    real_k = k if k < N else k-N
    return np.array([_distance_RGB(Vjyxc_cam[real_j, int(Yj_linspace[i][1]), int(Yj_linspace[i][0]), slice_for_RGB_j], 
                                   Vjyxc_cam[real_k, int(Yk_linspace[i][1]), int(Yk_linspace[i][0]), slice_for_RGB_k])
                                   for i in range(N_integration)])


def _build_all_y(args, all_X, N_edges, N_integration): #, all_X, N_edges, N_integration) :
    """
    Retroprojecte sur une vue donnee l'ensemble des aretes visibles
    **Input : **
        - all_X : l'ensemble des aretes etalees : shape (N_edges, N_integration, 3))
        - view_to_is_edge_visible : shape (N_edge, 2)
                                    pour chaque edge, dit si elle est visible depuis la vue en question
    """
    j, rot_j, t_j, view_to_is_edge_visible_l, view_to_is_edge_visible_r = args
    all_Y_l = np.zeros((N_edges, N_integration, 2))
    all_Y_r = np.zeros((N_edges, N_integration, 2))
    for idx_edge in range(N_edges) :
        all_Y_l[idx_edge, :, :] = np.array([back_projeter(X, rot_j, t_j)[0] if view_to_is_edge_visible_l[idx_edge]
                                            else np.zeros((2,))
                                            for X in all_X[idx_edge]])
        all_Y_r[idx_edge, :, :] = np.array([back_projeter(X, rot_j, t_j)[0] if view_to_is_edge_visible_r[idx_edge]
                                            else np.zeros((2,))
                                            for X in all_X[idx_edge]])
    return j, (all_Y_l, all_Y_r)


def build_Wpqjk(N, Vjyxc_cam, 
                rot_images, t_images, 
                vertices, edges_set,
                Mpj_cam, N_integration=10) :
    """
    Calcule tout Wpqjk_cam (les valeurs non nulles et non inf) et le renvoie
    **Input :**
        - Vjyxc_cam : np array de shape (N, h, w, 6) ou le dernier channel est RGBRGB 
                      pour l'image gauche et l'image droite
        - Mpj_cam : np array de shape (K, N, 2), tenseur de visibilite
    **Output :**
        - Wpqjk_cam : dict : {(p, q) : { (j,k) : cost }} le tenseur de cout croise 
                      si j < N : image gauche, si N <= j < 2N : image droite
    """
    linspace_t = np.linspace(0, 1, N_integration)
    N_edges = len(edges_set)
    Wpq_X1X2 = dict() # associe une arete (p,q) au segment dans l'espace [X1, X2] : {(p, q) : np array de shape (N_integration, 3)}
    Wpq_len_X1X2 = dict() # la longueur du segment en question
    Wpqjk_cam = {key : dict() for key in edges_set.keys()}

    # -- ETAPE 1 -- Chargement des aretes et des images visibles
    print("-- ETAPE 1 -- Chargement des aretes et des images visibles")
    for p, q in tqdm(edges_set, total=N_edges):
        # aretes p < q
        v1, v2 = edges_set[(p,q)] 
        X1, X2 = vertices[v1], vertices[v2] # [X1, X2] est le segment de l'arete dans R^3
        X1_X2_linspace = np.array([X2*t + X1*(1-t) for t in linspace_t])
        Wpq_X1X2[(p, q)]  = X1_X2_linspace
        Wpq_len_X1X2[(p,q)] = np.linalg.norm(X1-X2)
    
    # -- ETAPE 2 -- Construction de view_to_is_edge_visible qui pour une vue donnee, 
    # contient l'information de la visibilite de chaque edge pour la cam gauche et droite
    print("-- ETAPE 2 -- Construction de view_to_is_edge_visible")
    view_to_is_edge_visible = np.full((2*N, N_edges), fill_value=False, dtype=bool)
    for j in tqdm(range(2*N), total=2*N) :
        real_j = j%N
        cam_j = j//N
        for idx, (p, q) in enumerate(edges_set) :
            if Mpj_cam[p, real_j, cam_j] or Mpj_cam[q, real_j, cam_j] : 
                view_to_is_edge_visible[j, idx] = True # l'une des deux faces est visible sur la vue j


    # -- ETAPE 3 -- Retroprojection sur chaque vue du tableau contenant toutes les aretes
    print("-- ETAPE 3 -- Retroprojection sur chaque vue du tableau contenant toutes les aretes")
    j_to_all_Y_l = dict()
    j_to_all_Y_r = dict()
    all_X = np.array([Wpq_X1X2[(p, q)] for _, (p, q) in enumerate(Wpq_X1X2)]) # (N_edges, N_integration, 3)
    # parcours parallele de l'ensemble des vues
    args = [(j, rot_images[j], t_images[j], view_to_is_edge_visible[j], view_to_is_edge_visible[j+N])
            for j in range(N)]
    _f = partial(_build_all_y,
                  all_X=all_X,
                  N_edges=N_edges,
                  N_integration=N_integration)
    with Pool(16) as p:
        for j, all_Y_lr in tqdm(p.imap_unordered(
            _f,
            args,
            chunksize=1
        ), total=N) :
            all_Y_l, all_Y_r = all_Y_lr
            j_to_all_Y_l[j] = all_Y_l
            j_to_all_Y_r[j] = all_Y_r
            
    # del all_X, args, Wpq_X1X2, view_to_is_edge_visible
    # gc.collect()
    # ctypes.CDLL("libc.so.6").malloc_trim(0)

    # -- ETAPE 4 -- Integration sur les aretes retro projetees
    print("ETAPE 4 -- Integration sur les aretes retro projetees")
    for idx, (p, q) in tqdm(enumerate(Wpqjk_cam), total=N_edges):
        for j in range(2*N) :
            for k in range(2*N) :
                #print(j, k)
                if j != k :
                    if ((j < N and Mpj_cam[p, j%N, 0]) or (j >= N and Mpj_cam[p, j%N, 1])) and \
                        ((k < N and Mpj_cam[q, k%N, 0]) or (k >= N and Mpj_cam[q, k%N, 1])) :
                        #print(j, k, "  ")
                        Y_pqj = j_to_all_Y_l[j][idx] if j < N else j_to_all_Y_r[j-N][idx] # (N_integration, 2) 
                        Y_pqk = j_to_all_Y_l[k][idx] if k < N else j_to_all_Y_r[k-N][idx]
                        y = _weight_seam(Vjyxc_cam, N, Y_pqj, Y_pqk, j, k, N_integration)
                        integr = trapezoidal_integration(linspace_t, y)
                        Wpqjk_cam[(p, q)][(j, k)] = Wpq_len_X1X2[(p,q)] * integr
    return Wpqjk_cam

    
    

if __name__ == "__main__" :

    # images
    N = 52
    h, w = 2000, 3000
    Vjyxc_cam = np.zeros((N, h, w, 6), dtype=int)
    for cam_idx, cam in enumerate(["l", "r"]):
        image_path = f"downsampled/scene_{cam}_"
        for j in range(1, N + 1):
            img = cv2.cvtColor(cv2.imread(image_path + f"{j:04d}.jpeg"), cv2.COLOR_BGR2RGB)
            Vjyxc_cam[j - 1, ..., cam_idx * 3:(cam_idx + 1) * 3] = img
    print(f"Vjyxc_cam charge. Shape : ({Vjyxc_cam.shape})")

    # rotations et translation des images
    rot_images = []
    t_images = []
    for j in range(N) :
        rot, t = get_image_data(j)
        rot_images.append(rot)
        t_images.append(t)
    rot_images = np.array(rot_images)
    t_images = np.array(t_images)
    print(f"Matrices de rotations chargées (shape : {rot_images.shape}), vecteurs de translation chargés : {t_images.shape}")

    # mesh
    mesh = o3d.io.read_triangle_mesh("ply/LOW_CLEAN_MESH.ply")
    # visibilite des faces
    Mpj_cam = np.load("tensors/Mpj_cam.npy")
    print(f"Mpj_cam ouvert : {Mpj_cam.shape}")

    # edges
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    K = len(triangles)
    edges_set = compute_edges(triangles)
    print(f"Nombre d'edges : {len(edges_set)}")

    Wpqjk_cam = build_Wpqjk(N, Vjyxc_cam, rot_images, t_images, vertices, edges_set, Mpj_cam, 10)
    np.save('tensors/Wpqjk_cam.npy', Wpqjk_cam, allow_pickle=True)

    print(Wpqjk_cam[(0,7)])

