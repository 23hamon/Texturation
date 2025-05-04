from backprojection import back_projeter
from utils import get_image_data, sqrt_newton
from main_func import compute_edges

import numpy as np
from numba import njit

import cv2
import open3d as o3d
from tqdm import tqdm
from collections import defaultdict

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


# /!\ POUR PLUS TARD
# plutot que de faire int(Y) on peut interpoler de maniere plus intelligente
@njit(error_model="numpy", cache=True)
def _weight_seam(Vjyxc, 
                 Yj_linspace, Yk_linspace, 
                 j, k, 
                 N_integration):
    """
    Donne le poids d'une couture
    """
    return np.array([_distance_RGB(Vjyxc[j, int(Yj_linspace[i][1]), int(Yj_linspace[i][0]), :], 
                                   Vjyxc[k, int(Yk_linspace[i][1]), int(Yk_linspace[i][0]), :])
                                   for i in range(N_integration)])


def build_Wpqjk(N, K, Vjyxc, cam, 
                rot_images, t_images, 
                vertices, edges_set,
                Mpj, N_integration=10) :
    """
    Calcule tout Wpqjk (les valeurs non nulles et non inf) et le renvoie
    """
    linspace_t = np.linspace(0, 1, N_integration)
    N_edges = len(edges_set)
    Wpq_X1X2 = dict()       # associe une arete (p,q) au segment dans l'espace [X1, X2] : {(p, q) : np array de shape (N_integration, 3)}
    Wpq_len_X1X2 = dict()
    Wpqjk_cost = {key : dict() for key in edges_set.keys()}

    # -- ETAPE 1 -- Chargement des aretes et des images visibles
    print("-- ETAPE 1 -- Chargement des aretes et des images visibles")
    for p, q in tqdm(edges_set, total=N_edges):
        # aretes p < q
        v1, v2 = edges_set[(p,q)] 
        X1, X2 = vertices[v1], vertices[v2] # [X1, X2] est le segment de l'arete dans R^3
        X1_X2_linspace = np.array([X2*t + X1*(1-t) for t in linspace_t])
        Wpq_X1X2[(p, q)]  = X1_X2_linspace
        Wpq_len_X1X2[(p,q)] = np.linalg.norm(X1-X2)
    
    # -- ETAPE 2 -- Retroprojection sur chaque vue du tableau contenant toutes les aretes
    # A FAIRE EN RENTRANT :
    # s'arranger pour qu'il ne backprojecte que les arretes visibles sur la vue (utiliser un mask numpy)
    # paralleliser le code pour calculer separement chaque vue
    print("-- ETAPE 2 -- Retroprojection sur chaque vue du tableau contenant toutes les aretes")
    j_to_all_Y = dict()
    all_X = np.array([Wpq_X1X2[(p, q)] for _, (p, q) in enumerate(Wpq_X1X2)]) # (N_edges, N_integration, 3)
    for j in tqdm(range(N)) :
        rot_j, t_j = rot_images[j], t_images[j]
        all_Y = np.zeros((N_edges, N_integration, 2))
        for idx_edge in range(N_edges) :
            all_Y[idx_edge, :, :] = np.array([back_projeter(X, rot_j, t_j)[0] for X in all_X[idx_edge]])
            j_to_all_Y[j] = all_Y

    # -- ETAPE 3 -- Integration sur les aretes retro projetees
    print("ETAPE 3 -- Integration sur les aretes retro projetees")
    for idx, (p, q) in tqdm(enumerate(Wpqjk_cost), total=N_edges):
        for j in range(N) :
            for k in range(N) :
                if j != k and Mpj[p, j] and Mpj[q, k] :
                    Y_pqj = j_to_all_Y[j][idx] # (N_integration, 2)
                    Y_pqk = j_to_all_Y[k][idx]
                    y = _weight_seam(Vjyxc, Y_pqj, Y_pqk, j, k, N_integration)
                    integr = trapezoidal_integration(linspace_t, y)
                    Wpqjk_cost[(p, q)][(j, k)] = Wpq_len_X1X2[(p,q)] * integr

    return Wpqjk_cost

    
    

if __name__ == "__main__" :

    # images
    cam = "l"
    N = 52
    image_path = f"downsampled/scene_{cam}_"
    Vjyxc = [
        cv2.cvtColor(cv2.imread(image_path + f"{j:04d}.jpeg"), cv2.COLOR_BGR2RGB)
        for j in range(1, N + 1)
    ]
    Vjyxc = np.stack(Vjyxc, axis=0) # shape  (N, h, w, 3) = (52, 2000, 3000, 3)
    h, w = Vjyxc[0].shape[:2]
    print(Vjyxc.shape)

    # rotations et translation des images
    rot_images = []
    t_images = []
    for j in range(N) :
        rot, t = get_image_data(j+1)
        rot_images.append(rot)
        t_images.append(t)
    rot_images = np.array(rot_images)
    t_images = np.array(t_images)
    print(f"Matrices de rotations chargées (shape : {rot_images.shape}), vecteurs de translation chargés : {t_images.shape}")

    # mesh
    mesh = o3d.io.read_triangle_mesh("ply/mesh_cailloux_luca_CLEAN.ply")

    # visibilite des faces
    Mpj = np.load("tensors/Mpj.npy")
    print(f"Mpj ouvert : {Mpj.shape}")

    # edges
    Fp = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    K = len(Fp)
    edges_set = compute_edges(Fp)
    print(f"Nombre d'edges : {len(edges_set)}")

    Wpqjk = build_Wpqjk(N, K, Vjyxc, "l", rot_images, t_images, vertices, edges_set, Mpj, 10)

    print(Wpqjk[(0,7)])
    np.save('tensors/Wpqjk.npy', Wpqjk, allow_pickle=True)

    
# def _cross_weight_seam(Vjyxc,                   # tenseur images
#                        X1_X2_linspace,          # echantillonnage de l'arete : [X1, X2]
#                        linspace_t,              # echantillonnage de [0,1]
#                        N_integration,           # pas d'echantillonnage
#                        j, k,                    # vues j et k
#                        rotj, tj, rotk, tk,
#                        cam="l"                  # type de camera
#                        ) : 
#     """
#     Donne le cout croise d'une arete entre les vues j et k. 
#     Ne fonctionnera que si l'arete est visible depuis les deux vues
#     """
#     Yj_linspace = np.array([back_projeter(X, rotj, tj, cam)[0] for X in X1_X2_linspace])
#     Yk_linspace = np.array([back_projeter(X, rotk, tk, cam)[0] for X in X1_X2_linspace])
#     y = _weight_seam(Vjyxc, Yj_linspace, Yk_linspace, j, k, N_integration)
#     integr = trapezoidal_integration(linspace_t, y)
#     return integr

# def build_Wpqjk(N, K, Vjyxc, cam, 
#                 rot_images, t_images, 
#                 vertices, edges_set,
#                 Mpj, 
#                 N_integration=100) :
#     """
#     Calcule tout Wpqjk (les valeurs non nulles et non inf) et le renvoie
#     """
#     linspace_t = np.linspace(0, 1, N_integration)
#     Wpqjk_dict = {key : dict() for key in edges_set}
#     len_edges_set = len(edges_set)
#     for p, q in tqdm(edges_set, total=len_edges_set):
#         # aretes ou p < q
#         v1, v2 = edges_set[(p,q)] 
#         X1, X2 = vertices[v1], vertices[v2] # [X1, X2] est le segment de l'arete dans R^3
#         X1_X2_linspace = np.array([X2*t + X1*(1-t) for t in linspace_t])
#         mask = Mpj[p] & Mpj[q]   # array de shape (N,), booleen : les vues communes a p et q
#         indices = np.where(mask)[0] 
#         j_idx, k_idx = np.meshgrid(indices, indices, indexing='ij')
#         pairs = np.stack([j_idx.ravel(), k_idx.ravel()], axis=1) 
#         # pairs (j, k) depuis lesquelles p et q sont visibles
#         N_pairs = len(pairs)
#         for j_k in tqdm(pairs, total=N_pairs) :
#             j, k = j_k[0], j_k[1]
#             if j != k :
#                 rot_j, t_j = rot_images[j], t_images[j]
#                 rot_k, t_k = rot_images[k], t_images[k]
#                 integr = _cross_weight_seam(Vjyxc, X1_X2_linspace, linspace_t, N_integration, j, k, rot_j, t_j, rot_k, t_k, cam)
#                 integr *= sqrt_newton(((X2-X1)**2).sum())
#                 Wpqjk_dict[(p, q)][(j, k)] = integr
#     return Wpqjk_dict