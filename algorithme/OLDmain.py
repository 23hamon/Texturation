#import open3d as o3d
import numpy as np
import cv2
import networkx as nx
import open3d as o3d

from main_func import compute_edges
from utils import get_image_data

from tqdm import tqdm

# -- NOMS DE VARIABLES --
# 
# - IMAGES
# N :           int, nombre de vues
# Vjyxc :       tenseur image de shape (N, h, w, 3). Contient l'intensite du channel par pixel par vue
# j, k :        int, indices des vues
# Y :           float np array de shape (2,), contient les coordonnees [x, y] d'un pixel 
#
# - Mesh
# K :           int, Nombre de faces
# p, q :        int, indices des pixels
# v :           int, indices de vertices
# X :           float np array de shape(3,), contient les coordonnees [x, y, z] dans R^3 d'un point
# edges_set :   dict au format {(p, q) : (v1, v2)} qui associe l'edge entre les faces p et q aux deux vertices 
#               v1 et v2 qui le forment. Seules les aretes ou p < q sont presentes
# 
# - MESH ET IMAGES
# Mpj :         bool np array de shape (K, N). M[p,j] vaut 1 si la face p est visible sur la vue j et 0 sinon
# Wpj :         float np array de shape (K, N). W[p,j] vaut le cout de la vue j sur la face p (np.inf si ladite
#               face n'est pas visible sur ladite vue) 
# Wpqjk :       dict au format {(p, q) : {(j, k) : float}}. W[(p,q)][(j,k)] donne le cout de l'arete (p,q) si
#               p est coloriee avec la vue j et q avec la vue k. Comme Wpqjk = Wqpkj, seules les aretes ou p < q
#               sont presentes
# full_Wpqjk :  fonction. L'objet mathematique precedent mais pour toutes valeurs de p, q, j, k.
#               full_Wpqjk(p, q, j, k) \in [0, +\infty]
#
# - ALGORITHME
# M :           int np array de shape (K,). M[p] appartient a {0, ..., N-1} et donne l'indice de la vue qui 
#               colore le pixel p
# alpha :       int, la vue qu'on cherche a etendre par un alpha-move




def full_Wpqjk(p, q, j, k,
               Mpj, Wpqjk,
               float_inf=1e9) :
    """
    Renvoie le cout croise entre deux faces pour deux vues
    Si p et q sont visibles depuis les vues j et k, alors Wpqjk est fini
    Sinon Wpqjk vaut np.inf
    Si j = k, Wpqjk = 0
    Wpqjk = Wqpkj
    """
    if Mpj[p, j] and Mpj[p, k] and Mpj[q, j] and Mpj[q, k] :
        if j == k :
            return 0.
        elif p < q :
            return Wpqjk[(p, q)][(j, k)]
        else :
            return Wpqjk[(q, p)][(k, j)]
    else :
        return float_inf
           

def E_Q(M, 
        K, Wpj) :
    return sum([Wpj[p, M[p]] for p in range(K)])

def E_S(M,
        edges_set, Mpj, Wpqjk) :
    return sum([full_Wpqjk(p, q, M[p], M[q],
                           Mpj, Wpqjk)
                           for _, (p, q) in enumerate(edges_set)])

def E(M,
      K, edges_set, Mpj, Wpj, Wpqjk,
      llambda=1) :
    return E_Q(M, K, Wpj) + llambda * E_S(M, edges_set, Mpj, Wpqjk)

# Optimisation de l'energie avec alpha-expansion



# Trouver M_hat qui minimise l'energie dans les voisins a une alpha-expansion

def _generate_G_alpha(M, alpha, 
                      K, edges_set, Wpqjk, Mpj, Wpj, 
                      float_inf=1e9) :
    """
    Renvoie G_alpha, le graph dont la coupe minimale donne le nouveau coloriage
    - M est le vecteur d'attribution des faces (np array d'entiers entre 1 et N et de taille K)
    - alpha est une face, un entier entre 1 et N
    """
    G_alpha = nx.Graph()

    # Noeuds -- 
    # alpha, alpha_bar, et les pixels de l'image (faces triangulaires)
    G_alpha.add_node("alpha")
    G_alpha.add_node("alpha_bar")
    for p in range(K) :
        G_alpha.add_node(p)
    # noeuds speciaux : a{p, q} pour un pixel p et un pixel q voisins tels que mp != mq
    for _, (p, q) in enumerate(edges_set) :
        if M[p] != M[q] :
            G_alpha.add_node(f"a({p},{q})")

    # Edges --
    # t-links : connecter chaque pixel a alpha et alpha_bar
    for p in range(K) :
        weight_tp_alpha = Wpj[p, alpha]
        weight_tp_alpha_bar = float_inf if M[p] == alpha else Wpj[p, M[p]]
        G_alpha.add_edge(p, "alpha", type="t-link", weight=weight_tp_alpha, weight_name=f"wpj[{p},alpha]") # t{p, alpha}
        G_alpha.add_edge(p, "alpha_bar", type="t-link", weight=weight_tp_alpha_bar, weight_name=f"{'inf' if weight_tp_alpha_bar == float_inf else f'wpj[{p}, M[{p}]]'}")
    # n-links : connecter les pixels voisins 
    for idx, (p, q) in enumerate(edges_set) :
        if M[p] == M[q] :
            weight_e_p_q = full_Wpqjk(p, q, M[p], alpha, Mpj, Wpqjk)
            G_alpha.add_edge(p, q, type="n-link", weight=weight_e_p_q, weight_name=f"wijkl[{p}, {q}, M[{p}], alpha]") # e{p,q}
        else :
            weight_e_p_a = full_Wpqjk(p, q, M[p], alpha, Mpj, Wpqjk)
            weight_e_a_q = full_Wpqjk(p, q, alpha, M[q], Mpj, Wpqjk)
            weight_t_a_alpha = full_Wpqjk(p, q, M[p], M[q], Mpj, Wpqjk)
            a = f"a({p},{q})"
            G_alpha.add_edge(p, a, type="n-link", weight=weight_e_p_a, weight_name=f"wijkl[{p}, {q}, M[{p}], alpha]") # e{p,a}
            G_alpha.add_edge(a, q, type="n-link", weight=weight_e_a_q, weight_name=f"wijkl[{p}, {q}, alpha, M[{q}]]") # e{a,q}
            G_alpha.add_edge(a, "alpha_bar", type="t-link", weight=weight_t_a_alpha, weight_name=f"wijkl[{p}, {q}, M[{p}], M[{q}]]") # t{a,alpha}
    # Poids des aretes
    return G_alpha

def _get_M_from_cut(M, partition, alpha,
                    K) :
    """
    Renvoie le vecteur de label M corresponant a la cut du graphe alpha
    """
    M_c = np.zeros((K,), dtype=int)
    for p in range(K) :
        if p in partition[0] :   # si t(alpha, p) est dans C
            M_c[p] = alpha
        elif p in partition[1] : # si t(alpha_bar, p) est dans C
            M_c[p] = M[p]
        #print(f"{p} in partition {0 if p in partition[0] else 1}, M_c[p] = {M_c[p]} (alpha= {alpha}, M[p] = {M[p]})")
    return M_c

def _get_best_M_alpha(M, alpha,
                      K, edges_set, Mpj, Wpj, Wpqjk,
                      float_inf=1e9) :
    """
    Renvoie le meilleur coloriage a une alpha-expansion du coloriage actuel
    Renvoie aussi le cout dudit coloriage
    """
    G_alpha = _generate_G_alpha(M, alpha,K, edges_set, Wpqjk, Mpj, Wpj, float_inf)
    cut_value, partition = nx.minimum_cut(G_alpha, "alpha", "alpha_bar", capacity='weight')
    # cut_calue : le cout de M_star
    # partition : partition[0] contient le set des sommets connectes a alpha, partition[1] ceux de alpha_bar
    # print(f"partition[0] : {len(partition[0])}")
    # print(f"partition[1] : {len(partition[1])}")
    best_M_alpha = _get_M_from_cut(M, partition, alpha, K)
    return best_M_alpha, cut_value

def alpha_expansion(N, K, edges_set, Mpj, Wpj, Wpqjk, float_inf=1e9):
    """
    Renvoie le meilleur coloriage M
    """
    M = np.array([np.random.choice(np.where(Mpj[p] == True)[0]) for p in range(K)])
    current_cost = E(M, K, edges_set, Mpj, Wpj, Wpqjk)
    print(f"M_init = {M}, cost = {current_cost}")
    is_improving = True
    while is_improving :
        is_improving = False
        for alpha in tqdm(range(N), desc="Alpha expansion", leave=False) :
            M_star, cost_M_star = _get_best_M_alpha(M, alpha,K, edges_set, Mpj, Wpj, Wpqjk, float_inf)
            if cost_M_star < current_cost :
                is_improving = True
                M = M_star
                current_cost = cost_M_star
        print(f"Passe de alpha terminee. M = {M}, cost = {current_cost}")
    return M

if __name__ == "__main__" :
    
    # -- Chargement des donnees --

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
    print(f"Vjyxc cree : {Vjyxc.shape}")

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
    Wpj = np.load("tensors/Wpj.npy")
    Wpqjk = np.load("tensors/Wpqjk.npy", allow_pickle=True).item()

    print(f"Mpj ouvert : {Mpj.shape}")
    print(f"Wpj ouvert : {Wpj.shape}")
    print(f"Wpj = {Wpj[5:,5:]}")

    # edges
    Fp = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    K = len(Fp)
    edges_set = compute_edges(Fp)
    print(f"Nombre d'edges : {len(edges_set)}")


    M_final = alpha_expansion(N, K, edges_set, Mpj, Wpj, Wpqjk)
    print(M_final)
    np.save("tensors/M_final.npy", M_final)


# ###### jeu de test ######
# def test() :
#     # data
#     vertices = np.array([[0,0,0],[1,1,0],[1,0,0],[2,1,0],[2,0,0],[3,1,0]], dtype=np.float64)
#     triangles = np.array([[0,1,2],[1,2,3],[2,3,4],[3,4,5]])
#     mesh = o3d.geometry.TriangleMesh()
#     mesh.vertices = o3d.utility.Vector3dVector(vertices)
#     mesh.triangles = o3d.utility.Vector3iVector(triangles)

#     Fi = np.asarray(mesh.triangles)
#     K = len(Fi)
#     N = 3
#     edges_set = compute_edges(Fi)
#     print(edges_set)
#     M = np.array([0,1,1,2])

#     wij = np.zeros((K, N))
#     wijkl = np.zeros((K, K, N, N))

#     G_a = _generate_G_alpha(K, edges_set, wij, wijkl, M, 2)

#     print(G_a.nodes())
#     for u, v, data in G_a.edges(data=True):
#         print(f"Arête ({u}, {v}) avec poids : {data['weight_name']} = {data['weight']}")

# test()

# if __name__ == "__main__" :
#     # chargement des donnees
#     mesh = o3d.io.read_triangle_mesh("../fichiers_ply/mesh_cailloux_low.ply")
#     Fi = np.asarray(mesh.triangles)
#     K = len(Fi)
#     N = 53 # nombre d'images
#     image_path = "../downsampled/scene_l_"
#     Vj = [cv2.imread(image_path +  f"{j:04d}.jpeg") for j in range(1, N+1)]
#     h, w = Vj[0].shape[:2]

#     print(f"{N} images chargeés de shape ({h}, {w}), Mesh chargé, {K} faces")

#     # Obtenir l'ensemble des arretes
#     edges_set = []
#     # On va stocker les edges dans un dictionnaire (v1, v2) ou v1 < v2
#     edge_to_faces = dict() # {(v1, v2) : [i, j]} -> faces Fi et Fj reliees par l'arrete (v1, v2)
#     for i, (v1, v2, v3) in enumerate(Fi):
#         edges = [tuple(sorted((v1, v2))),
#                 tuple(sorted((v2, v3))),
#                 tuple(sorted((v3, v1)))]
#         for edge in edges:
#             if edge in edge_to_faces :
#                 edge_to_faces[edge].append(i)
#             else : 
#                 edge_to_faces[edge] = [i]
#     # On ne conserve que les arretes qui separent deux faces
#     for edge in edge_to_faces :
#         faces = edge_to_faces[edge]
#         if len(faces) == 2 :
#             edges_set.append(tuple(sorted(faces)))
#             # on stocke au format (i, j)
