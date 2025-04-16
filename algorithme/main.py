#import open3d as o3d
import numpy as np
import cv2
import networkx as nx
import open3d as o3d

from main_func import compute_edges
from ..utils import get_image_data

# tenseurs et fonctions
from ..cross_weight import Wpqjk

def E_Q(K, # nombre de faces
        Wpj,
        M  # vecteur des vues par face
        ) :
    return sum([Wpj[p, M[p]] for p in range(K)])

def E_S(N, Vjyxc, cam, rot_images, t_images, vertices, edges_set, 
        Mpj,
        M
        ) :
    return sum([ Wpqjk(N, Vjyxc, cam, rot_images, t_images, 
                       vertices, edges_set, Mpj, 
                       p, q, M[p], M[q])
                       for id, (p, q) in enumerate(edges_set)])

def E(N, K, Vjyxc, cam, rot_images, t_images, vertices, edges_set,
      Mpj, Wpj,
      M, 
      llambda=1) :
    return E_Q(K, Wpj, M) + llambda * E_S(N, Vjyxc, cam, rot_images, t_images, vertices, edges_set, Mpj, M)

# Optimisation de l'energie avec alpha-expansion



# Trouver M_hat qui minimise l'energie dans les voisins a une alpha-expansion

def _generate_G_alpha(N, K, Vjyxc, cam, rot_images, t_images, vertices, edges_set,
                      Mpj, Wpj,
                      M, alpha) :
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
    for i in range(K) :
        G_alpha.add_node(i)
    # noeuds speciaux : a{p, q} pour un pixel p et un pixel q voisins tels que mp != mq
    for idx, (p, q) in enumerate(edges_set) :
        if M[p] != M[q] :
            G_alpha.add_node(f"a({p},{q})")

    # Edges --
    # t-links : connecter chaque pixel a alpha et alpha_bar
    for i in range(K) :
        weight_tp_alpha = Wpj[i, alpha]
        weight_tp_alpha_bar = np.inf if M[i] == alpha else Wpj[i, M[i]]
        G_alpha.add_edge(i, "alpha", type="t-link", weight=weight_tp_alpha, weight_name=f"wij[{i},alpha]") # t{p, alpha}
        G_alpha.add_edge(i, "alpha_bar", type="t-link", weight=weight_tp_alpha_bar, weight_name=f"{'inf' if weight_tp_alpha_bar == np.inf else f'wij[{i}, M[{i}]]'}")
    # n-links : connecter les pixels voisins 
    for idx, (p, q) in enumerate(edges_set) :
        if M[p] == M[q] :
            weight_e_p_q = Wpqjk(N, Vjyxc, cam, rot_images, t_images, vertices, edges_set, Mpj, 
                                 p, q, M[p], alpha)
            G_alpha.add_edge(p, q, type="n-link", weight=weight_e_p_q, weight_name=f"wijkl[{p}, {q}, M[{p}], alpha]") # e{p,q}
        else :
            weight_e_p_a = Wpqjk(N, Vjyxc, cam, rot_images, t_images, vertices, edges_set, Mpj, 
                                 p, q, M[p], alpha)
            weight_e_a_q = Wpqjk(N, Vjyxc, cam, rot_images, t_images, vertices, edges_set, Mpj, 
                                 p, q, alpha, M[q])
            weight_t_a_alpha = Wpqjk(N, Vjyxc, cam, rot_images, t_images, vertices, edges_set, Mpj, 
                                     p, q, M[p], M[q])
            a = f"a({p},{q})"
            G_alpha.add_edge(p, a, type="n-link", weight=weight_e_p_a, weight_name=f"wijkl[{p}, {q}, M[{p}], alpha]") # e{p,a}
            G_alpha.add_edge(a, q, type="n-link", weight=weight_e_a_q, weight_name=f"wijkl[{p}, {q}, alpha, M[{q}]]") # e{a,q}
            G_alpha.add_edge(a, "alpha_bar", type="t-link", weight=weight_t_a_alpha, weight_name=f"wijkl[{p}, {q}, M[{p}], M[{q}]]") # t{a,alpha}
    # Poids des aretes
    return G_alpha

def _get_minimal_cut(G_alpha) :
    """
    Renvoie la liste des aretes de la coupe minimale de G_alpha
    """
    cut_value, partition = nx.minimum_cut(G_alpha, "alpha", "alpha_bar", capacity='weight')
    cut_edges = []
    for u, v in G_alpha.edges():
        if (u in partition[0] and v in partition[1]) or (u in partition[1] and v in partition[0]):
            cut_edges.append((u, v))
    return cut_edges

def _get_M_from_cut(K,
                    M,
                    cut, alpha) :
    """
    Renvoie le vecteur de label M corresponant a la cut du graphe alpha
    """
    M_c = np.zeros((K,))
    for p in range(K) :
        if (p, "alpha") in cut : # si t(alpha, p) est dans C
            M_c[p] = alpha
        else :                   # si t(alpha_bar, p) est dans C
            M_c[p] = M[p]
    return M_c

def _get_best_M_alpha(N, K, Vjyxc, cam, rot_images, t_images, vertices, edges_set, 
                      Mpj, Wpj,
                      M, alpha
                      ) :
    """
    Renvoie le meilleur coloriage a une alpha-expansion du coloriage actuel
    """
    G_alpha = _generate_G_alpha(N, K, Vjyxc, cam, rot_images, t_images, vertices, edges_set, Mpj, Wpj, M, alpha)
    best_cut = _get_minimal_cut(G_alpha)
    best_M_alpha = _get_M_from_cut(K, M, best_cut, alpha)
    return best_M_alpha

def alpha_expansion(N, K, Vjyxc, cam, rot_images, t_images, vertices, edges_set, 
                    Mpj, Wpj
                    ):
    """
    Renvoie le meilleur coloriage M
    """
    M = np.array([np.random.randint(N) for i in range(K)])
    is_improving = True
    while is_improving :
        is_improving = False
        for alpha in range(N) :
            M_star = _get_best_M_alpha(N, K, Vjyxc, cam, rot_images, t_images, vertices, edges_set, Mpj, Wpj, M, alpha)
            if E(N, K, Vjyxc, cam, rot_images, t_images, vertices, edges_set, Mpj, Wpj, M_star) < E(N, K, Vjyxc, cam, rot_images, t_images, vertices, edges_set, Mpj, Wpj, M) :
                is_improving = True
                M = M_star
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
    mesh = o3d.io.read_triangle_mesh("fichiers_ply/mesh_cailloux_luca_LOW.ply")
    # visibilite des faces
    Mpj = np.load("fichiers_intermediaires/MijLOW.npy")
    print(f"Mpj ouvert : {Mpj.shape}")
    # edges
    Fp = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    K = len(Fp)
    edges_set = compute_edges(Fp)
    print(f"Nombre d'edges : {len(edges_set)}")



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
