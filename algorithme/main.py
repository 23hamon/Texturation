#import open3d as o3d
import numpy as np
import cv2
import networkx as nx
import open3d as o3d

# tenseurs et fonctions

def compute_edges(Fi) :
    """
    Renvoie edges_set, le dictionnaire qui contiet les edges et leurs sommets
    {(i, j) : (v1, v2)} signifie l'arete entre la face i et la face j, dont les sommets sont v1 et v2
    """
    edges_set = dict()
    # On va stocker les edges dans un dictionnaire (v1, v2) ou v1 < v2
    edge_to_faces = dict() # {(v1, v2) : [i, j]} -> faces Fi et Fj reliees par l'arrete (v1, v2)
    for i, (v1, v2, v3) in enumerate(Fi):
        edges = [tuple(sorted((v1, v2))),
                tuple(sorted((v2, v3))),
                tuple(sorted((v3, v1)))]
        for edge in edges:
            if edge in edge_to_faces :
                edge_to_faces[edge].append(i)
            else : 
                edge_to_faces[edge] = [i]
    # On ne conserve que les arretes qui separent deux faces
    for edge in edge_to_faces :
        faces = edge_to_faces[edge]
        if len(faces) == 2 :
            edges_set[tuple(sorted(faces))] = edge
            # on stocke au format (i, j)
    return edges_set

def E_Q(K, # nombre de faces
        wij,
        M  # vecteur des vues par face
        ) :
    return sum([wij[i, M[i]] for i in range(K)])

def E_S(edges_set, # liste des edges [(i, j), ...]
        wijkl,
        M
        ) :
    return sum([wijkl[i, j, M[i], M[j]] for id, (i, j) in enumerate(edges_set)])

def E(K, edges_set, 
      M, 
      llambda=1) :
    return E_Q(K, M) + llambda * E_S(edges_set, M)

# Optimisation de l'energie avec alpha-expansion



# Trouver M_hat qui minimise l'energie dans les voisins a une alpha-expansion

def _generate_G_alpha(K, edges_set,
                      wij, wijkl,
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
        weight_tp_alpha = wij[i, alpha]
        weight_tp_alpha_bar = np.inf if M[i] == alpha else wij[i, M[i]]
        G_alpha.add_edge(i, "alpha", type="t-link", weight=weight_tp_alpha, weight_name=f"wij[{i},alpha]") # t{p, alpha}
        G_alpha.add_edge(i, "alpha_bar", type="t-link", weight=weight_tp_alpha_bar, weight_name=f"{'inf' if weight_tp_alpha_bar == np.inf else f'wij[{i}, M[{i}]]'}")
    # n-links : connecter les pixels voisins 
    for idx, (p, q) in enumerate(edges_set) :
        if M[p] == M[q] :
            weight_e_p_q = wijkl[p, q, M[p], alpha]
            G_alpha.add_edge(p, q, type="n-link", weight=weight_e_p_q, weight_name=f"wijkl[{p}, {q}, M[{p}], alpha]") # e{p,q}
        else :
            weight_e_p_a = wijkl[p, q, M[p], alpha]
            weight_e_a_q = wijkl[p, q, alpha, M[q]]
            weight_t_a_alpha = wijkl[p, q, M[p], M[q]]
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

def _get_best_M_alpha(K, edges_set,
                      M, alpha,
                      wij, wijkl) :
    """
    Renvoie le meilleur coloriage a une alpha-expansion du coloriage actuel
    """
    G_alpha = _generate_G_alpha(K, edges_set, wij, wijkl, M, alpha)
    best_cut = _get_minimal_cut(G_alpha)
    best_M_alpha = _get_M_from_cut(K, M, best_cut, alpha)
    return best_M_alpha

def alpha_expansion(K, N, edges_set,    # donnees de l'image et des vues
                    wij, wijkl          # cout et cout-croise
                    ):
    """
    Renvoie le meilleur coloriage M
    """
    M = np.array([np.random.randint(N) for i in range(K)])
    is_improving = True
    while is_improving :
        is_improving = False
        for alpha in range(N) :
            M_star = _get_best_M_alpha(K, edges_set, M, alpha, wij, wijkl)
            if E(K, edges_set, M_star) < E(K, edges_set, M) :
                is_improving = True
                M = M_star
    return M

if __name__ == "__main__" :
    # chargement des donnees
    mesh = o3d.io.read_triangle_mesh("../fichiers_ply/mesh_cailloux_low.ply")
    Fi = np.asarray(mesh.triangles)
    K = len(Fi)
    N = 53 # nombre d'images
    image_path = "../downsampled/scene_l_"
    Vj = [cv2.imread(image_path +  f"{j:04d}.jpeg") for j in range(1, N+1)]
    h, w = Vj[0].shape[:2]

    print(f"{N} images chargeés de shape ({h}, {w}), Mesh chargé, {K} faces")
    
    wij = NotImplemented
    wijkl = NotImplemented
    edges_set = compute_edges(Fi)

    M_best = alpha_expansion(K, N, edges_set, wij, wijkl)
    print(M_best)




###### jeu de test ######
def test() :
    # data
    vertices = np.array([[0,0,0],[1,1,0],[1,0,0],[2,1,0],[2,0,0],[3,1,0]], dtype=np.float64)
    triangles = np.array([[0,1,2],[1,2,3],[2,3,4],[3,4,5]])
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    Fi = np.asarray(mesh.triangles)
    K = len(Fi)
    N = 3
    edges_set = compute_edges(Fi)
    print(edges_set)
    M = np.array([0,1,1,2])

    wij = np.zeros((K, N))
    wijkl = np.zeros((K, K, N, N))

    G_a = _generate_G_alpha(K, edges_set, wij, wijkl, M, 2)

    print(G_a.nodes())
    for u, v, data in G_a.edges(data=True):
        print(f"Arête ({u}, {v}) avec poids : {data['weight_name']} = {data['weight']}")

test()

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
