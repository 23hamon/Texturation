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
               float_inf) :
    """
    Renvoie le cout croise entre deux faces pour deux vues
    Si p et q sont visibles depuis les vues j et k, alors Wpqjk est fini
    Sinon Wpqjk vaut np.inf
    Si j = k, Wpqjk = 0
    Wpqjk = Wqpkj
    """
    if Mpj[p, j] and Mpj[q, k] :
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
        edges_set, Mpj, Wpqjk, 
        float_inf) :
    return sum([full_Wpqjk(p, q, M[p], M[q],
                           Mpj, Wpqjk, float_inf)
                           for _, (p, q) in enumerate(edges_set)])

def E(M,
      K, edges_set, Mpj, Wpj, Wpqjk,
      float_inf, llambda=1) :
    return E_Q(M, K, Wpj) + llambda * E_S(M, edges_set, Mpj, Wpqjk, float_inf)

# Optimisation de l'energie avec alpha-expansion


def _generate_G_alpha(M, alpha, 
                      K, edges_set, Wpqjk, Mpj, Wpj, 
                      float_inf) :
    """
    Renvoie G_alpha, le graph dont la coupe minimale donne le nouveau coloriage
    - M est le vecteur d'attribution des faces (np array d'entiers entre 1 et N et de taille K)
    - alpha est une face, un entier entre 1 et N
    """
    G_alpha = nx.Graph()

    # Noeuds -- 
    # On ajoute explicitement les pixels de l'image (faces triangulaires)
    # alpha est la "source"
    # alpha_bar est le "puit"
    # noeuds speciaux : a{p, q} pour un pixel p et un pixel q voisins tels que mp != mq
    idx_special_node = K
    special_nodes = dict()
    for _, (p, q) in enumerate(edges_set) :
        if M[p] != M[q] :
            special_nodes[(p, q)] = idx_special_node
            #print(f"a({p},{q}) : idx = {idx_special_node}")
            idx_special_node += 1
    #print(f"{idx_special_node} noeuds dans le graph")
    # Edges --
    # t-links : connecter chaque pixel a alpha et alpha_bar
    for p in range(K) :
        weight_tp_alpha = Wpj[p, alpha]
        weight_tp_alpha_bar = float_inf if M[p] == alpha else Wpj[p, M[p]]
        G_alpha.add_edge(p, "alpha", weight=weight_tp_alpha) # t{p, alpha}
        G_alpha.add_edge(p, "alpha_bar", weight=weight_tp_alpha_bar) #t{p, alpha_bar}
        #print(f"t-links - t({p},alpha) : {weight_tp_alpha} - t({p},alpha_bar) : {weight_tp_alpha_bar}")
    for _, (p, q) in enumerate(edges_set) : # noeuds speciaux
        if M[p] != M[q] :
            a = special_nodes[(p, q)]
            weight_t_a_alpha_bar = full_Wpqjk(p, q, M[p], M[q], Mpj, Wpqjk, float_inf)
            G_alpha.add_edge(a, "alpha_bar", weight=weight_t_a_alpha_bar) # t{a, alpha_bar}
            #print(f"t-link - t[a({p},{q}), alpha_bar] (idx={a}) : {weight_t_a_alpha_bar}")
    # n-links : connecter les pixels voisins 
    for _, (p, q) in enumerate(edges_set) :
        if M[p] == M[q] :
            weight_e_p_q = full_Wpqjk(p, q, M[p], alpha, Mpj, Wpqjk, float_inf)
            G_alpha.add_edge(p, q, weight=weight_e_p_q) # e{p,q}
            #print(f"n-link - n({p},{q}) : {weight_e_p_q}")
        else :
            a = special_nodes[(p, q)]
            weight_e_p_a = full_Wpqjk(p, q, M[p], alpha, Mpj, Wpqjk, float_inf)
            weight_e_a_q = full_Wpqjk(p, q, alpha, M[q], Mpj, Wpqjk, float_inf)
            G_alpha.add_edge(p, a, weight=weight_e_p_a) # e{p,a}
            G_alpha.add_edge(a, q, weight=weight_e_a_q) # e{a,q}
            #print(f"n-link - n[a({p},{q}), {q}] (idx={a}) : {weight_e_a_q}")
            #print(f"n-link - n[{p}, a({p},{q})] (idx={a}) : {weight_e_p_a}")
    return G_alpha


def _get_best_M_alpha(M, alpha,
                      K, edges_set, Mpj, Wpj, Wpqjk,
                      float_inf) :
    """
    Renvoie le meilleur coloriage a une alpha-expansion du coloriage actuel
    Renvoie aussi le cout dudit coloriage
    """
    G_alpha = _generate_G_alpha(M, alpha,K, edges_set, Wpqjk, Mpj, Wpj, float_inf) 
    #print(f"G_{alpha} genere ")
    cut_value, partition = nx.minimum_cut(G_alpha, "alpha", "alpha_bar", capacity="weight")
    reachable, non_reachable = partition    # reachable : les sommets connectes a alpha dans le graphe coupe G(C)
    M_c = np.zeros((K,), dtype=int) # generer le meilleur label 
    for p in range(K) :
        if p in non_reachable :     # si p est reachable, alors t{p, alpha} n'a pas ete coupe donc n'appartient pas a C
            M_c[p] = alpha
        else :                  # si p n'est pas reachable alors t{p, alpha} a ete coupe
            M_c[p] = M[p]  
    #print("coupe minimale calculee")
    return M_c, cut_value

def invalid_p(K, M, Mpj):
    return [p for p in range(K) if not Mpj[p, M[p]]]

def alpha_expansion(N, K, edges_set, Mpj, Wpj, Wpqjk, float_inf=np.inf):
    """
    Renvoie le meilleur coloriage M
    """
    M = np.array([np.random.choice(np.where(Mpj[p] == True)[0]) for p in range(K)])
    current_cost = E(M, K, edges_set, Mpj, Wpj, Wpqjk, float_inf)
    print(f"M_init = {M}, cost = {current_cost}")
    is_improving = True
    while is_improving :
        is_improving = False
        for alpha in tqdm(range(N), desc="Alpha expansion", leave=False) :
            M_star, cost_M_star = _get_best_M_alpha(M, alpha, K, edges_set, Mpj, Wpj, Wpqjk, float_inf)
            # hypothese : parfois il n'y a pas de coupe minimale finie possible, mais il en calcule une quand meme, et etrangement son poids n'est pas infini
            if cost_M_star < current_cost and E(M_star, K, edges_set, Mpj, Wpj, Wpqjk, float_inf) < current_cost : # parfois different du poids de la coupe
                # print(f"{cost_M_star} < {current_cost}")
                # invalid_p_list = invalid_p(K, M_star, Mpj)
                # print(f"pixels invalides dans le nouveau M : {invalid_p_list}")
                # print(f"labels des pixels invalides : {[M_star[p] for p in invalid_p_list]}")
                # print(f"poids des pixels invalides sur la vue {alpha} : {[f'W{p},{M_star[p]} = {Wpj[p, M_star[p]]}' for p in invalid_p_list]}")
                # print(f"Nouveau M : {M_star} - cost = {cost_M_star} = {E(M_star, K, edges_set, Mpj, Wpj, Wpqjk, float_inf)} (E_S = {E_S(M_star, edges_set, Mpj, Wpqjk, float_inf)}, E_Q = {E_Q(M_star, K, Wpj)})")
                is_improving = True
                M = M_star
                current_cost = cost_M_star
        print(f"Passe de alpha terminee. M = {M}, cost = {current_cost} (E_S = {E_S(M, edges_set, Mpj, Wpqjk, float_inf)}, E_Q = {E_Q(M, K, Wpj)})")
    return M

if __name__ == "__main__" :
    # mesh2 = o3d.io.read_triangle_mesh("ply/mesh_cailloux_luca_CLEAN.ply")
    # mesh1 = o3d.io.read_triangle_mesh("ply/mesh_cailloux_luca_LOW.ply")
    # print(f"mesh1 : N_tri = {len(np.asarray(mesh1.triangles))} - N_ver = {len(np.asarray(mesh1.vertices))}")
    # print(f"mesh2 : N_tri = {len(np.asarray(mesh2.triangles))} - N_ver = {len(np.asarray(mesh2.vertices))}")
    # print(f"mesh1 : {len(compute_edges(np.asarray(mesh1.triangles)))} edges")
    # print(f"mesh2 : {len(compute_edges(np.asarray(mesh2.triangles)))} edges")

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

    Mpj_copy = Mpj.copy()
    Wpj_copy = Wpj.copy()

    print(f"Mpj ouvert : {Mpj.shape}")
    print(f"Wpj ouvert : {Wpj.shape}")

    # edges
    Fp = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    K = len(Fp)
    edges_set = compute_edges(Fp)
    print(f"Nombre d'edges : {len(edges_set)}")

    print(f"M1050,51 = {Mpj[1050,51]}, W1050,51 = {Wpj[1050,51]}")
    M_final = alpha_expansion(N, K, edges_set, Mpj, Wpj, Wpqjk, 1e20)
    print(M_final)
    print(f"E(M_final) = {E(M_final, K, edges_set, Mpj, Wpj, Wpqjk, 1e20)}")
    np.save("tensors/M_final.npy", M_final)

    # print("comparaison de M et W")
    # for p in range(K) :
    #     for j in range(N) :
    #         if Mpj[p, j] != Mpj_copy[p, j] :
    #             print(f"Mpj : ancien = {Mpj_copy[p, j]}, nao = {Mpj[p, j]}")
    #         if Wpj[p, j] != Wpj_copy[p, j] :
    #             print(f"Mpj : ancien = {Wpj_copy[p, j]}, nao = {Wpj[p, j]}")
    # print(f"M1050,51 = {Mpj[1050,51]}, W1050,51 = {Wpj[1050,51]}")

    # invalid_p_list = invalid_p(K, M_final, Mpj)
    # print(f"pixels invalides dans le nouveau M : {invalid_p_list}")
    # print(f"labels des pixels invalides : {[M_final[p] for p in invalid_p_list]}")
    # print(f"poids des pixels invalides : {[f'W{p},{M_final[p]} = {Wpj[p, M_final[p]]}' for p in invalid_p_list]}")

    # def test() :
    #     # data
    #     vertices = np.array([[0,0,0],[1,1,0],[1,0,0],[2,1,0],[2,0,0],[3,1,0]], dtype=np.float64)
    #     triangles = np.array([[0,1,2],[1,2,3],[2,3,4],[3,4,5]])
    #     mesh = o3d.geometry.TriangleMesh()
    #     mesh.vertices = o3d.utility.Vector3dVector(vertices)
    #     mesh.triangles = o3d.utility.Vector3iVector(triangles)

        
    #     K = len(triangles)
    #     N = 3
    #     edges_set = compute_edges(triangles)
    #     print(edges_set)
    #     M = np.array([0,1,1,2])
    #     alpha = 2
    #     views_set ={(0,1),(0,2), (1,0), (1,2), (2,0), (2,1)}
    #     Wpqjk = {(p,q) : {(j, k) : 10*p + q + 0.1 * j + 0.01 * k for _, (j, k) in enumerate(views_set)} for _, (p,q) in enumerate(edges_set)}
    #     Wpj = np.array([[101,102,103],
    #                     [201,202,203],
    #                     [301,302,303],
    #                     [401,402,403]])
    #     Mpj = np.full((4,3), True)
    #     float_inf=1e9
    #     G_alpha = _generate_G_alpha(M, alpha, K, edges_set, Wpqjk, Mpj, Wpj, float_inf)
    #     nxG_alpha = G_alpha.get_nx_graph()
    #     print(f"nodes : {list(nxG_alpha.nodes)}")
    #     print(f"edges : {nxG_alpha.edges(data=True)}")
    # test()


   