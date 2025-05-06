from backprojection import back_projeter
from utils import get_image_data
from algorithme.main_func import compute_edges

import numpy as np
from scipy import integrate

import random
import cv2
import open3d as o3d

def _distance_RGB(xRGB, yRGB) :
    """
    Calcule la distance euclidienne dans l'espace RGB 
    """
    return np.linalg.norm(xRGB-yRGB)

def _sample_edge(X1, X2, N_pts=100) :
    """
    Echantillonne un segment dans l'espace R^3 en un tableau de points
    Genere un tableau de N_pts points du segment [X1, X2] dans R^3,
    dans l'ordre et regulierement repartis
    """
    t = np.linspace(0, 1, N_pts)[:, None]  # shape (N_pts, 1)
    points = (1 - t) * X1 + t * X2
    return t.squeeze(), points

# /!\ POUR PLUS TARD
# plutot que de faire int(Y) on peut interpoler de maniere plus intelligente
def _weight_point_on_seam(Vjyxc,
                          X, j, k, 
                          rotj, tj, rotk, tk,
                          cam="l") :
    """
    Donne le poids individuel d'un point sur une couture
    /!\ N'est pertinent que si j et k sont bien des faces adjacentes
    - j, k  : index des vues (image 1 -> indice 0)
    """
    Yj, _, _ = back_projeter(X, rotj, tj, cam)
    Yk, _, _ = back_projeter(X, rotk, tk, cam)
    RGBj = Vjyxc[j, int(Yj[1]), int(Yj[0]), :]
    RGBk = Vjyxc[k, int(Yk[1]), int(Yk[0]), :]
    return _distance_RGB(RGBj, RGBk)

def _cross_weight_seam(Vjyxc,       # tenseur images
                       t, X_t,      # echantillonnage de l'arete : t ~= [0,1], X_t ~= [X1, X2]
                       j, k,        # vues j et k
                       rotj, tj, rotk, tk,
                       cam="l"      # type de camera
                       ) : 
    """
    Donne le cout croise d'une arete entre les vues j et k. 
    Ne fonctionnera que si l'arete est visible depuis les deux vues
    """
    y = np.array([_weight_point_on_seam(Vjyxc, X, j, k, rotj, tj, rotk, tk, cam) for X in X_t])
    integr = integrate.trapezoid(y, t)
    return integr

Wpqjk_dict = dict()

def Wpqjk(N, Vjyxc, cam,              # donnees sur les images
            rot_images, t_images,       # contient les rotations et translation
            vertices, edges_set,        # donnees sur le mesh
            Mpj,                        # matrice de visibilite d'une face sur une image
            p, q, j, k):
    """
    Renvoie la valeur de Wpqjk tout en la stockant en memoire 
    Construit Wpqjk_dict, la structure qui contient les donneees du tenseur de cout croise. 
    **Output** :
    ---
    Wpqjk_dict : dict
    - keys : edge (p, q) avec p < q
    - valkues : dict {(k, j) : cout} avec Mpj == 1 et Mqk == 1
    On ne stocke pas Wqp.. avec q > p puisque Wqpkj = Wpqjk
    /!\ Wpqjk != Wpqkj 
    /!\ au decalage d'indice : image 1 => indice 0
    """
    global Wpqjk_dict
    # initialisation 
    if not Wpqjk_dict : # si c'est vide on initialise
        Wpqjk_dict = {key : dict() for key in edges_set}
    if p == q :
        raise ValueError("Erreur : impossible de calculer le cout croise pour deux faces egales")
    if Mpj[p, j] and Mpj[q, j] and  Mpj[p, k] and Mpj[q, k] : # On impose que les deux faces soient visibles sur les deux vues
        if k == j :
            # si k = j, la vue est la mme et le cout vaut 0
            return 0
        if p < q : # si p et q sont bien ordonnes
            if (j, k) in Wpqjk_dict[(p, q)] :
                return Wpqjk_dict[(p,q)][(j, k)]
            else :
                v1, v2 = edges_set[(p,q)] 
                X1, X2 = vertices[v1], vertices[v2] # [X1, X2] est le segment de l'arete dans R^3
                rot_j, t_j = rot_images[j], t_images[j]
                rot_k, t_k = rot_images[k], t_images[k]
                # calcul de N_sample dans chacune des deux images
                Y_begin_j, _, _ = back_projeter(X1, rot_j, t_j, cam)
                Y_end_j, _, _ = back_projeter(X2, rot_j, t_j, cam) # [Y_end, Y_begin] est le segment de l'arete dans l'image
                N_sample_j = int(np.linalg.norm(Y_begin_j-Y_end_j))
                Y_begin_k, _, _ = back_projeter(X1, rot_k, t_k, cam)
                Y_end_k, _, _ = back_projeter(X2, rot_k, t_k, cam) # [Y_end, Y_begin] est le segment de l'arete dans l'image
                N_sample_k = int(np.linalg.norm(Y_begin_k-Y_end_k))
                # integration sur l'arete
                N_sample = max(N_sample_j, N_sample_k) # comme ca les deux segments sont parcourus en entier
                t, X_t = _sample_edge(X1, X2, N_sample)
                integr = _cross_weight_seam(Vjyxc, t, X_t, j, k, rot_j, t_j, rot_k, t_k, cam)
                Wpqjk_dict[(p, q)][(j, k)] = integr
                #print(f"W{p},{q},{j},{k} = {integr}")
                return integr
        else :
            return Wpqjk(N, Vjyxc, cam, rot_images, t_images, vertices, edges_set, Mpj, q, p, k, j)
    else :
        return np.inf

    

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

    N_rand = 15
    cles_aleatoires = random.sample(list(edges_set.keys()), N_rand)
    print(cles_aleatoires)
    vues_aleatoires = [(random.randint(0, 51), random.randint(0, 52)) for tyty in range(N_rand)]
    for idx, cle in enumerate(cles_aleatoires) :
        p, q = cle
        j, k = vues_aleatoires[idx]
        print(f"W_{p},{q},{j},{k} = {Wpqjk(N, Vjyxc, cam, rot_images, t_images, vertices, edges_set, Mpj, p, q, j, k)}")
    print("Et maintenant :\n\n")
    for idx, cle in enumerate(cles_aleatoires) :
        p, q = cle
        j, k = vues_aleatoires[idx]
        print(f"W_{p},{q},{j},{k} = {Wpqjk(N, Vjyxc, cam, rot_images, t_images, vertices, edges_set, Mpj, p, q, j, k)}")

    # #### test
    # imid = 38
    # rot, t = get_image_data(imid)

    # # N_k = 200
    # # keys = [random.choice(list(edges_set.keys())) for k in range(N_k)]
    # keys = list(edges_set.keys())

    # img = cv2.imread(image_path + f"{imid:04d}.jpeg")
    # img_with_points = img.copy()

    # for key in keys :
    #     p, q = key
    #     if Mpj[p, imid-1] and Mpj[q, imid-1] :
    #         v1, v2 = edges_set[key]
    #         x1, x2 = vertices[v1], vertices[v2]
    #         Y_begin, _, _ = back_projeter(x1, rot, t, "l")
    #         Y_end, _, _ = back_projeter(x2, rot, t, "l")
    #         N_sample = int(np.linalg.norm(Y_begin-Y_end))
    #         N_sample = int(N_sample*0.75)
    #         print(N_sample)
    #         _, X_t = sample_edge(x1, x2, N_sample)
    #         Y_t = []
            
    #         for x in X_t :
    #             Y, _, _ = back_projeter(x, rot, t, "l")
    #             Y_t.append((int(Y[0]), int(Y[1])))
    #             for pt in Y_t : 
    #                 x, y = pt
    #                 img_with_points[y, x] = (0, 0, 255) 
    # cv2.imwrite("image_avec_points.jpg", img_with_points)

    # edges_mesh = []
    # for key in keys :
    #     p, q = key
    #     v1, v2 = edges_set[key]
    #     x1, x2 = vertices[v1], vertices[v2]
    #     # Creer un segment rouge entre x1 et x2
    #     line_points = [x1, x2]
    #     line_indices = [[0, 1]]  # Connecte le point 0 à 1
    #     line_color = [[1, 0, 0]]  # Rouge

    #     line_set = o3d.geometry.LineSet(
    #         points=o3d.utility.Vector3dVector(line_points),
    #         lines=o3d.utility.Vector2iVector(line_indices),
    #     )
    #     line_set.colors = o3d.utility.Vector3dVector(line_color)
    #     edges_mesh.append(line_set)

    # # Afficher le mesh + le segment
    # mesh.paint_uniform_color([0.5, 0.5, 0.5])
    # mesh.compute_vertex_normals()    
    # o3d.visualization.draw_geometries([mesh] + edges_mesh)

