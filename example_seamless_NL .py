import trimesh
import numpy as np
from collections import defaultdict
import xatlas
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import least_squares
from skimage.draw import polygon
np.random.seed(42)


#imports
data = np.load('fichiers_intermediaires/images.npz')
Vue1 = data['Vue1']
Vue2 = data['Vue2']
Vue3 = data['Vue3']
texture_map = np.load('fichiers_intermediaires/ex_texture_map.npy')


# EXEMPLE carré divisé en deux triangles
vertices = np.array([
    [0, 0, 0],  # 0
    [1, 0, 0],  # 1
    [1, 1, 0],  # 2
    [0, 1, 0]   # 3
])
faces = np.array([
    [0, 1, 2],  # triangle 1
    [0, 2, 3]   # triangle 2
])

#création du mesh, et de son mapping
mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
shared = np.intersect1d(mesh.faces[0], mesh.faces[1])
uvs = 1 - uvs #le mapping est inversé pour avoir 0 en bas à gauche et 1 en bas à droite 


#Best views
views = {0: Vue1, 1: Vue2, 2: Vue3} #on a donc trois vues différentes 
n_views = len(views)
n_faces = len(mesh.faces)
Wij = np.random.rand(n_faces, n_views)
best_views = np.argmin(Wij, axis=1) #et on récupère les meilleures vues pour chaque face , ici vue1 pour la face 0, et vue3 pour la face 1


# sommets adjacents

def adjacent_vertices(mesh):
    L = defaultdict(list)
    for face in mesh.faces:
        for i in range(3):
            for j in range(i + 1, 3):
                if face[j] not in L[face[i]]:
                    L[face[i]].append(face[j])
                if face[i] not in L[face[j]]:
                    L[face[j]].append(face[i])
    return L

L = adjacent_vertices(mesh)


# sommets colorés par vue 

C = defaultdict(set)
for face, view in enumerate(best_views):
    for vertex in mesh.faces[face]:
        C[view].add(vertex)

#sommets avec vues associées
M = [(i, j) for j, vertices in C.items() for i in vertices]

# -- fonction d'intensité 
n_vertices = len(mesh.vertices)
n_views = Wij.shape[1]


def intensity(view_id, color_channel):
    image = views[view_id]
    h, w = image.shape[:2]
    f = np.zeros((h, w))

    for face in mesh.faces:
        uvs_coords = uvs[face]
        c = (uvs_coords[:, 0] * w).astype(int)
        r = (uvs_coords[:, 1] * h).astype(int)
        rr, cc = polygon(r, c)
        for y, x in zip(rr, cc):
            if 0 <= x < w and 0 <= y < h: 
                f[y, x] = image[int(y), int(x), color_channel]
    return f



view_id = 0
color_channel = 2  

rouge_intensity = intensity(view_id, 0)

# # Fonctions d'intensité pour chaque vue
# def intensity(view_id,color):
#     f = np.zeros((texture_map.shape[0], texture_map.shape[1])) #matrice vide
#     for face in mesh.faces:
#         image = views[view_id]
#         uvs_coords = uvs[face] #on récupère les coordonnées UV des sommets de la face
#         r = uvs_coords[:, 0] * image.shape[1] #on récupère les lignes de chaque sommet
#         c = uvs_coords[:, 1] * image.shape[0] #et les colonnes de chaque sommet
#         rr, cc = polygon(r, c) #on crée un polygone
#         rr = np.clip(rr, 0, image.shape[0] - 1) #pas sortir de l'image
#         cc = np.clip(cc, 0, image.shape[1] - 1)
#         for pixel in zip(rr, cc):
#             x, y = pixel
#             color = image[x, y, color] #0,1ou2 si on veut  rouge, vert ou bleu
#             #on va mettre la couleur dans la texture_map
#             f[x, y] = color
#     return f

# rouge_intensity = intensity(Vue1, 0)
# print(rouge_intensity)




        #entre ces trois sommets, on va récupérer les couleurs de tous les pixels
        #dans la vue, et on va les mettre dans une liste

    
# # Systeme equations lineaitre
# lambda_seam = 100
# index_map = { (i, j): idx for idx, (i, j) in enumerate(M) }
# n = len(M)


# def g(x):

#     residuals_smoothness = []
#     residuals_same_view = []
#     lambda_seam = 100

#     # entre idx1 et idx2, si i1 et 12 voisins : g[idx1] - g[idx2] ≈ 0
#     for (i1, j) in M:
#         for i2 in L[i1]:
#             if (i2, j) in index_map:
#                 idx1 = index_map[(i1, j)]
#                 idx2 = index_map[(i2, j)]
#                 g_idx1 = x[idx1] 
#                 g_idx2 = x[idx2]
#                 residuals_smoothness.append(g_idx1 - g_idx2) 

#     # Contraintes entre idx_j1 et idx_j2 : g[idx_j1] - g[idx_j2] ≈ f[idx_j2] - f[idx_j1]
#     for (i, j1) in M:
#         for (i2, j2) in M:
#             if i == i2 and j1 != j2:
#                 idx_j1 = index_map[(i, j1)]
#                 idx_j2 = index_map[(i2, j2)]
#                 g_j1 = x[idx_j1]
#                 g_j2 = x[idx_j2]
#                 f_j1 = f_all_views[j1][i]
#                 f_j2 = f_all_views[j2][i]
#                 residuals_same_view.append(g_j1 - g_j2 - (f_j2 - f_j1))

#     cost_smoothness = np.sum(np.array(residuals_smoothness)**2)
#     cost_same_view = np.sum(np.array(residuals_same_view)**2)
#     total_cost = cost_smoothness + lambda_seam * (cost_same_view)

#     return np.array(total_cost, dtype=np.float32)


# x0 = np.zeros(n) 

# res = least_squares(g, x0, jac='2-point')

# optimal_x = res.x
# print(M)
# print(optimal_x) # valeur de g pour le sommet 0 et la vue 0
# print(f_all_views)

# # Mettre à jour f_all_views avec la nouvelle fonction d'intensité (g + f_all_views)
# for view_id, view in views.items():
#     for i in range(len(mesh.vertices)):
#         if (i, view_id) in M:
#             f_all_views[view_id][i] = optimal_x[index_map[(i, view_id)]] + f_all_views[view_id][i]
#         else:
#             f_all_views[view_id][i] = 0
     



