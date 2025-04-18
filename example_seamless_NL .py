import trimesh
import numpy as np
from collections import defaultdict
import xatlas
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm


#np.set_printoptions(threshold=np.inf)
np.random.seed(42)
data = np.load('fichiers_intermediaires/images.npz')
Vue1 = data['Vue1']
Vue2 = data['Vue2']
Vue3 = data['Vue3']


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

mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
shared = np.intersect1d(mesh.faces[0], mesh.faces[1])
uvs[:, 1] = 1 - uvs[:, 1] #le mapping est inversé

views = {0: Vue1, 1: Vue2, 2: Vue3} #on a donc trois vues différentes 



# best views

n_views = 3
n_faces = len(mesh.faces)
Wij = np.random.rand(n_faces, n_views)
best_views = np.argmin(Wij, axis=1) #et on récupère les meilleures vues pour chaque face , ici vue1 pour la face 0, et vue3 pour la face 1

# plt.figure(figsize=(15, 5))
# for i in range(n_faces):
#     img = views[best_views[i]]  # Sélection de l'image correspondante à la meilleure vue pour la face i
#     plt.subplot(1, n_faces, i+1)
#     plt.imshow(img)
#     plt.axis('off')
#     plt.title(f'Face {i} - Vue {best_views[i]+1}')

# plt.tight_layout()
# plt.show()

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

#on crée une image de texture qui prend la Vue1 pour colorer la face 0 du maillage, et la Vue3 pour colorer la face 1 du maillage

texture_image = np.zeros((512, 512, 3), dtype=np.uint8)
projected = vertices[:, :2] * [511, 511]
for i, face in enumerate(faces):
    img = views[best_views[i]]  
    v0, v1, v2 = projected[face].astype(np.float32)
    # bounding box du triangle
    min_x = max(int(np.floor(min(v0[0], v1[0], v2[0]))), 0)
    max_x = min(int(np.ceil(max(v0[0], v1[0], v2[0]))), texture_image.shape[1] - 1)
    min_y = max(int(np.floor(min(v0[1], v1[1], v2[1]))), 0)
    max_y = min(int(np.ceil(max(v0[1], v1[1], v2[1]))), texture_image.shape[0] - 1)

    # Matrice de transformation (pour la barycentricité)
    T = np.array([[v1[0] - v0[0], v2[0] - v0[0]], [v1[1] - v0[1], v2[1] - v0[1]]])
    T_inv = np.linalg.inv(T)

    # Remplir chaque pixel du triangle avec la couleur de la vue
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            p = np.array([x - v0[0], y - v0[1]])
            u, v = T_inv @ p
            w = 1 - u - v

            if 0 <= u <= 1 and 0 <= v <= 1 and 0 <= w <= 1:
                # Interpolation barycentrique
                color = u * img[int(v0[1]), int(v0[0])] + v * img[int(v1[1]), int(v1[0])] + w * img[int(v2[1]), int(v2[0])]
                texture_image[y, x] = np.clip(color, 0, 255)

# Affichage de l'image de texture
plt.imshow(texture_image)
plt.axis('off')
plt.show()

# # Fonctions d'intensité pour chaque vue
# def intensity(texture_map, n_vertices, n_views, color=0):
#     """"
#     param color: couleur à extraire (0 = rouge, 1 = vert, 2 = bleu)
#     output: tableau 2D de forme (n_views, n_vertices) contenant les intensités
#     """
    

#     for view in range(n_views):
#         intensity_map[view] = texture_map[view][:, color]

#     return intensity_map

# # Calcul de la fonction d'intensité pour chaque vue automatiquement
# f_all_views = {}

# for view_id, view in views.items():
#     f_all_views[view_id] = intensity(view, len(mesh.vertices), n_views, color=0)

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

# # Mettre à jour f_all_views avec la nouvelle fonction d'intensité
# for j in f_all_views:
#     for i in range (len(mesh.vertices)):
#         if (i, j) in M:
#             f_all_views[j][i] = f_all_views[j][i] + optimal_x[index_map[(i, j)]]
# print('Nouvelle fonction d\'intensité :')
# print(f_all_views)  # nouvelle valeur de f pour le sommet 0 et la vue 1
# # valeur de g pour le sommet 0 et la vue 1



