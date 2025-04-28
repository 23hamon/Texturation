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
indices = np.load('fichiers_intermediaires/indices.npy')
uvs = np.load('fichiers_intermediaires/uvs.npy')
vmapping = np.load('fichiers_intermediaires/mapping.npy')
mesh = trimesh.load('fichiers_intermediaires/mesh.obj')

print("\n--- UVs par sommet ---")
for i, uv in enumerate(uvs):
    print(f"Sommet {i} : UV = {uv}")


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
    for face_id, face in enumerate(mesh.faces):
        if best_views[face_id] != view_id:
            continue  # on ignore les faces qui ne proviennent pas de cette vue
        uvs_coords = uvs[face]
        c = (uvs_coords[:, 0] * w).astype(int)
        r = (uvs_coords[:, 1] * h).astype(int)
        rr, cc = polygon(r, c)
        for y, x in zip(rr, cc):
            if 0 <= x < w and 0 <= y < h: 
                f[y, x] = image[int(y), int(x), color_channel]
    return f


#fonction pour récupérer la fonction d'intensité d'un sommet pour une vue donnée
def get_intensity_vertice(f, view_id, vertex_id):
    h, w = f.shape

    for face_id, face in enumerate(mesh.faces):
        if best_views[face_id] != view_id:
            continue
        if vertex_id in face:
            local_index = np.where(face == vertex_id)[0][0]
            uv = uvs[face[local_index]]
            print(uv)
            x = int(uv[0])
            y = int(uv[1])
            print(x, y)
            if 0 <= x < w and 0 <= y < h:
                return f[y, x]
    return None

            

print("\n--- UVs par face et sens ---")
def signed_area(uvs):
    return 0.5 * ((uvs[0,0]*uvs[1,1] - uvs[1,0]*uvs[0,1]) +
                  (uvs[1,0]*uvs[2,1] - uvs[2,0]*uvs[1,1]) +
                  (uvs[2,0]*uvs[0,1] - uvs[0,0]*uvs[2,1]))



image = Vue2
h, w = image.shape[:2]

plt.figure(figsize=(8, 8))
plt.imshow(image)  # pour être cohérent avec les UVs
plt.title("Vue 2 avec triangles et numéros des sommets")
plt.axis('off')

# Tracer les triangles
for face in mesh.faces:
    face_uv = uvs[face]
    x = face_uv[:, 0] * w
    y = face_uv[:, 1] * h

    # Fermer le triangle
    x = np.append(x, x[0])
    y = np.append(y, y[0])

    plt.plot(x, y, color='white', linewidth=1.5)

# Afficher les numéros de sommets
for idx, uv in enumerate(uvs):
    x = uv[0] * w
    y = uv[1] * h
    plt.text(x, y, str(idx), color='yellow', fontsize=12, ha='center', va='center', weight='bold')

plt.show()






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
     



