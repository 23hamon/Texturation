import trimesh
import numpy as np
from collections import defaultdict
import xatlas
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import least_squares

#np.set_printoptions(threshold=np.inf)
np.random.seed(42)

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
uvs[:, 1] = 1 - uvs[:, 1] 

# if len(shared) == 2:
#     segment = trimesh.load_path(mesh.vertices[shared])
#     scene = trimesh.Scene([mesh, segment])
#     scene.show()


#creation image de texture (pas utilisé encore)
red_shades = np.array([[255, 0, 0], [100, 0, 0]])

def create_image(brightness_factor):
    Image = np.zeros((512, 512, 3), dtype=np.uint8)
    for y in range(512):
        for x in range(512):
            color = red_shades[0] if y + x < 511 else red_shades[1]
            Image[y, x] = np.clip(color * brightness_factor, 0, 255)
    return Image

# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# for i, factor in enumerate([0.5, 1.3, 1.8]):
#     axes[i].imshow(create_image(factor))
#     axes[i].axis('off')
#     axes[i].set_title(f'Luminosité x{factor}')

# plt.tight_layout()
# plt.show()

Vue1 = np.array(create_image(0.3), dtype=np.float32)
Vue2 = np.array(create_image(0.9), dtype=np.float32)
Vue3 = np.array(create_image(1.3), dtype=np.float32)

views = {0: Vue1, 1: Vue2, 2: Vue3}

# best views

n_views = 3
n_faces = len(mesh.faces)
Wij = np.random.rand(n_faces, n_views)
best_views = np.argmin(Wij, axis=1)

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

edges = set()
for i, neighbors in L.items():
    for j in neighbors:
        if i < j:
            edges.add((i, j))

# -- fonction d'intensité 
n_vertices = len(mesh.vertices)
n_views = Wij.shape[1]

# Fonctions d'intensité pour chaque vue
def intensity(texture_map, n_vertices, n_views, color=0):
    f = np.full((n_vertices, n_views), np.nan)  # remplie par défaut à NaN
    for j, vertices in C.items():
        for i in vertices:
            u, v = uvs[i]
            pixel = texture_map[int(v * 511), int(u * 511)]
            intensity = pixel[color]
            f[i, j] = intensity
    return f

# Calcul de la fonction d'intensité pour chaque vue automatiquement
f_all_views = {}

for view_id, view in views.items():
    f_all_views[view_id] = intensity(view, len(mesh.vertices), n_views, color=0)

print(f_all_views[1][0])

# Systeme equations lineaitre
lambda_seam = 100
index_map = { (i, j): idx for idx, (i, j) in enumerate(M) }
n = len(M)


def g(x):

    residuals_smoothness = []
    residuals_same_view = []
    lambda_seam = 100

    # entre idx1 et idx2, si i1 et 12 voisins : g[idx1] - g[idx2] ≈ 0
    for (i1, j) in M:
        for i2 in L[i1]:
            if (i2, j) in index_map:
                idx1 = index_map[(i1, j)]
                idx2 = index_map[(i2, j)]
                g_idx1 = x[idx1] 
                g_idx2 = x[idx2]
                residuals_smoothness.append(g_idx1 - g_idx2) 

    # Contraintes entre idx_j1 et idx_j2 : g[idx_j1] - g[idx_j2] ≈ f[idx_j2] - f[idx_j1]
    for (i, j1) in M:
        for (i2, j2) in M:
            if i == i2 and j1 != j2:
                idx_j1 = index_map[(i, j1)]
                idx_j2 = index_map[(i2, j2)]
                g_j1 = x[idx_j1]
                g_j2 = x[idx_j2]
                f_j1 = f_all_views[j1][i]
                f_j2 = f_all_views[j2][i]
                residuals_same_view.append(g_j1 - g_j2 - (f_j2 - f_j1))

    cost_smoothness = np.sum(np.array(residuals_smoothness)**2)
    cost_same_view = np.sum(np.array(residuals_same_view)**2)
    total_cost = cost_smoothness + lambda_seam * (cost_same_view)

    return np.array(total_cost, dtype=np.float32)


x0 = np.zeros(n) 

res = least_squares(g, x0, jac='2-point')

optimal_x = res.x
print(M)
print(optimal_x) # valeur de g pour le sommet 0 et la vue 0
print(f_all_views)

# Mettre à jour f_all_views avec la nouvelle fonction d'intensité
for j in f_all_views:
    for i in range (len(mesh.vertices)):
        if (i, j) in M:
            f_all_views[j][i] = f_all_views[j][i] + optimal_x[index_map[(i, j)]]
print('Nouvelle fonction d\'intensité :')
print(f_all_views)  # nouvelle valeur de f pour le sommet 0 et la vue 1
# valeur de g pour le sommet 0 et la vue 1



