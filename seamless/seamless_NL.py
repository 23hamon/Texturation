import trimesh
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import least_squares
from skimage.draw import polygon
from example_texture_map import texture_triangles
from tqdm import tqdm
np.set_printoptions(threshold=np.inf)
np.random.seed(42)


# --- Chargement des données ---
data = np.load('seamless/images.npz')
Vue1, Vue2, Vue3 = data['Vue1'], data['Vue2'], data['Vue3']
texture_map = np.load('seamless/text_map.npy')
indices = np.load('seamless/indices.npy')
uvs = np.load('seamless/uvs.npy')
vmapping = np.load('seamless/mapping.npy')
mesh = trimesh.load('seamless/mesh.obj')
views = {0: Vue1, 1: Vue2, 2: Vue3}
n_views = len(views)
n_faces = len(mesh.faces)

print(uvs)
# Best views ici déterminé (comme pour les autres fichiers)
best_views = [0, 2]

#1) récupération des sommets voisins
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

#2) récupération des sommets colorés par chaque vue
C = defaultdict(set)
for face, view in enumerate(best_views):
    for vertex in mesh.faces[face]:
        C[view].add(vertex)

M = [(i, j) for j, vertices in C.items() for i in vertices]
index_map = { (i, j): idx for idx, (i, j) in enumerate(M)}
n = len(M)

# -- Fonctions pour intensités --
def intensity(view_id, channel_id):
    image = views[view_id]
    h, w = image.shape[:2]
    f = np.zeros((h, w))
    for face_id, face in enumerate(mesh.faces):
        uvs_coords = uvs[face]
        c = (uvs_coords[:, 0] * w).astype(int)
        r = (uvs_coords[:, 1] * h).astype(int)
        c = np.clip(c, 0, w - 1)
        r = np.clip(r, 0, h - 1)
        rr, cc = polygon(r, c)
        for y, x in zip(rr, cc):
            f[y, x] = image[int(y), int(x), channel_id]
    return f

# plt.figure(figsize=(6,6))
# im = plt.imshow(intensity(0,0), origin='lower')
# plt.axis('off')
# plt.title('Intensité VERTE Vue 0')
# plt.colorbar(im, label="Intensité")
# plt.show()

# print(Vue1[511, 511])

# red = intensity(0, 0)
# print(red[511, :])


def get_intensity_vertice(view_id, vertex_id, channel_id):
    f = intensity(view_id, channel_id)
    image_size = views[view_id].shape[:2]
    h, w = image_size
    uv_vertex = uvs[vertex_id]
    x = (uv_vertex[0] * w).astype(int)
    y = (uv_vertex[1] * h).astype(int)
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)
    print(x,y)
    if (x, y) not in M:
        return 0
    return f[y, x]

print("Intensité du vertex 0 dans la vue 0 (canal bleu) :", get_intensity_vertice(0, vertex_id=2, channel_id=2))


def intensity_all_views(views, M, channel_id):
    all_views_intensities = {}
    for view_id in range(len(views)):
        for vertex_id in range(len(mesh.vertices)):
            if (vertex_id, view_id) in M:
                all_views_intensities[(vertex_id, view_id)] = get_intensity_vertice(view_id, vertex_id, channel_id)
            else:
                all_views_intensities[(vertex_id, view_id)] = 0
    return all_views_intensities


# calcul de G
def build_g_function(intensity_all_views, L, M, index_map, lambda_seam=100):
    def g(x):
        residuals_smoothness = []
        residuals_same_view = []

        for (i1, j) in M:
            for i2 in L[i1]:
                if (i2, j) in index_map:
                    idx1 = index_map[(i1, j)]
                    idx2 = index_map[(i2, j)]
                    g_idx1 = x[idx1]
                    g_idx2 = x[idx2]
                    residuals_smoothness.append(g_idx1 - g_idx2)

        for (i, j1) in M:
            for (i2, j2) in M:
                if i == i2 and j1 != j2:
                    int_i_j1 = intensity_all_views[(i, j1)]
                    int_i_j2 = intensity_all_views[(i, j2)]
                    idx_j1 = index_map[(i, j1)]
                    idx_j2 = index_map[(i2, j2)]
                    g_j1 = x[idx_j1]
                    g_j2 = x[idx_j2]
                    residuals_same_view.append(g_j1 - g_j2 - (int_i_j2 - int_i_j1))

        cost_smoothness = np.sum(np.array(residuals_smoothness)**2)
        cost_same_view = np.sum(np.array(residuals_same_view)**2)
        total_cost = cost_smoothness + lambda_seam * cost_same_view

        return np.array(total_cost, dtype=np.float32)

    return g



# -- main  --

def run_channel(channel_id, color_name):
    print(f"\n--- Traitement du canal {color_name.upper()} ---")
    # Calculer les intensités pour toutes les vues
    intensities = {view_id: intensity(view_id, channel_id) for view_id in range(n_views)}
    all_views = intensity_all_views(views, M, channel_id)

    # Optimiser
    g = build_g_function(all_views, L, M, index_map)
    x0 = np.zeros(n)
    res = least_squares(g, x0, jac='2-point')
    optimal_x = res.x
    print("Résultat de l'optimisation :", optimal_x)

    # Mise à jour des intensités avec la solution optimale
    for (i, j) in all_views:
        idx = index_map.get((i, j), None)
        if idx is not None and (i,j) in M:
            all_views[(i, j)] += optimal_x[idx]
            all_views[(i, j)] = np.clip(all_views[(i, j)], 0, 255)

    # Générer l'image de texture pour ce canal
    texture_size = 512
    texture_image = np.ones((texture_size, texture_size, 3), dtype=np.uint8) * 255

    for face_id, face in enumerate(mesh.faces):
        for view_id in range(n_views):
            h, w = texture_image.shape[:2]
            uvs_coord = uvs[face]
            img_coords = (uvs_coord * [w, h]).astype(int)
            img_coords = np.clip(img_coords, 0, [w - 1, h - 1])

            colors = [[all_views.get((vertex, view_id), 0) if k == channel_id else 0 for k in range(3)] for vertex in face]
            texture_triangles(uvs_coord, colors, texture_image)

        plt.figure(figsize=(6,6))
        plt.imshow(texture_image, origin='lower')
        plt.axis('off')
        plt.title(f'Texture générée ({color_name})')
        plt.show()



# # --- main ---
# run_channel(0, "rouge")
# run_channel(1, "vert")
run_channel(2, "bleu")