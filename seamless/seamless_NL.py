import trimesh
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import least_squares
from skimage.draw import polygon
from example_texture_map import texture_triangles
np.set_printoptions(threshold=np.inf)
np.random.seed(42)

# --- Chargement des données ---
data = np.load('seamless/images.npz')
Vue1, Vue2, Vue3 = data['Vue1'], data['Vue2'], data['Vue3']

texture_map = np.load('seamless/text_map.npy')
indices = np.load('seamless/indices.npy')
uvs = np.load('seamless/uvs.npy')
uvs = 1 - (uvs[:, [1, 0]])  # inversion
vmapping = np.load('seamless/mapping.npy')
mesh = trimesh.load('seamless/mesh.obj')

views = {0: Vue1, 1: Vue2, 2: Vue3}
n_views = len(views)
n_faces = len(mesh.faces)

# Best views ici déterminé
best_views = [0, 2]

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

# Initialisation
C = defaultdict(set)
for face, view in enumerate(best_views):
    for vertex in mesh.faces[face]:
        C[view].add(vertex)

M = [(i, j) for j, vertices in C.items() for i in vertices]
index_map = { (i, j): idx for idx, (i, j) in enumerate(M)}
n = len(M)

# --- Fonctions pour intensités ---
def intensity(view_id, color_channel):
    image = views[view_id]
    h, w = image.shape[:2]
    f = np.zeros((h, w))
    for face_id, face in enumerate(mesh.faces):
        if best_views[face_id] != view_id:
            continue
        uvs_coords = uvs[face]
        c = (uvs_coords[:, 0] * w).astype(int)
        r = (uvs_coords[:, 1] * h).astype(int)
        rr, cc = polygon(r, c)
        for y, x in zip(rr, cc):
            if 0 <= x < w and 0 <= y < h:
                f[y, x] = image[int(y), int(x), color_channel]
    return f

def get_intensity_vertice(f, view_id, vertex_id):
    h, w = f.shape
    for face_id, face in enumerate(mesh.faces):
        if best_views[face_id] != view_id:
            continue
        if vertex_id in face:
            local_index = np.where(face == vertex_id)[0][0]
            uv = uvs[face[local_index]]
            x = int(uv[0] * w)
            y = int(uv[1] * h)
            if 0 <= x < w and 0 <= y < h:
                return f[y, x]
    return 0

def intensity_all_views(views, intensities, M):
    all_views_intensities = {}
    for view_id in range(len(views)):
        for vertex_id in range(len(mesh.vertices)):
            if (vertex_id, view_id) in M:
                all_views_intensities[(vertex_id, view_id)] = get_intensity_vertice(intensities[view_id], view_id, vertex_id)
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
def run_texture_generation():
    print("\n--- Début de la génération de la texture complète (R, G, B) ---")
    
    # Calcul des intensités pour chaque canal
    intensities_per_channel = []
    all_views_per_channel = []

    for channel_id in range(3):
        intensities = {view_id: intensity(view_id, channel_id) for view_id in range(n_views)}
        all_views = intensity_all_views(views, intensities, M)
        intensities_per_channel.append(intensities)
        all_views_per_channel.append(all_views)

    # Optimisation pour chaque canal
    optimal_x_per_channel = []

    for channel_id in range(3):
        g = build_g_function(all_views_per_channel[channel_id], L, M, index_map)
        x0 = np.zeros(n)
        res = least_squares(g, x0, jac='2-point')
        optimal_x_per_channel.append(res.x)
        print(f"Canal {channel_id} : Coût final = {res.x}")

    # Mise à jour des intensités optimisées
    for channel_id in range(3):
        all_views = all_views_per_channel[channel_id]
        optimal_x = optimal_x_per_channel[channel_id]
        for (i, j) in all_views:
            idx = index_map.get((i, j), None)
            if idx is not None and (i, j) in M:
                all_views[(i, j)] += optimal_x[idx]
                all_views[(i, j)] = np.clip(all_views[(i, j)], 0, 255)
        print(f"Intensités optimisées pour le canal {channel_id} :")
        for (i, j) in M:
            idx = index_map.get((i, j), None)
            if idx is not None and (i, j) in M:
                all_views[(i, j)] = np.clip(all_views[(i, j)], 0, 255)
                print(f"Vertex {i}, Vue {j} : Intensité optimisée = {all_views[(i, j)]}")
    # Génération de l'image de texture finale
    texture_size = 512
    texture_image = np.ones((texture_size, texture_size, 3), dtype=np.uint8) * 255

    for face_id, face in enumerate(mesh.faces):
        for view_id in best_views:
            h, w = texture_image.shape[:2]
            uvs_coord = uvs[face]
            img_coords = (uvs_coord * [w, h]).astype(int)
            img_coords = np.clip(img_coords, 0, [w - 1, h - 1])

            colors = []
            for vertex in face:
                print(f"Traitement du vertex {vertex} dans la vue {view_id}")
                rgb = [
                    all_views_per_channel[0].get((vertex, view_id), 0),  # rouge
                    all_views_per_channel[1].get((vertex, view_id), 0),  # vert
                    all_views_per_channel[2].get((vertex, view_id), 0)   # bleu
                ]
                if vertex == 1:
                    print(all_views_per_channel[1].get((vertex, view_id), 0))
                colors.append(rgb)

            texture_triangles(uvs_coord, colors, texture_image)

        # Affichage
        plt.figure(figsize=(8, 8))
        plt.imshow(texture_image, origin='lower')
        plt.axis('off')
        plt.title('Texture finale générée (R, G, B)')
        plt.show()

# --- main ---
run_texture_generation()


