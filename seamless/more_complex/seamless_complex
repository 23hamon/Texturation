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

data = np.load('seamless/npy/images.npz')
Vue1, Vue2, Vue3 = data['Vue1'], data['Vue2'], data['Vue3']

texture_map = np.load('seamless/npy/text_map.npy')
indices = np.load('seamless/npy/indices.npy')
uvs = np.load('seamless/npy/uvs.npy')
vmapping = np.load('seamless/npy/mapping.npy')
mesh = trimesh.load('seamless/npy/mesh.obj')
views = {0: Vue1, 1: Vue2, 2: Vue3}
n_views = len(views)
n_faces = len(mesh.faces)
best_views = [0, 2]

# --- Voisinage ---
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

# --- Sommets colorés par vue ---
C = defaultdict(set)
for face, view in enumerate(best_views):
    for vertex in mesh.faces[face]:
        C[view].add(vertex)

M = [(i, j) for j, vertices in C.items() for i in vertices]
index_map = { (i, j): idx for idx, (i, j) in enumerate(M)}
n = len(M)

# --- Fonctions d'intensité ---
def intensity(view_id, channel_id):
    image = views[view_id]
    h, w = image.shape[:2]
    f = np.zeros((h, w))
    for face_id, face in enumerate(mesh.faces):
        uvs_coords = uvs[face]
        c = (uvs_coords[:, 0] * w).astype(int)
        r = (uvs_coords[:, 1] * h).astype(int)
        rr, cc = polygon(r, c)
        for y, x in zip(rr, cc):
            f[y, x] = image[int(y), int(x), channel_id]
    return f

def get_intensity_vertice(view_id, vertex_id, channel_id):
    f = intensity(view_id, channel_id)
    h, w = views[view_id].shape[:2]
    uv_vertex = uvs[vertex_id]
    x = (uv_vertex[0] * w).astype(int)
    y = (uv_vertex[1] * h).astype(int)
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)
    return f[y, x]

def intensity_all_views(views, M, channel_id):
    all_views_intensities = {}
    for view_id in range(len(views)):
        for vertex_id in range(len(mesh.vertices)):
            if (vertex_id, view_id) in M:
                all_views_intensities[(vertex_id, view_id)] = get_intensity_vertice(view_id, vertex_id, channel_id)
            else:
                all_views_intensities[(vertex_id, view_id)] = 0
    return all_views_intensities

# --- Construction de la fonction g ---
def build_g_function(intensity_all_views, L, M, index_map, lambda_seam=100):
    def g(x):
        residuals_smoothness = []
        residuals_same_view = []

        for (i1, j) in M:
            for i2 in L[i1]:
                if (i2, j) in index_map:
                    idx1 = index_map[(i1, j)]
                    idx2 = index_map[(i2, j)]
                    residuals_smoothness.append(x[idx1] - x[idx2])

        for (i, j1) in M:
            for (i2, j2) in M:
                if i == i2 and j1 != j2:
                    intensity_i_j1 = intensity_all_views[(i, j1)]
                    intensity_i_j2 = intensity_all_views[(i, j2)]
                    idx_j1 = index_map[(i, j1)]
                    idx_j2 = index_map[(i2, j2)]
                    residuals_same_view.append(
                        x[idx_j1] - x[idx_j2] - (intensity_i_j2 - intensity_i_j1)
                    )

        cost = (
            np.sum(np.square(residuals_smoothness)) +
            lambda_seam * np.sum(np.square(residuals_same_view))
        )
        return np.array(cost, dtype=np.float32)
    return g

# --- Génération d'une texture complète ---

def generate_texture_image(channels_to_display=[0, 1, 2]):
    texture_size = 512
    texture_image = np.ones((texture_size, texture_size, 3), dtype=np.uint8) * 255
    optimized_channels = {}

    channel_names = {0: 'red', 1: 'green', 2: 'blue'}

    for channel_id in [0, 1, 2]:
        print(f"\n--- Traitement du canal {channel_names[channel_id].upper()} ---")
        all_views = intensity_all_views(views, M, channel_id)
        g = build_g_function(all_views, L, M, index_map)
        x0 = np.zeros(n)
        res = least_squares(g, x0, jac='2-point')
        optimal_x = res.x

        print(f"Résultat de l'optimisation ({channel_names[channel_id]}):", optimal_x)

        for (i, j) in all_views:
            idx = index_map.get((i, j), None)
            if idx is not None and (i, j) in M:
                all_views[(i, j)] += optimal_x[idx]
                all_views[(i, j)] = np.clip(all_views[(i, j)], 0, 255)

        optimized_channels[channel_id] = all_views

        if channel_id in channels_to_display:

            # Affichage canal choisi
            channel_image = np.ones((texture_size, texture_size), dtype=np.uint8) * 255
            channel_image_rgb = np.stack([channel_image]*3, axis=-1)

            for face_id, face in enumerate(mesh.faces):
                view_id = best_views[face_id]
                uvs_coord = uvs[face]
                h, w = channel_image.shape
                img_coords = (uvs_coord * [w, h]).astype(int)
                img_coords = np.clip(img_coords, 0, [w - 1, h - 1])

                colors = []
                for vertex in face:
                    val = all_views.get((vertex, view_id), 0)
                    rgb_val = [0, 0, 0]
                    rgb_val[channel_id] = val
                    colors.append(rgb_val)

                texture_triangles(uvs_coord, colors, channel_image_rgb)

            plt.figure(figsize=(5, 5))
            plt.imshow(channel_image_rgb)
            plt.axis('off')
            plt.title(f'Texture optimisée - Canal {channel_names[channel_id].upper()}')
            plt.show()

    # Génération de la texture finale RGB
    for face_id, face in enumerate(mesh.faces):
        view_id = best_views[face_id]
        uvs_coord = uvs[face]
        h, w = texture_image.shape[:2]
        img_coords = (uvs_coord * [w, h]).astype(int)
        img_coords = np.clip(img_coords, 0, [w - 1, h - 1])

        colors = []
        for vertex in face:
            rgb = [
                optimized_channels[0].get((vertex, view_id), 0),
                optimized_channels[1].get((vertex, view_id), 0),
                optimized_channels[2].get((vertex, view_id), 0)
            ]
            colors.append(rgb)

        texture_triangles(uvs_coord, colors, texture_image)

    plt.figure(figsize=(6, 6))
    plt.imshow(texture_image)
    plt.imsave('seamless/png/text_final.png', texture_image)
    plt.axis('off')
    plt.title('Texture finale (R+G+B)')
    plt.show()

# --- main ---
generate_texture_image(channels_to_display=[])