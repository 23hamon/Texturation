import trimesh
import numpy as np
import xatlas
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.optimize import least_squares
from skimage.draw import polygon
from backprojection import back_projeter
from tqdm import tqdm
from PIL import Image
from scipy.sparse import lil_matrix
import cv2
import json
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # répertoire du script
PATH_TENSORS = os.path.join(ROOT_DIR, "tensors")
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(ROOT_DIR, "absolute_transforms_luca.json")



def get_image_data(image_id=26):
    """
    Renvoie (rot, t) ou rot est la matrice de rotation de la camera, et t son vecteur de translation
    """
    # position de l'image
    with open(JSON_PATH) as f:
        data = json.load(f)
        image_id=str(image_id)
        r = np.array(data["0"][image_id][0], dtype=np.float64)
        t = np.array(data["0"][image_id][1], dtype=np.float64)
        rot, _ = cv2.Rodrigues(r)
    return (rot, t)

def barycentric_coordinates(p, A, B, C):
    v0, v1, v2 = B - A, C - A, p - A #chaque cote du triange
    d00, d01, d11 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v1, v1)
    d20, d21 = np.dot(v2, v0), np.dot(v2, v1)

    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
   
    return v, w, u


def bilinear_interpolate(image, x, y):
    x0 = int(np.floor(x))
    x1 = min(x0 + 1, image.shape[1] - 1)
    y0 = int(np.floor(y))
    y1 = min(y0 + 1, image.shape[0] - 1)
    wx = x - x0
    wy = y - y0
    top = (1 - wx) * image[y0, x0] + wx * image[y0, x1]
    bottom = (1 - wx) * image[y1, x0] + wx * image[y1, x1]
    return ((1 - wy) * top + wy * bottom).astype(np.uint)


# --- Chargement des images ---
cam = "l"
N = 52
image_path = f"downsampled/scene_{cam}_"

Vjyxc = [
    cv2.cvtColor(cv2.imread(image_path + f"{j:04d}.jpeg"), cv2.COLOR_BGR2RGB)
    for j in range(1, N + 1)
]
Vjyxc = np.stack(Vjyxc, axis=0)  # shape (N, h, w, 3)
h, w = Vjyxc[0].shape[:2]

# Données des images
views = {}
for i in range(Vjyxc.shape[0]):
    views[i] = Vjyxc[i]

M_final = np.load(os.path.join(PATH_TENSORS, "M_final.npy"))


# --- Données du mesh ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # Répertoire où se trouve le script
PLY_PATH = os.path.join(ROOT_DIR, "ply", "mesh_cailloux_luca_CLEAN.ply")

mesh = trimesh.load(PLY_PATH)

vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
faces = mesh.faces
vertices = mesh.vertices

# --- Fonction de correspondance ancienne/nouvelle face (optionnel) ---
def find_new_face_id():
    ancienne_vers_nouvelle = defaultdict(list)
    original_face_sets = {i: set(face) for i, face in enumerate(faces)}

    for new_face_id, face_uv in enumerate(indices):
        original_vertices = {vmapping[i] for i in face_uv}
        for old_face_id, vertex_set in original_face_sets.items():
            if original_vertices == vertex_set:
                ancienne_vers_nouvelle[old_face_id].append(new_face_id)
                break
    return ancienne_vers_nouvelle

# --- Calcul du voisinage ---
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

# --- Couple face - vue qui la colore ---
def couple_vertex_view():
    M = set()
    for view in views:
        for face in range(len(indices)):
            if M_final[face] == view:
                for vertex in indices[face]:
                    M.add((vertex, view))
    return list(M)

# --- Fonctions d'intensité ---
def from_vertice_uv_to_pixel(vertex_uv_id, view_id):
    rot, t = get_image_data(view_id + 1)
    vertex_mesh = mesh.vertices[vmapping[vertex_uv_id]]
    pixel = back_projeter(vertex_mesh, rot, t, "l")[0]
    return pixel[0], pixel[1]

def from_face_to_all_pixel(face_id, view_id, image_shape):
    rot, t = get_image_data(view_id + 1)
    face = indices[face_id]
    uv = uvs[face]
    r, c = uv[:, 1] * image_shape[0], uv[:, 0] * image_shape[1]
    rr, cc = polygon(r, c, image_shape[:2])
    pixels = []
    points_3d = []
    vertices = [mesh.vertices[vmapping[face[i]]] for i in range(3)]

    A = np.array([c[0], r[0]])
    B = np.array([c[1], r[1]])
    C = np.array([c[2], r[2]])
    for i in range(len(rr)):
        p = np.array([cc[i], rr[i]])
        u, v, w = barycentric_coordinates(p, A, B, C)
        points_3d.append((1 - u - v) * vertices[0] + u * vertices[1] + v * vertices[2])

    for point_3d in points_3d:
        pixel = back_projeter(point_3d, rot, t, "l")[0]
        if pixel is not None:
            X, Y = int(pixel[0]), int(pixel[1])
            pixels.append((X, Y))

    return pixels

def f_all_summit(view_id, channel_id=0):
    image = Vjyxc[view_id]
    h, w = image.shape[:2]
    intensities = np.zeros((h, w))
    for vertex_id in range(len(vmapping)):
        x, y = from_vertice_uv_to_pixel(vertex_id, view_id)
        x = int(x)
        y = int(y)
        if 0 <= y < h and 0 <= x < w:
            intensities[y, x] = image[y, x, channel_id]
    return intensities

def get_intensity_vertice(vertex_id, view_id, channel_id):
    if (vertex_id, view_id) in M:
        x, y = from_vertice_uv_to_pixel(vertex_id, view_id)
        x = int(x)
        y = int(y)
        return Vjyxc[view_id, y, x, channel_id]
    return 0

def intensity_all_views(views, channel_id, vertex_in_mesh):
    all_views_intensities = {}
    for view_id in tqdm(range(len(views)), desc="Calcul de intensity_all_views"):
        rot, t = get_image_data(view_id + 1)
        for (vertex_id, vertex_mesh) in vertex_in_mesh[view_id]:
            pixel = back_projeter(vertex_mesh, rot, t, "l")[0]
            x = int(pixel[0])
            y = int(pixel[1])
            all_views_intensities[(vertex_id, view_id)] = Vjyxc[view_id, y, x, channel_id]
    return all_views_intensities

def build_jacobian_sparsity(n_vars, smooth_idx_1, smooth_idx_2, same_idx_1, same_idx_2):
    n_smooth = len(smooth_idx_1)
    n_same = len(same_idx_1)
    n_residuals = n_smooth + n_same

    J = lil_matrix((n_residuals, n_vars), dtype=bool)

    for row, (i1, i2) in enumerate(zip(smooth_idx_1, smooth_idx_2)):
        J[row, i1] = True
        J[row, i2] = True

    for k, (i1, i2) in enumerate(zip(same_idx_1, same_idx_2)):
        row = n_smooth + k
        J[row, i1] = True
        J[row, i2] = True

    return J

def build_g_function(intensity_all, L, M, index_map, lambda_seam=1000):
    print("Préparation des résidus")

    smooth_idx_1, smooth_idx_2 = [], []
    same_idx_1, same_idx_2, delta_I = [], [], []

    for (i1, j) in M:
        for i2 in L.get(i1, []):
            if (i2, j) in index_map:
                smooth_idx_1.append(index_map[(i1, j)])
                smooth_idx_2.append(index_map[(i2, j)])

    for (i, j1) in M:
        for j2 in range(len(views)):
            if j1 != j2 and (i, j2) in index_map:
                idx1 = index_map[(i, j1)]
                idx2 = index_map[(i, j2)]
                I1 = int(intensity_all[(i, j1)])
                I2 = int(intensity_all[(i, j2)])
                same_idx_1.append(idx1)
                same_idx_2.append(idx2)
                # print(f"Intensité de la vue {j1} : {I1}, Intensité de la vue {j2} : {I2}")
                # print(f"I2 - I1 : {I2 - I1}")
                delta_I.append((I2 - I1))

    smooth_idx_1 = np.array(smooth_idx_1)
    smooth_idx_2 = np.array(smooth_idx_2)
    same_idx_1 = np.array(same_idx_1)
    same_idx_2 = np.array(same_idx_2)
    delta_I = np.array(delta_I)

    jac_sparsity = build_jacobian_sparsity(
        n_vars=len(M),
        smooth_idx_1=smooth_idx_1,
        smooth_idx_2=smooth_idx_2,
        same_idx_1=same_idx_1,
        same_idx_2=same_idx_2
    )

    def g(x):
        res_smooth = x[smooth_idx_1] - x[smooth_idx_2]
        res_seam = (x[same_idx_1] - x[same_idx_2]) - delta_I
        lambda_seam_sqrt = lambda_seam ** 0.5
        return np.concatenate([res_smooth, lambda_seam_sqrt * res_seam])

    print("Lancement de l'optimisation")
    x0 = np.zeros(len(M))
    res = least_squares(g, x0, method='trf', jac_sparsity=jac_sparsity, verbose = 2)

    print("Optimisation terminée.")
    assert len(res.x) == len(M), f"Incohérence : {len(res.x)} valeurs optimisées, {len(M)} couples attendus."
    return res.x


# -- main

M = couple_vertex_view()
index_map = { (i, j): idx for idx, (i, j) in enumerate(M)}

M_set = set(M) #accelère la recherche

vertex_to_analyse = {}
vertex_in_mesh = {}

mapping_mesh = mesh.vertices[vmapping]

for view_id in tqdm(range(len(views))):
    vertex_to_analyse[view_id] = [vertex_id for vertex_id in range(len(vmapping)) if (vertex_id, view_id) in M_set]
    vertex_in_mesh[view_id] = []
    for vertex_id in vertex_to_analyse[view_id]:
        if ((vertex_id, mapping_mesh[vertex_id])) not in vertex_in_mesh[view_id]:
            vertex_in_mesh[view_id].append((vertex_id, mapping_mesh[vertex_id]))


# red_intensity_all_views = intensity_all_views(views, 0, vertex_in_mesh)
# blue_intensity_all_views = intensity_all_views(views, 2, vertex_in_mesh)
# green_intensity_all_views = intensity_all_views(views,1, vertex_in_mesh)


# g_red = build_g_function(red_intensity_all_views, L, M, index_map)
# g_blue = build_g_function(blue_intensity_all_views, L, M, index_map)
# g_green = build_g_function(green_intensity_all_views, L, M, index_map)

g = np.zeros((len(M), 3), dtype=np.float32)

for i in range(len(M)):
    g[i, 0] = int(g_red[i])
    g[i, 1] = int(g_green[i])
    g[i, 2] = int(g_blue[i])

# g[i, 0] = np.clip(np.round(g_red[i]), 0, 255)
# g[i, 1] = np.clip(np.round(g_green[i]), 0, 255)
# g[i, 2] = np.clip(np.round(g_blue[i]), 0, 255)

#print(g)

map_texture = np.ones((512, 512, 3), dtype=np.uint) * 255
#final_texture = np.ones((512, 512, 3), dtype=np.uint) * 255

colors_seamless = {}  # À définir en dehors de la boucle sur les faces
colors_texture = {}
final_colors = {}

for face_id in tqdm(range(len(indices)), desc="Traitement des faces"):
    sommets_uv = indices[face_id]
    view_id = M_final[face_id]
    image = Vjyxc[view_id]
    rot, t = get_image_data(view_id + 1)
    face = indices[face_id]
    uv = uvs[face]

    colors = []

    vertices = [
        mesh.vertices[vmapping[face[0]]],
        mesh.vertices[vmapping[face[1]]],
        mesh.vertices[vmapping[face[2]]]
    ]


    # for vertex_id in sommets_uv:
    #     vertex_intensity = g[index_map[(vertex_id, view_id)]]
    #     colors.append(vertex_intensity)

    # colors = np.array(colors)

    uv = uvs[sommets_uv]

    r = uv[:, 1] * map_texture.shape[0]  # Lignes
    c = uv[:, 0] * map_texture.shape[1]  # Colonnes

    rr, cc = polygon(r, c, map_texture.shape[:2])

    A = np.array([c[0], r[0]])
    B = np.array([c[1], r[1]])
    C = np.array([c[2], r[2]])

    for x, y in zip(cc, rr):
        p = np.array([x, y])
       
        u, v, w = barycentric_coordinates(p, A, B, C)
        # color = (1 - u - v) * colors[0] + u * colors[1] + v * colors[2]
       
        # Utilise y, x (pas rr[i], cc[i])
        # colors_seamless[(y, x)] = tuple(np.round(color).astype(int))

        pt_3d = (1 - u - v) * vertices[0] + u * vertices[1] + v * vertices[2]        
        pixel = back_projeter(pt_3d, rot, t, "l")[0]
       
        if pixel is not None:
            px, py = int(pixel[0]), int(pixel[1])
        else:
            continue
       
        if 0 <= px < image.shape[1] and 0 <= py < image.shape[0]:
            intensity = image[py, px]
        else:
            continue

        colors_texture[(y, x)] = intensity
        map_texture[y, x] = intensity

        final_colors[(y, x)] = intensity #+ color
        #final_texture[y, x] = final_colors[(y, x)]

        # Enregistrement dans le dictionnaire
np.save(os.path.join(PATH_TENSORS, "final_colors.npy"), final_colors)
np.save(os.path.join(PATH_TENSORS, "map_texture.npy"), map_texture)
np.save(os.path.join(PATH_TENSORS, "colors_seamless.npy"), colors_seamless)




# Affichage de l'image générée
fig, axes = plt.subplots(1, 2)

axes[0].imshow(map_texture)
axes[0].set_title("Texture originale")
axes[0].axis('off')

axes[1].imshow(final_texture)
axes[1].set_title("Texture seamless")
axes[1].axis('off')

plt.tight_layout()
plt.show()

