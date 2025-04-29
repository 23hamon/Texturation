import trimesh
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import least_squares
from skimage.draw import polygon
np.random.seed(42)
from example_texture_map import texture_triangles, barycentric


#imports
data = np.load('seamless/images.npz')
Vue1 = data['Vue1']
Vue2 = data['Vue2']
Vue3 = data['Vue3']
texture_map = np.load('seamless/text_map.npy')
indices = np.load('seamless/indices.npy')
uvs = np.load('seamless/uvs.npy')
uvs = 1 - (uvs[:, [1, 0]])  # Inversion
vmapping = np.load('seamless/mapping.npy')
mesh = trimesh.load('seamless/mesh.obj')

# print("\n--- UVs par sommet ---")
# for i, uv in enumerate(uvs):
#     print(f"Sommet {i} : UV = {uv}")

# image = Vue2
# h, w = image.shape[:2]
# plt.figure(figsize=(8, 8))
# plt.imshow(image, origin = 'lower')  # pour être cohérent avec les UVs
# plt.title("Vue 2 avec triangles et numéros des sommets")
# plt.axis('off')
# plt.show()



#Best views
views = {0: Vue1, 1: Vue2, 2: Vue3} #on a donc trois vues différentes 
n_views = len(views)
n_faces = len(mesh.faces)
Wij = np.random.rand(n_faces, n_views)
#best_views = np.argmin(Wij, axis=1) #et on récupère les meilleures vues pour chaque face , ici vue1 pour la face 0, et vue3 pour la face 1
#ici, j'impose best_views = [0,2]
best_views = [0, 2]

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
            x = int(uv[0] * w)
            y = int(uv[1] * h)
            if 0 <= x <= w and 0 <= y <= h:
                return f[y, x]
    return 0



# on réupére des intensités des sommets pour chaque vue
def intensity_all_views(views, intensities, M):
    """
    Récupère les intensités des sommets pour chaque vue pour un canal de couleur donné.
    - color_channel : Le canal de couleur à récupérer (0 = rouge, 1 = vert, 2 = bleu)
    output :
    - Un dictionnaire contenant les intensités des sommets pour chaque vue et canal de couleur.
    """
    all_views_intensities = {}
    for view_id in range(len(views)):
        for vertex_id in range(n_vertices):
            if (vertex_id, view_id) in M:
                all_views_intensities[(vertex_id, view_id)] = get_intensity_vertice(intensities[view_id], view_id, vertex_id)
            else:
                all_views_intensities[(vertex_id, view_id)] = 0
    return all_views_intensities



#CALCUL DE G

lambda_seam = 100
index_map = { (i, j): idx for idx, (i, j) in enumerate(M) }
n = len(M)


# def g(x):

#     residuals_smoothness = []
#     residuals_same_view = []
#     lambda_seam = 100

#     # entre idx1 et idx2, si i1 et 12 voisins : g[idx1] - g[idx2] = 0
#     for (i1, j) in tqdm(M):
#         for i2 in L[i1]:
#             if (i2, j) in index_map:
#                 idx1 = index_map[(i1, j)]
#                 idx2 = index_map[(i2, j)]
#                 g_idx1 = x[idx1] 
#                 g_idx2 = x[idx2]
#                 residuals_smoothness.append(g_idx1 - g_idx2) 

#     # Contraintes entre idx_j1 et idx_j2 : g[idx_j1] - g[idx_j2] = f[idx_j2] - f[idx_j1]
#     for (i, j1) in tqdm(M):
#         for (i2, j2) in M:
#             if i == i2 and j1 != j2:
#                 red_i_j1 = intensity_all_views[(i, j1)]
#                 red_i_j2 = intensity_all_views[(i, j2)]
#                 idx_j1 = index_map[(i, j1)]
#                 idx_j2 = index_map[(i2, j2)]
#                 g_j1 = x[idx_j1]
#                 g_j2 = x[idx_j2]
#                 residuals_same_view.append(g_j1 - g_j2 - (red_i_j2 - red_i_j1))

#     cost_smoothness = np.sum(np.array(residuals_smoothness)**2)
#     cost_same_view = np.sum(np.array(residuals_same_view)**2)
#     total_cost = cost_smoothness + lambda_seam * (cost_same_view)

#     return np.array(total_cost, dtype=np.float32)


# x0 = np.zeros(n) 

# res = least_squares(g, x0, jac='2-point')

# optimal_x = res.x
# np.save('seamless/optimal_x.npy', optimal_x)

optimal_x = np.load('seamless/optimal_x.npy')


intensities_red = {view_id: intensity(view_id, 0) for view_id in range(len(views))}
intensities_green = {view_id: intensity(view_id, 1) for view_id in range(len(views))}
intensities_blue = {view_id: intensity(view_id, 2) for view_id in range(len(views))}

red_all_views = intensity_all_views(views, intensities_red, M)
green_all_views = intensity_all_views(views, intensities_green, M)
blue_all_views = intensity_all_views(views, intensities_blue, M)



# 3. Mise à jour des intensités avec optimal_x
for (i, j) in red_all_views:
    idx = index_map.get((i, j), None)
    if idx is not None:
        red_all_views[(i, j)] += optimal_x[idx]

#à présent, on crée une image de text à partir de toutes les vues, en interpolant pour colorer tous les pixels. 
texture_size = 512
texture_image = np.ones((texture_size, texture_size, 3), dtype=np.uint8) * 255
def color(view, vertex_id, color_channel=0):
    """
    Récupère la couleur d'un sommet pour une vue donnée et un canal de couleur spécifique.
    """
    if (vertex_id, view) in red_all_views:
        return red_all_views[(vertex_id, view)]
    else:
        return 0
for face_id, face in enumerate(mesh.faces):
    for view_id, view in enumerate(views):
        h, w = texture_image.shape[:2]
        uvs_coord = uvs[face]
        img_coords = (uvs_coord * [w, h]).astype(int)
        img_coords = np.clip(img_coords, 0, [w - 1, h - 1])
        if (vertex, view_id) in red_all_views:
            colors = [[red_all_views[vertex, view_id], 0, 0] for vertex in face]
        else:
            print(f"Clé ({view_id}, {vertex}) non trouvée dans red_all_views")
            colors = [[0, 0, 0] for vertex in face] 
        texture_triangles(uvs_coord, colors, texture_image)
        print("face", face_id, "view", view_id, "colors", colors, "OK")


# Affichage
plt.imshow(texture_image, origin='lower', cmap='Reds')
plt.axis('off')
plt.title('Texture générée à partir des vues avec vertices')
plt.show()

