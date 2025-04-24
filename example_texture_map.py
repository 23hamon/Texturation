import trimesh
import numpy as np
from collections import defaultdict
import xatlas
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

#CREATION D UNE IMAGE DE TEXTURE
#np.set_printoptions(threshold=np.inf)
np.random.seed(42)
data = np.load('fichiers_intermediaires/images.npz')
Vue1 = data['Vue1']
Vue2 = data['Vue2']
Vue3 = data['Vue3']


#IMAGE DE TEXTURE
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

views = {0: Vue1, 1: Vue2, 2: Vue3}

mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
uvs[:, 1] = 1 - uvs[:, 1] #le mapping est inversé
n_views = 3
n_faces = len(mesh.faces)
Wij = np.random.rand(n_faces, n_views)
best_views = np.argmin(Wij, axis=1)
texture_size = 512 
texture_image = np.ones((texture_size, texture_size, 3), dtype=np.uint8) * 255


# Fonction pour dessiner un triangle avec interpolation des couleurs dans la texture
def texture_triangles(uv_coords, colors, texture):
    uv_pixels = (uv_coords * texture_size).astype(np.int32)
    mask = np.zeros((texture_size, texture_size), dtype=np.uint8)
    pts = np.array([uv_pixels], dtype=np.int32)

    cv2.fillPoly(mask, pts, 1)
    ys, xs = np.where(mask == 1)

    for x, y in zip(xs, ys):
        A = uv_pixels[0]
        B = uv_pixels[1]
        C = uv_pixels[2]
        P = np.array([x, y])

        def barycentric(A, B, C, P):
            s = np.vstack([B - A, C - A]).T
            try:
                inv_s = np.linalg.inv(s)
            except np.linalg.LinAlgError:
                return np.array([1/3, 1/3, 1/3])
            v = P - A
            uv = inv_s @ v
            u, v = uv
            w = 1 - u - v
            return np.array([w, u, v])

        bary = barycentric(A, B, C, P)
        bary = np.clip(bary, 0, 1)
        color = np.dot(bary, colors)
        texture[y, x] = color.astype(np.uint8)

# Construction de la texture à partir des vues
for face_id, face in enumerate(mesh.faces):
    view_id = best_views[face_id]
    image = views[view_id]

    h, w = image.shape[:2]
    uv_coords = uvs[face]
    img_coords = (uv_coords * [w, h]).astype(int)
    img_coords = np.clip(img_coords, 0, [w - 1, h - 1])

    # Couleur des trois sommets projetés dans la meilleure vue
    colors = [image[y, x] for x, y in img_coords]
    print(colors)

    # Rasterisation du triangle dans la texture finale
    texture_triangles(uv_coords, colors, texture_image)

# Affichage
plt.imshow(texture_image)
plt.axis('off')
plt.title('Texture générée à partir des vues 1 et 3')
plt.show()

print(texture_image)
np.save('fichiers_intermediaires/ex_texture_map.npy', texture_image)


