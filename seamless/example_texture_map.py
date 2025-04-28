import trimesh
import numpy as np
import xatlas
import matplotlib.pyplot as plt
import cv2
np.random.seed(42)

# Chargement des données
data = np.load('seamless/images.npz')
Vue1 = data['Vue1']
Vue2 = data['Vue2']
Vue3 = data['Vue3']
views = {0: Vue1, 1: Vue2, 2: Vue3}

indices = np.load('seamless/indices.npy')
uvs = np.load('seamless/uvs.npy')
vmapping = np.load('seamless/mapping.npy')
mesh = trimesh.load('seamless/mesh.obj')

vertices = mesh.vertices
faces = mesh.faces

n_views = 3
n_faces = len(faces)
Wij = np.random.rand(n_faces, n_views)
best_views = np.argmin(Wij, axis=1)
print(best_views)

# Initialisation de la texture blanche
texture_size = 512
texture_image = np.ones((texture_size, texture_size, 3), dtype=np.uint8) * 255

# Fonction de barycentrique
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

# Fonction de texturage
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
        bary = barycentric(A, B, C, P)
        bary = np.clip(bary, 0, 1)
        color = np.dot(bary, colors)
        texture[y, x] = color.astype(np.uint8)

# Fonction pour dessiner les vertices sur la texture
def draw_vertices_on_texture(uvs, texture, vertices, texture_size):
    for i, uv in enumerate(uvs):
        uv_img = (uv * texture_size).astype(np.int32)
        cv2.circle(texture, tuple(uv_img), 5, (0, 0, 255), -1)  # Cercle rouge
        # Optionnel : ajouter le texte avec le numéro du sommet
        cv2.putText(texture, str(i), tuple(uv_img), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

# Boucle principale
for face_id, face in enumerate(faces):
    view_id = best_views[face_id]
    image = views[view_id]
    h, w = image.shape[:2]

    uv_coords = uvs[face]

    uv_coords_img = uv_coords.copy()
    uv_coords_img[:, 1] = 1.0 - uv_coords_img[:, 1]  # Inversion Y

    img_coords = (uv_coords_img * [w, h]).astype(int)
    img_coords = np.clip(img_coords, 0, [w - 1, h - 1])

    # Couleurs des sommets projetés dans l’image
    colors = [image[y, x] for x, y in img_coords]

    # Remplissage de la texture
    texture_triangles(uv_coords, colors, texture_image)

# Dessiner les vertices sur la texture
draw_vertices_on_texture(uvs, texture_image, vertices, texture_size)

# Affichage
plt.imshow(texture_image)
plt.axis('off')
plt.title('Texture générée à partir des vues avec vertices')
plt.show()

# Sauvegarde
np.save('fichiers_intermediaires/ex_texture_map_with_vertices.npy', texture_image)
