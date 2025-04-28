import trimesh
import numpy as np
import xatlas
import matplotlib.pyplot as plt

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
uvs = 1 - np.round(uvs[:, [1, 0]])

print(vmapping)
print(uvs)

# if len(shared) == 2:
#     segment = trimesh.load_path(mesh.vertices[shared])
#     scene = trimesh.Scene([mesh, segment])
#     scene.show()

# creation de 3 images avec des nuances de rouge comme si c'étaient 3 vues différentes du même objet

colors = np.array([
    [255, 0, 0],     # sommet 0 - rouge vif
    [0, 255, 0],     # sommet 1 - vert vif
    [0, 0, 255],     # sommet 2 - bleu vif
    [255, 255, 0]    # sommet 3 - jaune vif
], dtype=np.float32)



def create_view_image(vertices, faces, colors, image_size=512, brightness_factor=1.0):
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    # Projection : normaliser et inverser y
    projected = vertices[:, :2].copy()
    projected[:, 0] *= (image_size - 1)                   # x
    projected[:, 1] *= (image_size - 1) 
    for face in faces:
        v0, v1, v2 = projected[face].astype(np.float32)
        c0, c1, c2 = colors[face]

        min_x = max(int(np.floor(min(v0[0], v1[0], v2[0]))), 0)
        max_x = min(int(np.ceil(max(v0[0], v1[0], v2[0]))), image_size - 1)
        min_y = max(int(np.floor(min(v0[1], v1[1], v2[1]))), 0)
        max_y = min(int(np.ceil(max(v0[1], v1[1], v2[1]))), image_size - 1)

        T = np.array([
            [v1[0] - v0[0], v2[0] - v0[0]],
            [v1[1] - v0[1], v2[1] - v0[1]]
        ])
        if np.linalg.det(T) == 0:
            continue  # triangle dégénéré

        T_inv = np.linalg.inv(T)

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                p = np.array([x - v0[0], y - v0[1]])
                u, v = T_inv @ p
                w = 1 - u - v

                if 0 <= u <= 1 and 0 <= v <= 1 and 0 <= w <= 1:
                    color = u * c1 + v * c2 + w * c0
                    color = np.clip(color * brightness_factor, 0, 255)
                    image[y, x] = color.astype(np.uint8)

    return image


Vue1 = create_view_image(vertices, faces, colors,512,0.5)
Vue2 = create_view_image(vertices, faces, colors,512,0.9)
Vue3 = create_view_image(vertices, faces, colors,512,1.8)

np.savez('seamless/images.npz', Vue1=Vue1, Vue2=Vue2, Vue3=Vue3)


images = [Vue1, Vue2, Vue3]
titles = ['Luminosité x 0.5', 'Luminosité x 1.2', 'Luminosité x 1.8']
plt.figure(figsize=(15, 5))
for i, (img, title) in enumerate(zip(images, titles), 1):
    plt.subplot(1, 3, i)
    plt.imshow(img, origin='lower')
    plt.axis('off')
    plt.title(f'Image {i} - {title}')
plt.tight_layout()
plt.show()

