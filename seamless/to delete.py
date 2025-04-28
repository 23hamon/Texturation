import matplotlib.pyplot as plt
import numpy as np
import xatlas
import trimesh

# Mesh carré divisé en deux triangles
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

# Image de fond (vue 2)
import numpy as np
data = np.load('fichiers_intermediaires/images.npz')
image = data['Vue2']  # shape (H, W, 3)
plt.imshow(image)
# Dimensions
h, w = image.shape[:2]

# Affichage
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(image, extent=(0, 1, 0, 1), origin='lower')  # origin 'lower' = UV(0,0) en bas à gauche

# Affichage des triangles dans l'espace UV
for face in mesh.faces:
    uv = uvs[face]
    tri = plt.Polygon(uv, edgecolor='black', fill=False, linewidth=1.5)
    ax.add_patch(tri)
    for i, (u, v) in zip(face, uv):
        ax.text(u, v, str(i), color='red', ha='center', va='center', fontsize=12)

ax.set_title("UVs + Image Vue2 en fond")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.show()
