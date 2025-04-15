import numpy as np
import trimesh
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr
import matplotlib.pyplot as plt

# 1. Création d’un mesh simple
vertices = np.array([
    [0, 0, 0],  # 0
    [1, 0, 0],  # 1
    [0, 1, 0],  # 2
    [1, 1, 0],  # 3
])
faces = np.array([
    [0, 1, 2],  # Face 0
    [1, 3, 2],  # Face 1
])
mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

# 2. Coordonnées UV associées aux sommets (mappées sur image 64x64)
uvs = {
    0: (0.0, 0.0),
    1: (1.0, 0.0),
    2: (0.0, 1.0),
    3: (1.0, 1.0),
}

# 3. Création de deux textures en dégradé de rouge
texture1 = np.zeros((64, 64, 3), dtype=np.uint8)
texture2 = np.zeros((64, 64, 3), dtype=np.uint8)

for x in range(64):
    texture1[:, x, 0] = int((x / 63) * 255)  # rouge horizontal

for y in range(64):
    texture2[y, :, 0] = int((y / 63) * 255)  # rouge vertical

# 4. Association des textures aux faces
face_to_view = {0: 0, 1: 1}
view_to_texture = {0: texture1, 1: texture2}

# 5. Détermination de Cj : sommets colorés par chaque vue
C = {0: set(faces[0]), 1: set(faces[1])}

# 6. Fonction f_red[j][i] = intensité rouge pour sommet i, vue j
f_red = {0: {}, 1: {}}
for j, vertices_set in C.items():
    tex = view_to_texture[j]
    for i in vertices_set:
        u, v = uvs[i]
        x = int(u * (tex.shape[1] - 1))
        y = int(v * (tex.shape[0] - 1))
        red = tex[y, x, 0]
        f_red[j][i] = red

# 7. Set M des paires (i, j)
M = [(i, j) for j in C for i in C[j]]
index_map = { (i, j): idx for idx, (i, j) in enumerate(M) }
n = len(index_map)

print(M)
# 8. Arêtes (adjacence)
edges = set()
for face in faces:
    for i in range(3):
        for j in range(i + 1, 3):
            u, v = face[i], face[j]
            if u != v:
                edges.add((min(u, v), max(u, v)))

# Construction de L : liste des voisins
L = {i: set() for i in range(len(vertices))}
for face in faces:
    for i in range(3):
        for j in range(i + 1, 3):
            u, v = face[i], face[j]
            if u != v:
                L[u].add(v)
                L[v].add(u)

# 9. Construction de A et b
lambda_reg = 10  # poids de la discontinuité
A = lil_matrix((1000, n))  # 1000 lignes max (temporairement)
b = np.zeros(1000)
row = 0

#deux voisins ont la même vue
for (i1, j) in M:
    for i2 in L[i1]:
        if (i2, j) in index_map:
            idx1 = index_map[(i1, j)]
            idx2 = index_map[(i2, j)]
            A[row, idx1] = 1
            A[row, idx2] = -1
            b[row] = 0
            row += 1

#un sommet a deux vues différentes
for (i,j1) in M:
    for j2 in C:
        if j1 != j2 and (i, j2) in index_map:
            idx1 = index_map[(i, j1)]
            idx2 = index_map[(i, j2)]
            A[row, idx1] = lambda_reg
            A[row, idx2] = -lambda_reg
            b[row] = lambda_reg * (f_red[j1][i] - f_red[j2][i])
            row += 1
    

# Réduction
A = A[:row]
b = b[:row]

# 10. Résolution
g_vector = lsqr(A.tocsr(), b)[0]

# 11. Affichage des résultats
print("\n--- Résultats ---")
for (i, j), idx in index_map.items():
    correction = g_vector[idx]
    print(f"g[{i},{j}] = {correction:.2f}, f_red = {f_red[j][i]}")
