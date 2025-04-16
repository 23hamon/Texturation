import trimesh
import numpy as np
from collections import defaultdict
import xatlas
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr
import matplotlib.pyplot as plt
from utils import barycentric_coordinates, bilinear_interpolate
from skimage.draw import polygon
from tqdm import tqdm
from backprojection import back_projeter

np.set_printoptions(threshold=np.inf)
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

# if len(shared) == 2:
#     segment = trimesh.load_path(mesh.vertices[shared])
#     scene = trimesh.Scene([mesh, segment])
#     scene.show()


#creation image de texture (pas utilisé encore)

red_shades = np.array([[255, 0, 0], [100, 0, 0]])

Image = np.zeros((512, 512, 3), dtype=np.uint8)

for y in range(512):
    for x in range(512):
        if y + x < 511:  
            Image[y, x] = red_shades[0]
        else:
            Image[y, x] = red_shades[1]

plt.imshow(Image)
plt.axis('off')
plt.show()

# best views

n_views = 3
n_faces = len(mesh.faces)
Wij = np.random.rand(n_faces, n_views)
best_views = np.argmin(Wij, axis=1)
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



edges = set()
for i, neighbors in L.items():
    for j in neighbors:
        if i < j:
            edges.add((i, j))

# -- fonction d'intensité 
f = np.array([
    [255, 0, 100],  # sommet 0
    [255, 0, 100],  # sommet 1
    [0, 0, 100],    
    [255, 0, 100]])   

# Systeme equations lineaitre
lambda_seam = 100
index_map = { (i, j): idx for idx, (i, j) in enumerate(M) }
n = len(M)

A = lil_matrix((100 * n, n), dtype=np.float32)
b = np.zeros(100 * n, dtype=np.float32)
row = 0

# Cas voisins
for (i1, j) in M:
    for i2 in L[i1]:
        if (i2, j) in index_map:
            A[row, index_map[(i1, j)]] = -1
            A[row, index_map[(i2, j)]] = 1
            b[row] = 0
            row += 1

# Cas vues différentes
for (i, j1) in M:
    for (i2, j2) in M:
        if i == i2 and j1 != j2:
            idx1 = index_map[(i, j1)]
            idx2 = index_map[(i, j2)]
            A[row, idx1] = -lambda_seam
            A[row, idx2] = lambda_seam
            b[row] = (f[i, j1] - f[i, j2]) * lambda_seam
            row += 1

A = A[:row]
b = b[:row]
g_vector = lsqr(A, b)[0]

print("g_vector:")
print(g_vector)

#mise à jour de la fonction d'intensité
for (i, j) in M:
    f[i, j] = f[i,j] + g_vector[index_map[(i, j)]]

Image_final = np.zeros((512, 512, 3), dtype=np.uint8)

for face_idx, face in enumerate(mesh.faces):
    view = best_views[face_idx]
    verts = face

    uv_coords = (uvs[verts] * 511).astype(np.int32)
    uv_coords = (uvs[verts] * 511).astype(np.int32)
    u = 511 - uv_coords[:, 0]
    v = uv_coords[:, 1]  # inversion verticale

    rr, cc = polygon(v, u, Image_final.shape[:2])

    try:
        red_values = [f[vi, view] for vi in verts]
    except IndexError:
        continue

    mean_red = np.mean(red_values).astype(np.uint8)

    Image_final[rr, cc, 0] = mean_red  # canal rouge
    # vert et bleu laissés à 0

plt.imshow(Image_final)
plt.axis('off')
plt.title("Image simplifiée depuis f")
plt.show()
