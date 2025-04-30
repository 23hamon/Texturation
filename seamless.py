import trimesh 
import numpy as np
from collections import defaultdict
import xatlas
mesh = trimesh.load_mesh('fichiers_ply/mesh_cailloux_low.ply')
np.set_printoptions(threshold=np.inf)
from collections import defaultdict
np.random.seed(42)
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr
# ensemble des inputs nécessaires pour le seamless: 

lambda_seam = 100

#les textures

texture_map = np.load("fichiers_intermediaires/texture_map.npy")

#la liste des best views
Wij = np.load("fichiers_intermediaires/Wij.npy")
best_views = np.argmin(Wij, axis=1)


#Liste L des sommets adjacents dans le mesh

def adjacent_vertices(mesh):
    """
    Calcule les sommets adjacents pour chaque sommet du maillage.
    Input:
    - mesh: maillage 3D
    Output:
    - adj: dictionnaire {i: [i1, i2, ...]} où i est un sommet et [i1, i2, ...] sont ses sommets adjacents
    """
    L = defaultdict(list)
    for face in mesh.faces:
        for i in range(3):
            for j in range(i + 1, 3):
                L[face[i]].append(face[j]) if face[j] not in L[face[i]] else None
                L[face[j]].append(face[i]) if face[i] not in L[face[j]] else None
    return L

L = adjacent_vertices(mesh)


#Cj : un set de l'ensemble des sommets ayant été colorés par la vue j

C = defaultdict(set)
for face, view in enumerate(best_views):
    for vertex in mesh.faces[face]:
        C[view].add(vertex)

#(C[0])  # les sommets colorés par la vue 0

#M est un ensemble (i, j) où i est un sommet et j est la vue qui l'a coloré
M = [(i, j) for j, vertices in C.items() for i in vertices]

#ensemble des arêtes du maillage
edges = set()
for i, neighbors in L.items():
    for j in neighbors:
        if i < j: # éviter les doublons
            edges.add((i, j))

#mapping
vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)


#création fonction d'intensité rouge

n_vertices = len(mesh.vertices)
n_views = Wij.shape[1]  # ou 52

def intensity(texture_map, n_vertices, n_views, color = 0):
    f = np.full((n_vertices, n_views), np.nan)  # remplie par défaut avec NaN
    for j, vertices in C.items():
        for i in vertices:
            u, v = uvs[i]
            pixel = texture_map[int(v * 511), int(u * 511)]
            intensity = pixel[color]
            f[i, j] = intensity
    return f

print(intensity(texture_map, n_vertices, n_views, color = 0))


#creation du système d'équations
index_map = { (i, j): idx for idx, (i, j) in enumerate(M) }
n = len(M)

A = lil_matrix((2 * n, n), dtype=np.float32)
b = np.zeros(n, dtype=np.float32)
row = 0

#cas i1, i2 voisins 

for (i1, j) in M:
    for i2 in L[i1]:
        if (i2, j) in index_map: #si le voisin de i1 est coloré par la même vue j
            idx = index_map[(i2, j)] #on récupère l'indice de i2
            A[row, index_map[(i1, j)]] = -1
            A[row, index_map[(i2, j)]] = 1
            b[row] = 0
            row += 1

#cas un sommet i est partagé entre deux vues j1 et j2

for (i, j1) in M:
    for (i, j2) in M:
        if j1 != j2: #si le sommet est partagé entre deux vues
            idx1 = index_map[(i, j1)]
            idx2 = index_map[(i, j2)]
            A[row, idx1] = -1 * lambda_seam
            A[row, idx2] = 1 * lambda_seam
            b[row] = (intensity(texture_map, n_vertices, n_views)[i, j1] - intensity(texture_map, n_vertices, n_views)[i, j2]) * lambda_seam
            row += 1

A = A[:row]
b = b[:row]

g_vector = lsqr(A.tocsr(), b)[0]
