import trimesh
import xatlas
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


mesh = trimesh.load_mesh('fichiers_ply/mesh_cailloux_min.ply')
vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)


# print(vmapping)  # indices du sommet d'origine de nouveau sommet
# print(indices)   # indices des triangles dans le mapping uv
print(uvs)       # (u,v) des nouveaux sommets

# Créer le maillage avec les coordonnées UV
new_mesh = trimesh.Trimesh(vertices=mesh.vertices[vmapping], faces=indices, process=False)
new_mesh.visual.uv = uvs
new_mesh.export("maillage_avec_uv.obj")

#plot
uvs_np = np.array(uvs)  
indices_np = np.array(indices)  

plt.figure(figsize=(6, 6))
triangulation = mtri.Triangulation(uvs_np[:, 0], uvs_np[:, 1], indices_np)
plt.triplot(triangulation, color='black', linewidth=0.5)
plt.gca().invert_yaxis()  
plt.xlabel("u")
plt.ylabel("v")
plt.tight_layout()
plt.show()
