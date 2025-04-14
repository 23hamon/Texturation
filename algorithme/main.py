import open3d as o3d
import numpy as np
import 
# chargement des fichiers

mesh = o3d.io.read_triangle_mesh("fichiers_ply/mesh_cailloux_low.ply")


Fi = np.asarray(mesh.triangles)