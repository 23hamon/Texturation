import open3d as o3d
import numpy as np
import time


# Charger le nuage de points
pcd = o3d.io.read_point_cloud("fichiers_ply/nuage_oriente.ply")
k = 150  # Nombre de voisins les plus proches

pcd.orient_normals_consistent_tangent_plane(k)

# Créer le maillage 3D à partir du nuage de points
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)

# Supprimer les sommets dont la densité est inférieure au seuil
densities = np.asarray(densities)
seuil = np.percentile(densities, 44)
mesh.remove_vertices_by_mask(densities < seuil)

mesh.orient_triangles()
mesh.compute_vertex_normals()

# Visualiser le maillage avec les normales
o3d.visualization.draw_geometries([mesh], point_show_normal=True)
o3d.io.write_triangle_mesh("fichiers_ply/mesh_cailloux_min.ply", mesh)