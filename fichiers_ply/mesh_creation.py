import open3d as o3d
import numpy as np


# Charger le nuage de points
pcd = o3d.io.read_point_cloud("fichiers_ply/nuage_de_pt_luca.ply")
k = 200  # Nombre de voisins les plus proches
pcd.estimate_normals()
pcd.orient_normals_consistent_tangent_plane(k)
pcd.normals = o3d.utility.Vector3dVector(-np.asarray(pcd.normals))

# Créer le maillage 3D à partir du nuage de points
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=12) # depth = 12

# Supprimer les sommets dont la densité est inférieure au seuil
densities = np.asarray(densities)
seuil = np.percentile(densities,1.75) #2
mesh.remove_vertices_by_mask(densities < seuil)

# conserver la plus grande composante connexe 
triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
triangle_clusters = np.asarray(triangle_clusters)
largest_cluster_idx = np.array(cluster_n_triangles).argmax()
tri_mask = triangle_clusters == largest_cluster_idx
mesh.remove_triangles_by_mask(~tri_mask)
mesh.remove_unreferenced_vertices()


mesh.orient_triangles()
mesh.compute_vertex_normals()

# Visualiser le maillage avec les normales
o3d.io.write_triangle_mesh("mesh_cailloux_luca_high.ply", mesh)

# Visualiser le maillage avec les normales
o3d.visualization.draw_geometries([mesh], point_show_normal=True)


## low
# import open3d as o3d
# import numpy as np


# # Charger le nuage de points
# pcd = o3d.io.read_point_cloud("fichiers_ply/nuage_de_pt_luca.ply")
# k = 200  # Nombre de voisins les plus proches
# pcd.estimate_normals()
# pcd.orient_normals_consistent_tangent_plane(k)
# pcd.normals = o3d.utility.Vector3dVector(-np.asarray(pcd.normals))

# # Créer le maillage 3D à partir du nuage de points
# mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8) # depth = 12

# # Supprimer les sommets dont la densité est inférieure au seuil
# densities = np.asarray(densities)
# seuil = np.percentile(densities,60) #2
# mesh.remove_vertices_by_mask(densities < seuil)

# # conserver la plus grande composante connexe 
# triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
# triangle_clusters = np.asarray(triangle_clusters)
# largest_cluster_idx = np.array(cluster_n_triangles).argmax()
# tri_mask = triangle_clusters == largest_cluster_idx
# mesh.remove_triangles_by_mask(~tri_mask)
# mesh.remove_unreferenced_vertices()


# mesh.orient_triangles()
# mesh.compute_vertex_normals()

# # Visualiser le maillage avec les normales
# o3d.io.write_triangle_mesh("mesh_cailloux_luca.ply", mesh)

# # Visualiser le maillage avec les normales
# o3d.visualization.draw_geometries([mesh], point_show_normal=True)
