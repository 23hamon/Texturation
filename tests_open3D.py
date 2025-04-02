import open3d as o3d
import cv2
import numpy as np

pcd = o3d.io.read_point_cloud("initial_cc_0.ply")

print(pcd)


# # calculer le mesh 3D
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50))
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=12)

densities = np.asarray(densities)
seuil = np.percentile(densities, 2)
mesh.remove_vertices_by_mask(densities < seuil)
o3d.visualization.draw_geometries([mesh])

points = pcd.points

