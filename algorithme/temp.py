import open3d as o3d
import numpy as np

original_mesh = o3d.io.read_triangle_mesh("ply/mesh_cailloux_luca_high.ply")
original_mesh.compute_triangle_normals()
gray_color = np.array([0.5, 0.5, 0.5])
original_mesh.paint_uniform_color(gray_color)
o3d.visualization.draw_geometries([original_mesh])
