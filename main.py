import param_calib
from calibration.utils import trace_refract_ray
import cv2
import numpy as np
import open3d as o3d

print(param_calib.air_K_l, param_calib.D_l, param_calib.n_l, param_calib.THICKNESS, param_calib.ETA_GLASS, param_calib.ETA_WATER)

def r0_rd(Y) :
    """Donne r0 et rd pour un point Y = [a, b] dans l'image"""
    return trace_refract_ray(Y, param_calib.air_K_l, param_calib.D_l, param_calib.n_l, param_calib.THICKNESS, param_calib.ETA_GLASS, param_calib.ETA_WATER)


image_path = "downsampled/scene_l_0026.jpeg"
image = cv2.imread(image_path)
hauteur_y, largeur_x = image.shape[:2] # 2000, 3000

x_image = 1700
y_image = 1200

Y= np.array([x_image, y_image], dtype=np.float64)


cv2.circle(image, (int(x_image), int(y_image)), radius=10, color=(0, 0, 255), thickness=-1)
cv2.imwrite("image_avec_point.jpg", image)


r0, rd = r0_rd(Y)
print(r0, rd)

mesh= o3d.io.read_triangle_mesh("mesh_cailloux.ply")

"""pcd = o3d.io.read_point_cloud("initial_cc_0.ply")
print(pcd)
# calculer le mesh 3D
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50))
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=12)
densities = np.asarray(densities)
seuil = np.percentile(densities, 2)
mesh.remove_vertices_by_mask(densities < seuil)
"""
# tracer le rayon

ligne = o3d.geometry.LineSet()
ligne.points = o3d.utility.Vector3dVector([r0-1200*rd, r0 -500 * rd])
ligne.lines = o3d.utility.Vector2iVector([[0, 1]])


# visualisation

o3d.visualization.draw_geometries([mesh, ligne])
#points = pcd.points
