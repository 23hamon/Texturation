import open3d as o3d
import cv2
import numpy as np

pcd = o3d.io.read_point_cloud("initial_cc_0.ply")

print(pcd)


# # calculer le mesh 3D
# pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=50))
# mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=12)

# densities = np.asarray(densities)
# seuil = np.percentile(densities, 2)
# mesh.remove_vertices_by_mask(densities < seuil)
# o3d.visualization.draw_geometries([mesh])

# Extraire les points SIFT de l'image
image = cv2.imread('scene_l_0009.jpeg', cv2.IMREAD_GRAYSCALE)
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(image, None)

# Affichage
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('Points SIFT dans l\'image', image_with_keypoints)
cv2.waitKey(0) 
cv2.destroyAllWindows()


#Paramètres de la caméra




#Projection du nuage de point sur une image 2D
