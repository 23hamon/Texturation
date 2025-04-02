from utils import r0_rd
import cv2
import numpy as np
import open3d as o3d


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

mesh= o3d.io.read_triangle_mesh("fichiers_ply/mesh_cailloux.ply")

# tracer le rayon
ligne = o3d.geometry.LineSet()
ligne.points = o3d.utility.Vector3dVector([r0-1200*rd, r0 -500 * rd])
ligne.lines = o3d.utility.Vector2iVector([[0, 1]])

# visualisation
o3d.visualization.draw_geometries([mesh, ligne])
