from utils import r0_rd
import cv2
import numpy as np
import open3d as o3d
import json

# chargement de l'image
image_id = 26
# position de l'image
with open("data/absolute_transforms.json") as f :
    data = json.load(f)
    t_rot = np.array(data["0"][str(image_id)]).flatten()
    rot,_ = cv2.Rodrigues(np.array(t_rot[:3], dtype=np.float64))
    t = np.array(t_rot[3:], dtype=np.float64)
print(rot, t)
    
image_path = f"downsampled/scene_l_00{str(image_id)}.jpeg"
print(image_path)
image = cv2.imread(image_path)
hauteur_y, largeur_x = image.shape[:2] # 2000, 3000

x_image = 1900
y_image = 1200

Y= np.array([x_image, y_image], dtype=np.float64)


cv2.circle(image, (int(x_image), int(y_image)), radius=10, color=(0, 0, 255), thickness=-1)
cv2.imwrite("image_avec_point.jpg", cv2.flip(image, 1))


r0, rd = r0_rd(Y, rot, t)
print(r0, rd)

mesh= o3d.io.read_triangle_mesh("fichiers_ply/mesh_cailloux.ply")

# tracer le rayon
ligne = o3d.geometry.LineSet()
ligne.points = o3d.utility.Vector3dVector([r0-1200*rd, r0 -500 * rd])
ligne.lines = o3d.utility.Vector2iVector([[0, 1]])

lignes_coins = []
for x in [0, 2999] :
    for y in [0, 1999] :
        Y= np.array([x, y], dtype=np.float64)
        r0_coin, rd_coin = r0_rd(Y, rot, t)
        ligne_coin = o3d.geometry.LineSet()
        ligne_coin.points = o3d.utility.Vector3dVector([r0_coin-1200*rd_coin, r0_coin -500 * rd_coin])
        ligne_coin.lines = o3d.utility.Vector2iVector([[0, 1]])
        lignes_coins.append(ligne_coin)

# visualisation
o3d.visualization.draw_geometries([mesh, ligne]+lignes_coins)
