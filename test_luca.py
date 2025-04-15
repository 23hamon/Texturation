from utils import r0_rd, get_image_data
import open3d as o3d
import numpy as np
import cv2

flip = False
image_id = 26
r,t = get_image_data(image_id)
rot, _ =cv2.Rodrigues(r)
print(f"r = {r}, t = {t}, \n rot = {rot}")
image_path_l = f"downsampled/scene_l_00{str(image_id)}.jpeg"
image_l = cv2.imread(image_path_l)

Y = np.array([1000,1500], dtype=np.float64)
r0, rd = r0_rd(Y, rot, t, "l")


ligne_l = o3d.geometry.LineSet()
ligne_l.points = o3d.utility.Vector3dVector([r0_l-1200*rd_l, r0_l -500 * rd_l])
ligne_l.lines = o3d.utility.Vector2iVector([[0, 1]])