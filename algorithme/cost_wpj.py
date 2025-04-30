import json
import numpy as np
import cv2
from tqdm import tqdm
import trimesh
import calib_luca as param_calib
from utils import get_image_data, FASTr0_rd, closest_point_to_two_lines

def dist_obj_cam(X, R_cam, t_cam, R_cam_l_r, t_cam_l_r, image_height=2000, image_width=3000):
    '''Le R-cam eet t_cam de chaque vue a été calculé en amout'''

    ro_l, rd_l = FASTr0_rd(X, R_cam, t_cam, cam="l")
    ro_r, rd_r = FASTr0_rd(X, R_cam_l_r, t_cam_l_r, cam="r")

    p, _ = closest_point_to_two_lines(ro_l, rd_l, ro_r, rd_r)
    distance_l = ((p - ro_l) ** 2).sum() ** 0.5
    distance_r = ((p - ro_r) ** 2).sum() ** 0.5

    return p, distance_l, distance_r

with open("absolute_transforms_luca.json", "r") as f:
    camera_data = json.load(f)["0"]

mesh = trimesh.load_mesh('ply/mesh_cailloux_luca_CLEAN.ply')
Mij = np.load("tensors/Mpj.npy")
R_cam_l_r = param_calib.R_DG
t_cam_l_r = param_calib.t_DG
n_l = param_calib.n_l

#initialisation
start, stop = 6, 7
mesh.visual.vertex_colors = [200, 200, 200, 255]
mesh.visual.face_colors[start:stop] = [255, 0, 0, 255]

transforms = []
for key in camera_data:
    R, t = get_image_data(key)
    transforms.append((R, t))

# function 
def cost_function(normal_face, normal_view, X, R_cam, t_cam, R_cam_l_r, t_cam_l_r):
    d = dist_obj_cam(X, R_cam, t_cam, R_cam_l_r, t_cam_l_r)
    return np.dot(normal_face, normal_view) * d[1]

#calcul de la matrice de cout
cost_matrix = np.full((len(mesh.faces), len(camera_data)), 1e20) #np.inf a la place de 1e9

for i in tqdm(range(len(mesh.faces))):
    X = mesh.triangles_center[i]
    normal_face = mesh.face_normals[i]
    for j, key in enumerate(camera_data):
        if Mij[i, int(key) - 1] == 1: #si face visible
            R, T = transforms[j]
            normal_view = - R.T @ n_l
            cost_matrix[i, int(key) - 1] = cost_function(normal_face, normal_view, X, R, T, R_cam_l_r, t_cam_l_r)

#visualisation
scene = mesh.scene()
scene.add_geometry(mesh)
lines = trimesh.load_path(mesh.vertices[mesh.edges_unique])
lines.colors = [[0, 0, 0, 255]] * len(lines.entities)
scene.add_geometry(lines)

i = 6 #par exemple
center = mesh.triangles_center[i]
normal = mesh.face_normals[i]
line = trimesh.load_path([center, center + 50 * normal])
line.colors = [[255, 0, 0, 255]]
scene.add_geometry(line)

for j, view in enumerate(camera_data):
    if Mij[i, int(view) - 1] == 1:
        R, t = get_image_data(view)
        normal_view = - R.T @ n_l
        line = trimesh.load_path([center, center + 50 * normal_view])
        line.colors = [[0, 255, 0, 255]]
        scene.add_geometry(line)

scene.show()

print(Mij[i, :])
print(cost_matrix[i, :])

Wij = np.save("tensors/Wpj.npy", cost_matrix)
