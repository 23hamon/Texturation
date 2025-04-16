import json
import numpy as np
import cv2
from tqdm import tqdm
import trimesh
from distance_obj_camera import dist_obj_cam
from data import param_calib
from utils import get_image_data

with open("data/absolute_transforms_luca.json", "r") as f:
    camera_data = json.load(f)["0"]

<<<<<<< Updated upstream
mesh = trimesh.load_mesh('fichiers_ply/mesh_cailloux_luca_LOW.ply')
Mij = np.load("fichiers_intermediaires/MijLOW.npy")
=======
mesh = trimesh.load_mesh('fichiers_ply/mesh_cailloux_luca_low.ply')
Mij = np.load("fichiers_intermediaires/Mij.npy")
>>>>>>> Stashed changes
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
cost_matrix = np.full((len(mesh.faces), len(camera_data)), np.inf)

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

Wij = np.save("fichiers_intermediaires/WijLOW.npy", cost_matrix)
