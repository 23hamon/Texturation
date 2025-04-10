
import json
import numpy as np
import cv2
from tqdm import tqdm
import trimesh
from distance_obj_camera import dist_obj_cam


with open("data/absolute_transforms.json", "r") as f:
    camera_data = json.load(f)["0"]

mesh = trimesh.load_mesh('fichiers_ply/mesh_cailloux_low.ply')


def cost_function(normal_face, normal_view):
        cost = np.dot(normal_face, normal_view)
        if cost <= 0:
            return cost
        else : 
            print("cost > 0")

#on se place à l'origine
for i, face in enumerate(mesh.faces[0:10]): 
    #print(i, face)
    face = mesh.faces[face] #on prend la face i
    for j, view in enumerate(camera_data.keys()): #on prend la vue 26 par exemple, alors j = 0 et view = 26
        #on prend l'indice de la vue j
        #on regarde si cette face est visible dans l'image --> on regarde la matrice (i,j) et on regarde si le mij=1
        #on calcule la normale de la face 
        normal_face = mesh.face_normals[face]
        #on calcule la normale de la vue 
        R, T = camera_data[view][0], camera_data[view][1]
        print(R, T)
        R, _ = cv2.Rodrigues(R)
        normal_view = R[:, 2]
        cost = cost_function(normal_face, normal_view)
        print(f"cost = {cost}")