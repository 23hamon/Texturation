
import json
import numpy as np
import cv2
from tqdm import tqdm
import trimesh
from distance_obj_camera import dist_obj_cam
from data import param_calib
from collections import Counter

with open("data/absolute_transforms_full.json", "r") as f:
    camera_data = json.load(f)["0"]

mesh = trimesh.load_mesh('fichiers_ply/mesh_cailloux_low.ply')
Mij = np.load("fichiers_intermediaires/Mij.npy")
np.set_printoptions(threshold=np.inf) 

print(Mij[500:510, :])


#-- functions 

def cost_function(normal_face, normal_view, X, R_cam, t_cam, R_cam_l_r, t_cam_l_r):
        d = dist_obj_cam(X, R_cam, t_cam, R_cam_l_r, t_cam_l_r)
        cost = np.dot(normal_face, normal_view) * d[1]  # distance à la caméra gauche par ex
        return cost
        # if cost < 0:
        #     return cost
        # else:
        #     return np.inf
             

#-- main

best_views = {}
cost_matrix = np.full((len(mesh.faces), len(camera_data)), np.inf)

R_cam_l_r = param_calib.RotationDroiteGauche
t_cam_l_r = param_calib.TranslationDroiteGauche

for i, face in tqdm(enumerate(mesh.faces)): 
    X = mesh.triangles_center[i]  #centre de la face i
    cost = np.inf  
    for j, view in enumerate(camera_data.keys()):  
        if Mij[i, j] == 0:  # si la face i n'est pas visible dans la vue j
            continue  # on passe à la vue suivante, wij=inf
        else : 
            normal_face = mesh.face_normals[i]
            Rvec, T = np.array(camera_data[view])
            R, _ = cv2.Rodrigues(Rvec) 
            normal_view = R.T @ np.array([0, 0, 1])  # direction de la vue dans le repère monde. On prend la ransposéee parce que on va depuis le monde vers la caméra
            cost_view = cost_function(normal_face, normal_view, X, R, T, R_cam_l_r, t_cam_l_r)
            cost_matrix[i, j] = cost_view

print(cost_matrix[500:510, :])



#visualisation 

mesh.visual.vertex_colors = [200, 200, 200, 255]  # Gris clair
start = 500
stop = 510
colors = [255, 0, 0, 255]
mesh.visual.face_colors[start:stop]  = colors

print({i: best_views[i] for i in range(start, stop) if i in best_views})




# combien de fois chaque vue apparaît comme best view
view_counts = Counter(best_views.values())
for view, count in sorted(view_counts.items(), key=lambda x: int(x[0])):
    print(f"Vue {view} : {count}")


#affichage des normales 

scene = mesh.scene()

face_normals = mesh.face_normals[start:stop] 
face_centroids = mesh.triangles_center[start:stop]  

# on distingue chaque face en créant les arêts
edges = mesh.edges_unique
lines = trimesh.load_path(mesh.vertices[edges])
lines.colors = [[0, 0, 0, 255]] * len(lines.entities) 

scene.add_geometry(mesh)
scene.add_geometry(lines)
scene.show()



