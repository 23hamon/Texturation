
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
        if cost < 0 :
            return cost
        else:
             return np.inf
        
best_views = {}
 
cost_matrix = np.full((len(mesh.faces), len(camera_data)), np.inf)

for i, face in enumerate(mesh.faces): 
    cost = np.inf  
    
    for j, view in enumerate(camera_data.keys()):  # parcourt toutes les vues
        #on prend l'indice de la vue j
        #on regarde si cette face est visible dans l'image --> on regarde la matrice (i,j) et on regarde si le mij=1
        normal_face = mesh.face_normals[i]
        Rvec, T = np.array(camera_data[view])
        R, _ = cv2.Rodrigues(Rvec)
        normal_view = R[:, 2] #on prend la direction de la caméra qui est en z pour la normale de la vue
        cost_view = cost_function(normal_face, normal_view)

        # Remplir la matrice des coûts
        cost_matrix[i, j] = cost_view

        #on regarde si ce coût est le meilleur pour cette face
        if cost_view < cost:
            cost = cost_view
            best_views[i] = view  #la meilleure vue pour cette face

#VISUALISATION

mesh.visual.vertex_colors = [200, 200, 200, 255]  # Gris clair
start = 3200
stop = 3210
colors = [255, 0, 0, 255]
mesh.visual.face_colors[start:stop]  = colors

print({i: best_views[i] for i in range(start, stop) if i in best_views})


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



