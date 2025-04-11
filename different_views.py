
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

# #on fait pour la caméra de référence
# ref = "26"
# ref_R_vec = np.array(camera_data[ref][0])
# ref_T = np.array(camera_data[ref][1])

# #on cree deux listes, qui pour chaque vue i lui associent Rotation, Translation, et enfin chemin de l'image

# rotr = []
# images= []
# image_dir = "downsampled/scene_l_"


# for view, (R, T) in tqdm(camera_data.items()):
#     R_vect = np.array(R)
#     R, _ = cv2.Rodrigues(R_vect)
#     T = np.array(T)
#     image_path = f"{image_dir}{view.zfill(4)}.jpeg"
#     rotr.append((R, T))
#     images.append(image_path)


# rot_mat = [pair[0] for pair in rotr]



# view_dirs = []
# for rvec in rot_mat: 
#     view_dirs.append(rvec[:,2])

# print(view_dirs[3])

# #puis on calcule la normale de chaque face du mesh 
# normals = np.asarray(mesh.vertex_normals)

# best_view = []
# best_image = []

# for normal in tqdm(normals):
#     dots = [np.dot(view_dir, normal) for view_dir in view_dirs]
#     best = np.argmax(dots)  # on prend la direction pour laquelle on est le plus opposé 
#     corresponding_image = images[best]
#     best_view.append(best)
#     best_image.append(corresponding_image)

# print(best_image[3200])
# print(best_view[0])


# mesh.visual.vertex_colors = [200, 200, 200, 255]  # Gris clair
# start = 3200
# stop = 3210

# #colorier 10 faces en rouge
# colors = [255, 0, 0, 255]
# mesh.visual.face_colors[start:stop]  = colors


# #affichage des normales 

# scene = mesh.scene()

# face_normals = mesh.face_normals[start:stop] 
# face_centroids = mesh.triangles_center[start:stop]  

# for i, (centroid, normal) in enumerate(zip(face_centroids, face_normals)):
#     line_start = centroid
#     line_end = centroid + normal
#     scene.add_geometry(trimesh.load_path([line_start, line_end]))
#     print(best_image[start + i])

# # on distingue chaque face en créant les arêts
# edges = mesh.edges_unique
# lines = trimesh.load_path(mesh.vertices[edges])
# lines.colors = [[0, 0, 0, 255]] * len(lines.entities) 

# scene.add_geometry(mesh)
# scene.add_geometry(lines)
# scene.show()



