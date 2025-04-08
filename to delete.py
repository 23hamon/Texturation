import json
import numpy as np
import cv2
from tqdm import tqdm
import trimesh
from trimesh.visual import texture

# Chargement des données de caméra
with open("data/absolute_transforms.json", "r") as f:
    camera_data = json.load(f)["0"]

# Chargement du mesh
mesh = trimesh.load_mesh('fichiers_ply/mesh_cailloux_min.ply')

# On définit la caméra de référence
ref = "26"
ref_R_vec = np.array(camera_data[ref][0])
ref_T = np.array(camera_data[ref][1])

# Création des listes pour les matrices de rotation, translations et images
rotr = []
images = []
image_dir = "downsampled/scene_l_"

for view, (R, T) in tqdm(camera_data.items()):
    R_vect = np.array(R)
    R, _ = cv2.Rodrigues(R_vect)
    T = np.array(T)
    image_path = f"{image_dir}{view.zfill(4)}.jpeg"
    rotr.append((R, T))
    images.append(image_path)

rot_mat = [pair[0] for pair in rotr]

# Calcul des directions des vues (normales)
view_dirs = []
for rvec in rot_mat:
    view_dirs.append(rvec[:, 2])

# Calcul des normales du mesh
normals = np.asarray(mesh.vertex_normals)

best_view = []
best_image = []

# Trouver l'image avec la normale la plus opposée à chaque normale de face
for normal in tqdm(normals):
    dots = [np.dot(view_dir, normal) for view_dir in view_dirs]
    best = np.argmin(dots)  # Direction la plus opposée à la normale
    corresponding_image = images[best]
    best_view.append(best)
    best_image.append(corresponding_image)

# Représentation des images sur le mesh en 3D
for i, normal in enumerate(normals):
    # Calculer la position du point sur la face du caillou
    vertex = mesh.vertices[i]
    
    # Récupérer l'image associée
    image_path = best_image[i]
    
    # Créer une texture à partir de l'image
    texture_image = cv2.imread(image_path)
    
    # Créer une sphère autour du vertex, de manière à représenter la position de la caméra
    camera_position = vertex + normal * 0.1  # décalage le long de la normale
    
    # Positionner la caméra dans la direction de la normale
    cam = trimesh.creation.camera(fovy=60.0, aspect_ratio=1.0)
    cam.apply_transform(mesh.isometry)
    
    # Appliquer la transformation pour orienter la caméra vers la face
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rot_mat[best_view[i]]  # Rotation
    transformation_matrix[:3, 3] = vertex  # Position du vertex
    cam.apply_transform(transformation_matrix)
    
    # Créer un matériel avec la texture
    material = trimesh.visual.materials.Material(texture=texture_image)
    
    # Associer la texture à une face
    mesh.visual.face_materials[i] = material

# Affichage du mesh avec les textures
mesh.show()
