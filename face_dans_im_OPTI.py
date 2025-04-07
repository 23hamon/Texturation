import open3d as o3d
import numpy as np
import cv2
from data.param_calib import n_l
from backprojection import back_projeter

def is_face_in_the_camera_direction(normale_face, normale_image, cos_theta_min:float=0) :
    """Renvoie True si la face est visible depuis la vue donnee, False sinon""" 
    return np.dot(normale_face, normale_image) > cos_theta_min

# Chargement du mesh entier

mesh = o3d.io.read_triangle_mesh("fichiers_ply/mesh_cailloux_low.ply")
image_path = "downsampled/scene_l_0026.jpeg"
image = cv2.imread(image_path)
h, w = image.shape[:2]

mesh.compute_triangle_normals()
vertices = np.asarray(mesh.vertices) # tableau de sommets (points dans R^3)
triangles = np.asarray(mesh.triangles) # tableau de tableau de taille 3, chaque tableau [v1, v2, v3] represente un triangle, de sommets d'indices v1, v2 et v3 dans vertices
triangle_normals = np.asarray(mesh.triangle_normals)

print(f"Mesh chargé. Nombre de faces : {len(triangles)}. Nombre de vertices : {len(vertices)}")

# -- Etape 1
# Extraction des faces dans la bonne direction
# Creation du mesh filtre

triangles_filtre = []
indices_mapping = {}
for i, (tri, normal) in enumerate(zip(triangles, triangle_normals)):
    if is_face_in_the_camera_direction(normal, n_l):
        triangles_filtre.append(tri)

triangles_filtre = np.array(triangles_filtre)
unique_vertex_indices, new_triangles = np.unique(triangles_filtre, return_inverse=True)
vertices_filtre = vertices[unique_vertex_indices]
triangles_filtre = new_triangles.reshape(-1, 3)

mesh_filtre = o3d.geometry.TriangleMesh()
mesh_filtre.vertices = o3d.utility.Vector3dVector(vertices_filtre)
mesh_filtre.triangles = o3d.utility.Vector3iVector(triangles_filtre)
mesh_filtre.compute_triangle_normals()

print(f"Mesh filtré. Nombre de faces : {len(triangles_filtre)}. Nombre de vertices : {len(vertices_filtre)}")

o3d.visualization.draw_geometries([mesh_filtre], window_name="Mesh - Faces Visibles")


# -- Etape 2
# Retro-projection du centre de chaque triangle du mesh
# Temps de calcul ~ 0.0022 sec par face

N_tri = len(triangles_filtre)
rays_to_faces = np.zeros((N_tri, 6), dtype=np.float64) # pour chaque triangle, r0 et rd seront stockes
for i in range(N_tri) :
    tri = triangles_filtre[i]
    X_center = vertices_filtre[tri].mean(axis=0) # centre de la face
    Y_best, r0, rd = back_projeter(X_center, h, w)
    rays_to_faces[i, :] = np.concatenate([np.array(r0), -np.array(rd)])  

np.save("fichiers_intermediaires/rays_to_faces.npy", rays_to_faces) 
print("Back_projection de chaque face enregistrée.")

print("Exemple de rayons :")
for i in range(5):  # Afficher 5 rayons
    print(f"Rayon {i}: Origine = {rays_to_faces[i, :3]}, Direction = {rays_to_faces[i, 3:]}")

# -- Etape 3
# Ray-tracing de chaque rayon visant une face

raycasting_scene = o3d.t.geometry.RaycastingScene()
vertices_tensor = o3d.core.Tensor(vertices_filtre, dtype=o3d.core.Dtype.Float32)
triangles_tensor = o3d.core.Tensor(triangles_filtre, dtype=o3d.core.Dtype.UInt32)
raycasting_scene.add_triangles(vertices_tensor, triangles_tensor)

faces_intersections = {}
rays_tensor = o3d.core.Tensor(rays_to_faces, dtype=o3d.core.Dtype.Float32)
hits = raycasting_scene.cast_rays(rays_tensor)

print("Nombre total d'intersections valides:", np.sum(hits["primitive_ids"].numpy() != 4294967295))

for face_idx, hit in enumerate(hits["primitive_ids"].numpy()):
    intersected_faces = hit[hit != 4294967295]  # Filtrer les valeurs "pas d'intersection"
    faces_intersections[face_idx] = list(intersected_faces)

faces_intersections_np = np.array(list(faces_intersections.items()), dtype=object)
np.save("fichiers_intermediaires/faces_intersections.npy", faces_intersections_np)
print("Liste d'intersection du rayon de chaque face enregistrée.")

# -- Etape 4 
# Conservation des faces visibles par la camera

faces_visibles = set(hit_faces[0] for hit_faces in faces_intersections.values() if len(hit_faces) > 0)
triangles_visibles = triangles_filtre[list(faces_visibles)]
unique_vertex_indices, new_triangles = np.unique(triangles_visibles, return_inverse=True)

vertices_visibles = vertices_filtre[unique_vertex_indices]
triangles_visibles = new_triangles.reshape(-1, 3)

mesh_visible = o3d.geometry.TriangleMesh()
mesh_visible.vertices = o3d.utility.Vector3dVector(vertices_visibles)
mesh_visible.triangles = o3d.utility.Vector3iVector(triangles_visibles)
mesh_visible.compute_triangle_normals()

o3d.visualization.draw_geometries([mesh_visible], window_name="Mesh - Faces Visibles")

print(f"Nombre de faces visibles : {len(triangles_visibles)}")
o3d.io.write_triangle_mesh("fichiers_intermediaires/mesh_visible.ply", mesh_visible)
