import open3d as o3d
import numpy as np
import cv2
from data.param_calib import n_l
from backprojection import back_projeter
from utils import r0_rd


mesh = o3d.io.read_triangle_mesh("fichiers_ply/mesh_cailloux.ply")
image_path = "downsampled/scene_l_0026.jpeg"
image = cv2.imread(image_path)

mesh.compute_triangle_normals()

vertices = np.asarray(mesh.vertices)
triangles = np.asarray(mesh.triangles)
triangle_normals = np.asarray(mesh.triangle_normals)


def is_face_in_the_camera_direction(normale_face, normale_image, cos_theta_min:float=0) :
    """Renvoie True si la face est visible depuis la vue donnee, False sinon""" 
    return np.dot(normale_face, normale_image) > cos_theta_min


def is_face_visible(idx_face, raycasting_scene) :
    """
    Renvoie True si la face est visible, False sinon
    - idx_face : l'indice de la face dans triangles
    """
    face = triangles[idx_face]
    X = vertices[face].mean(axis=0)
    Y = back_projeter(X, image)
    r0, rd = r0_rd(Y) 
    rays = np.hstack((np.array([r0]), np.array([rd])))
    rays_tensor = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    hits = raycasting_scene.cast_rays(rays_tensor)
    if hits["t_hit"].numpy()[0] == float('inf'):  # Si t_hit est inf, cela signifie qu'aucune intersection
        return False  # Aucune intersection, la face n'est donc pas visible

    # Si t_hit n'est pas inf, récupérer l'ID du triangle touché
    hit_face = hits["primitive_ids"].numpy()[0]
    
    # Vérifier si la face touchée est la bonne (la face qu'on testait)
    return hit_face == idx_face


# etape 1 -- filtration selon la direction
good_direction_faces_indices = []
for i in range(len(triangles)):
    if is_face_in_the_camera_direction(triangle_normals[i], n_l, 0.1):
        good_direction_faces_indices.append(i)

# etape 2 -- selon la visibilite

raycasting_scene = o3d.t.geometry.RaycastingScene()
vertices_tensor = o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32)
triangles_tensor = o3d.core.Tensor(triangles, dtype=o3d.core.Dtype.UInt32)
raycasting_scene.add_triangles(vertices_tensor, triangles_tensor)

visible_faces_indices = []
for i in good_direction_faces_indices:
    if is_face_visible(i, raycasting_scene):
        visible_faces_indices.append(i)

final_faces = triangles[visible_faces_indices]
final_mesh = o3d.geometry.TriangleMesh()
final_mesh.vertices = o3d.utility.Vector3dVector(vertices)
final_mesh.triangles = o3d.utility.Vector3iVector(final_faces)
final_mesh.compute_triangle_normals()


# Tracer le vecteur n_l
origin = -n_l * 200
nl_endpoint = origin - n_l * 300
line_nl = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector([origin, nl_endpoint]),
    lines=o3d.utility.Vector2iVector([[0, 1]])
)
line_nl.paint_uniform_color([1, 0, 0])  # Rouge

# Affichage final
o3d.visualization.draw_geometries([final_mesh, line_nl], 
                                  window_name="Mesh - Faces Réellement Visibles",
                                  mesh_show_back_face=True)