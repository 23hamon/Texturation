import open3d as o3d
import numpy as np
import cv2
#from data.param_calib import n_l, n_r
from data.calib_luca import n_l, n_r
from backprojection import back_projeter
from utils import get_image_data, r0_rd

def is_face_in_the_camera_direction(normale_face, normale_image, cos_theta_max:float=0) :
    """Renvoie True si la face est dans le meme sens que la camera depuis la vue donnee, False sinon""" 
    return np.dot(normale_face, normale_image) < cos_theta_max

def get_visible_faces(mesh,                         # mesh 3D
                      h, w,                         # taille de l'image
                      rot, t,                       # position du rig par rapport au repere monde
                      type_camera="l",              # type de camera ("l" ou "r")
                      cos_theta_max_pre_filtrage=0  # cos minimal de l'angle relatif entre les normales de la camera et de la face
                      ) :
    """
    Utilise le raycasting pour determiner les faces visibles depuis une vue
    Renvoie un np array de 0 et de 1 ou chaque case correspond a la visibilite d'une face du meme indice 
    /!\ Ne regarde que si le centre de chaque triangle est visible
    """
    n_cam = n_l if type_camera=="l" else n_r
    n_monde = rot.T @ n_cam 

    # chargement des faces
    mesh.compute_triangle_normals()
    vertices = np.asarray(mesh.vertices) # tableau de sommets (points dans R^3) : shape (N_ver, 3)
    triangles = np.asarray(mesh.triangles) # tableau de tableau de taille 3 : shape (N_tri, 3)
    # Chaque tableau [v1, v2, v3] represente un triangle, de sommets d'indices v1, v2 et v3 dans vertices
    N_tri = len(triangles)
    triangle_normals = np.asarray(mesh.triangle_normals) # tableau de vecteurs : shape (N_tri, 3)
    are_triangles_visible = np.zeros(N_tri, dtype=int)
   
    # -- Etape 1 -- filtration par extraction des faces dans la bonne direction
    # -- Etape 2 -- Retro-projection du centre de chaque triangle dans la bonne direction
    rays_to_faces = np.zeros((N_tri, 6), dtype=np.float64) # pour chaque triangle, r0 et rd seront stockes
    # rays_to_faces contiendra r0=0,rd=0 pour les triangles mal orientes
    for i in range(N_tri) :
        # filtration par extraction des faces dans la bonne direction
        normal = triangle_normals[i, :]
        if is_face_in_the_camera_direction(normal, n_monde, cos_theta_max_pre_filtrage) : 
            tri = triangles[i]
            X_center = vertices[tri].mean(axis=0) # centre de la face
            _, r0, rd = back_projeter(X_center, h, w, rot, t, type_camera) # retro-projection
            rays_to_faces[i, :] = np.concatenate([np.array(r0), np.array(rd)])  

    # -- Etape 3 -- Ray-tracing de chaque rayon visant une face
    # creation de la scene de raytracing
    raycasting_scene = o3d.t.geometry.RaycastingScene()
    vertices_tensor = o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32)
    triangles_tensor = o3d.core.Tensor(triangles, dtype=o3d.core.Dtype.UInt32)
    raycasting_scene.add_triangles(vertices_tensor, triangles_tensor)

    # mask pour ne garder que les triangles bien orientes
    ray_norms = np.linalg.norm(rays_to_faces, axis=1)  # norme globale du vecteur [r0|rd]
    valid_rays_mask = ray_norms > 0
    valid_rays = rays_to_faces[valid_rays_mask]
    valid_face_indices = np.nonzero(valid_rays_mask)[0]  # pour remapper apres
    rays_tensor = o3d.core.Tensor(valid_rays, dtype=o3d.core.Dtype.Float32)
    hits = raycasting_scene.cast_rays(rays_tensor)

    # -- Etape 4 -- Garder les faces visibles d'apres le raycasting
    for i, hit in enumerate(hits["primitive_ids"].numpy()): # hits["primitive_ids"] : le tableau des faces touchees en premier de chaque hit
        original_face_idx = valid_face_indices[i]
        if hit == original_face_idx:
            are_triangles_visible[original_face_idx] = 1

    return are_triangles_visible

def reconstruct_visible_mesh(original_mesh,
                             are_triangles_visible) :
    """
    Reconstruit a partrir du tableau des faces visibles, le mesh contenant uniquement les faces visibles 
    Utile pour la visualisation
    """
    triangles = np.asarray(original_mesh.triangles)
    vertices = np.asarray(original_mesh.vertices)
    visible_faces_indices = np.where(are_triangles_visible == 1)[0]
    visible_triangles = triangles[visible_faces_indices]

    used_vertices_indices = np.unique(visible_triangles)
    new_vertices = vertices[used_vertices_indices]

    index_remap = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices_indices)}
    remapped_triangles = np.vectorize(index_remap.get)(visible_triangles)

    visible_mesh = o3d.geometry.TriangleMesh()
    visible_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
    visible_mesh.triangles = o3d.utility.Vector3iVector(remapped_triangles)
    visible_mesh.compute_vertex_normals()

    return visible_mesh

def generate_view_matrix(mesh, transforms, h, w, type_camera) :
    """
    Genere la matrice Mij ou i represente la face, et j la vue
    Mij = 1 si la face i est visible sur la vue j, 0 sinon
    **INPUT** : 
    - mesh : le mesh 3D contenant les triangles
    - transforms : le tableau des transformations des vues : transforms[j-1] = (rot_j, t_j)
    - h, w : les dimensions d'une image
    - type_camera : "l" pour la camera de gauche, "r" pour la camera de droite
    """
    triangles = np.asarray(mesh.triangles)
    N_tri = len(triangles)
    N_views = len(transforms)
    Mij = np.zeros((N_tri, N_views))
    for j in range(N_views) :
        rot, t = transforms[j]
        Mij[:, j] = get_visible_faces(mesh, h, w, rot, t, type_camera)
        print(f"Vue {j+1}/{N_views} terminee")
    return Mij


if __name__ == "__main__" :

    # image_id = 29
    # type_camera = "l"
    # image_path = f"downsampled/scene_{type_camera}_00{image_id}.jpeg"
    # image = cv2.imread(image_path)
    # h, w = image.shape[:2]
    # r, t = get_image_data(image_id)
    # rot, _ = cv2.Rodrigues(r)
    # mesh = o3d.io.read_triangle_mesh("fichiers_ply/mesh_cailloux_luca.ply")
    # # Calcul des faces visibles
    # are_triangles_visible = get_visible_faces(mesh, h, w, rot, t, type_camera)
    # visible_mesh = reconstruct_visible_mesh(mesh, are_triangles_visible)
    # # Affichage 
    # lignes_coins_l = []
    # lignes_coins_r = []
    # for x in [0, 2999] :
    #     for y in [0, 1999] :
    #         Y= np.array([x, y], dtype=np.float64)
    #         r0_coin_l, rd_coin_l = r0_rd(Y, rot, t, "l")
    #         r0_coin_r, rd_coin_r = r0_rd(Y, rot, t, "r")
    #         ligne_coin_l = o3d.geometry.LineSet()
    #         ligne_coin_l.points = o3d.utility.Vector3dVector([r0_coin_l+1200*rd_coin_l, r0_coin_l +500 * rd_coin_l])
    #         ligne_coin_l.lines = o3d.utility.Vector2iVector([[0, 1]])
    #         ligne_coin_l.colors = o3d.utility.Vector3dVector([[0, 0, 1]])
    #         lignes_coins_l.append(ligne_coin_l)
    #         ligne_coin_r = o3d.geometry.LineSet()
    #         ligne_coin_r.points = o3d.utility.Vector3dVector([r0_coin_r+1200*rd_coin_r, r0_coin_r +500 * rd_coin_r])
    #         ligne_coin_r.lines = o3d.utility.Vector2iVector([[0, 1]])
    #         ligne_coin_r.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
    #         lignes_coins_r.append(ligne_coin_r)
    # o3d.visualization.draw_geometries([visible_mesh]+lignes_coins_l+lignes_coins_r)

    mesh = o3d.io.read_triangle_mesh("fichiers_ply/mesh_cailloux_luca.ply")
    image_path = f"downsampled/scene_{"l"}_00{"32"}.jpeg"
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    transforms = []
    for j in range(52) :
        r, t = get_image_data(j+1)
        rot, _ = cv2.Rodrigues(r)
        transforms.append((rot, t)) 
    Mij = generate_view_matrix(mesh, transforms, h, w, "l")
    np.save("fichiers_intermediaires/Mij.npy", Mij)



# ancien code
# def show_visible_faces(mesh,                        # mesh 3D
#                       h, w,                         # taille de l'image
#                       rot, t,                       # position du rig par rapport au repere monde
#                       type_camera="l",              # type de camera ("l" ou "r")
#                       cos_theta_min_pre_filtrage=0  # cos minimal de l'angle relatif entre les normales de la camera et de la face
#                       ) :
#     """
#     utilise le raycasting pour determiner les faces visibles depuis une vue
#     Affiche la partie du mesh qui est visible
#     /!\ Ne regarde que si le centre de chaque triangle est visible
#     """
#     n_cam = n_l if type_camera=="l" else n_r
#     n_monde = rot.T @ n_cam 

#     # chargement des faces
#     mesh.compute_triangle_normals()
#     vertices = np.asarray(mesh.vertices) # tableau de sommets (points dans R^3)
#     triangles = np.asarray(mesh.triangles) # tableau de tableau de taille 3, chaque tableau [v1, v2, v3] represente un triangle, de sommets d'indices v1, v2 et v3 dans vertices
#     triangle_normals = np.asarray(mesh.triangle_normals)

#     # -- Etape 1 -- filtration par extraction des faces dans la bonne direction
#     triangles_filtre = []
#     for i, (tri, normal) in enumerate(zip(triangles, triangle_normals)):
#         if is_face_in_the_camera_direction(normal, n_monde, cos_theta_min_pre_filtrage):
#             triangles_filtre.append(tri)
#     triangles_filtre = np.array(triangles_filtre)
#     unique_vertex_indices, new_triangles = np.unique(triangles_filtre, return_inverse=True)
#     vertices_filtre = vertices[unique_vertex_indices]
#     triangles_filtre = new_triangles.reshape(-1, 3)

#     mesh_filtre = o3d.geometry.TriangleMesh()
#     mesh_filtre.vertices = o3d.utility.Vector3dVector(vertices_filtre)
#     mesh_filtre.triangles = o3d.utility.Vector3iVector(triangles_filtre)
#     mesh_filtre.compute_triangle_normals

#     # -- Etape 2 -- Retro-projection du centre de chaque triangle du mesh
#     N_tri = len(triangles_filtre)
#     rays_to_faces = np.zeros((N_tri, 6), dtype=np.float64) # pour chaque triangle, r0 et rd seront stockes
#     for i in range(N_tri) :
#         tri = triangles_filtre[i]
#         X_center = vertices_filtre[tri].mean(axis=0) # centre de la face
#         _, r0, rd = back_projeter(X_center, h, w, rot, t, type_camera)
#         rays_to_faces[i, :] = np.concatenate([np.array(r0), -np.array(rd)])  

#     # -- Etape 3 -- Ray-tracing de chaque rayon visant une face
#     raycasting_scene = o3d.t.geometry.RaycastingScene()
#     vertices_tensor = o3d.core.Tensor(vertices_filtre, dtype=o3d.core.Dtype.Float32)
#     triangles_tensor = o3d.core.Tensor(triangles_filtre, dtype=o3d.core.Dtype.UInt32)
#     raycasting_scene.add_triangles(vertices_tensor, triangles_tensor)
#     faces_intersections = {}
#     rays_tensor = o3d.core.Tensor(rays_to_faces, dtype=o3d.core.Dtype.Float32)
#     hits = raycasting_scene.cast_rays(rays_tensor)

#     for face_idx, hit in enumerate(hits["primitive_ids"].numpy()):
#         intersected_faces = hit[hit != 4294967295]  # Filtrer les valeurs "pas d'intersection"
#         faces_intersections[face_idx] = list(intersected_faces)

#     # -- Etape 4 -- Conservation des faces visibles par la camera
#     faces_visibles = set(hit_faces[0] for hit_faces in faces_intersections.values() if len(hit_faces) > 0)
#     triangles_visibles = triangles_filtre[list(faces_visibles)]
#     unique_vertex_indices, new_triangles = np.unique(triangles_visibles, return_inverse=True)
#     vertices_visibles = vertices_filtre[unique_vertex_indices]
#     triangles_visibles = new_triangles.reshape(-1, 3)

#     mesh_visible = o3d.geometry.TriangleMesh()
#     mesh_visible.vertices = o3d.utility.Vector3dVector(vertices_visibles)
#     mesh_visible.triangles = o3d.utility.Vector3iVector(triangles_visibles)
#     mesh_visible.compute_triangle_normals()

#     o3d.visualization.draw_geometries([mesh_visible], window_name="Mesh - Faces Visibles")
#     #o3d.io.write_triangle_mesh("fichiers_intermediaires/mesh_visible_test.ply", mesh_visible)