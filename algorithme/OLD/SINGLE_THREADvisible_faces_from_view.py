import open3d as o3d
import numpy as np
from calib_luca import n_l, n_r
from backprojection import back_projeter
from tqdm import tqdm


import multiprocessing as mp

def is_face_in_the_camera_direction(normale_face, normale_image, cos_theta_max:float=0) :
    """Renvoie True si la face est dans le meme sens que la camera depuis la vue donnee, False sinon""" 
    return np.dot(normale_face, normale_image) < cos_theta_max

def get_visible_faces(vertices, triangles, triangle_normals,     # mesh 3D
                      rot, t,                                    # position du rig par rapport au repere monde
                      type_camera="l",                           # type de camera ("l" ou "r")
                      cos_theta_max_pre_filtrage=0               # cos minimal de l'angle relatif entre les normales de la camera et de la face
                      ) :
    """
    Utilise le raycasting pour determiner les faces visibles depuis une vue
    Renvoie un np array de 0 et de 1 ou chaque case correspond a la visibilite d'une face du meme indice 
    /!\ Ne regarde que si le centre de chaque triangle est visible
    """
    n_cam = n_l if type_camera=="l" else n_r
    n_monde = rot.T @ n_cam 

    N_tri = len(triangles)
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
            _, r0, rd = back_projeter(X_center,rot, t, type_camera) # retro-projection
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


def build_Mpj(mesh, transforms, type_camera, cos_theta_max=0) :
    """
    Genere la matrice Mpj ou p represente la face, et j la vue
    Mpj = True si la face p est visible sur la vue j, 0 sinon
    **INPUT** : 
    - mesh : le mesh 3D contenant les triangles
    - transforms : le tableau des transformations des vues : transforms[j-1] = (rot_j, t_j)
    - type_camera : "l" pour la camera de gauche, "r" pour la camera de droite
    """
    # chargement des faces
    mesh.compute_triangle_normals()
    vertices = np.asarray(mesh.vertices) # tableau de sommets (points dans R^3) : shape (N_ver, 3)
    triangles = np.asarray(mesh.triangles) # tableau de tableau de taille 3 : shape (N_tri, 3)
    triangle_normals = np.asarray(mesh.triangle_normals) # tableau de vecteurs : shape (N_tri, 3)
    N_tri = len(triangles)
    N_views = len(transforms)
    Mpj = np.zeros((N_tri, N_views), dtype=np.bool)
    for j in tqdm(range(N_views)) :
        rot, t = transforms[j]
        Mpj[:, j] = get_visible_faces(vertices, triangles, triangle_normals, rot, t, type_camera, cos_theta_max)
    return Mpj

#if __name__ == "__main__" :
    # mesh = o3d.io.read_triangle_mesh("fichiers_ply/mesh_cailloux_luca_LOW.ply")
    # image_path = f"downsampled/scene_{"l"}_00{"32"}.jpeg"
    # image = cv2.imread(image_path)
    # h, w = image.shape[:2]
    # transforms = []
    # for j in range(52) :
    #     rot, t = get_image_data(j+1)
    #     transforms.append((rot, t)) 
    # Mij = generate_view_matrix(mesh, transforms, h, w, "l")
    # np.save("fichiers_intermediaires/MijLOW.npy", Mij)
