import open3d as o3d
import numpy as np
from calib_luca import n_l, n_r
from backprojection import back_projeter
from tqdm import tqdm


from multiprocessing import Pool
from functools import partial


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

    K = len(triangles)
    are_triangles_visible = np.zeros(K, dtype=int)
   
    # -- Etape 1 -- filtration par extraction des faces dans la bonne direction
    # -- Etape 2 -- Retro-projection du centre de chaque triangle dans la bonne direction
    rays_to_faces = np.zeros((K, 6), dtype=np.float64) # pour chaque triangle, r0 et rd seront stockes
    # rays_to_faces contiendra r0=0,rd=0 pour les triangles mal orientes
    # d'abord on stocke tous les X dans la bonne direction, puis on retroprojecte tout 
    all_good_X = []
    correspondance_p_X = np.full((K,), None) # associe a chaque p None si la face n'est pas dans la
    # bonne direction et l'id du centre de la face dans all_good_X sinon
    idx_good_X = 0
    for p in range(K) :
        # filtration par extraction des faces dans la bonne direction
        normal = triangle_normals[p, :]
        if is_face_in_the_camera_direction(normal, n_monde, cos_theta_max_pre_filtrage) : 
            tri = triangles[p]
            X_center = vertices[tri].mean(axis=0) # centre de la face
            all_good_X.append(X_center)
            correspondance_p_X[p] = idx_good_X
            idx_good_X += 1
    # back_projection
    all_good_r0rd = [back_projeter(all_good_X[correspondance_p_X[p]],rot, t, type_camera) 
                     if correspondance_p_X[p] is not None else None for p in range(K)]
    # stockage
    for p, _r0rd in enumerate(all_good_r0rd) :
        if _r0rd is not None :
            _, r0, rd =  all_good_r0rd[p]
            rays_to_faces[p, :] = np.concatenate([np.array(r0), np.array(rd)])  

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

def compute_visibility(args, vertices, triangles, triangle_normals, type_camera, cos_theta_max):
    j, rot_image, t_image = args
    visible = get_visible_faces(vertices, triangles, triangle_normals, rot_image, t_image, type_camera, cos_theta_max)
    return j, visible

def build_Mpj(mesh, rot_images, t_images, type_camera, cos_theta_max=0) :
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
    N_views = len(rot_images)
    Mpj = np.zeros((N_tri, N_views), dtype=np.bool)
    args = [(j, rot_images[j], t_images[j]) for j in range(N_views)]
    with Pool(24) as p:
        for j, visible in tqdm(p.imap_unordered(
            partial(
                compute_visibility, 
                vertices=vertices,
                triangles=triangles,
                triangle_normals=triangle_normals,
                type_camera=type_camera,
                cos_theta_max=cos_theta_max
            ), 
            args,
            chunksize=1
        ), total=N_views) :
            Mpj[:, j] = visible     
    return Mpj
