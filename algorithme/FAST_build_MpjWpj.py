# Construit simultanement Mpj et Wpj pour optimiser les performances
# en ne retroprojectant qu'une fois
import numpy as np
import open3d as o3d

from backprojection import back_projeter
from utils import get_image_data

from numba import njit
from tqdm import tqdm

from multiprocessing import Pool
from functools import partial
import multiprocessing


# Cette partie du code construit r0_rd_for_good_p, qui contient les rayons de chaque "bonne" face
# vers chaque vue. Les "bonnes" faces sont cettes dont la normale forme un angle avec la camera
# superieur a theta_max
#
# Cet objet prend du temps a construire (retroprojection sur un grand nombre de faces) mais permet
# de gagner un temps considerable pour calculer Mpj et Wpj

@njit(error_model="numpy", cache=True)
def _is_face_in_the_camera_direction(normale_face, normale_image, cos_theta_max) :
    """Renvoie True si la face est dans le meme sens que la camera depuis la vue donnee, False sinon""" 
    return np.dot(normale_face, normale_image) < cos_theta_max


def _build_r0_rd(args, K, n_l, n_r, all_Xp, all_np, cos_theta_max) :
    """
    **Input :**
        - all_Xp : (K, 3)
        - all_np : (K, 3)
    **Output :**
        - allr0rdr0rd : (K, 12)
    """
    j_allr0rdr0rd = np.zeros((K, 12), dtype=np.float64)
    j, rot_j, t_j = args
    n_l_monde = rot_j.T @ n_l 
    n_r_monde = rot_j.T @ n_r
    all_Yr0rdG = [back_projeter(Xp, rot_j, t_j, "l") 
                  if _is_face_in_the_camera_direction(all_np[p], n_l_monde, cos_theta_max)
                  else None for p, Xp in enumerate(all_Xp)]
    all_Yr0rdD = [back_projeter(Xp, rot_j, t_j, "r") 
                  if _is_face_in_the_camera_direction(all_np[p], n_r_monde, cos_theta_max)
                  else None for p, Xp in enumerate(all_Xp)]
    for p in range(K) :
        r0rdr0rdG = np.concatenate([np.array(all_Yr0rdG[p][1]), np.array(all_Yr0rdG[p][2])]) if all_Yr0rdG[p] is not None else np.zeros((6,))
        r0rdr0rdD = np.concatenate([np.array(all_Yr0rdD[p][1]), np.array(all_Yr0rdD[p][2])]) if all_Yr0rdD[p] is not None else np.zeros((6,))
        j_allr0rdr0rd[p] = np.concatenate([r0rdr0rdG, r0rdr0rdD])
    return j, j_allr0rdr0rd

# avec le gros mesh, tourne en ~15min. crash avec Pool(20)
def build_r0_rd_for_good_p(K, N,
                           vertices, triangles, triangle_normals,
                           rot_images, t_images, n_l, n_r,
                           cos_theta_max) :
    """
    Construit le tableau des r0G|rdG|r0D|rdD pour chaque pixel pour chaque vue
    **Output :**
        - r0_rd_for_good_p : np array de shape (N, K, 12)
            r0_rd_for_good_p[j, p, :] vaut les quatres vecteurs concatenes r0G|rdG|r0D|rdD 
            si la face p n'est pas visible depuis la vue j, vaudra 0
    """
    # construction de all_Xp
    all_Xp = np.zeros((K, 3), dtype=np.float64)
    for p in range(K) :
        tri = triangles[p]
        X_center = vertices[tri].mean(axis=0)
        all_Xp[p] = X_center
    
    # backprojection pour chaque j
    r0_rd_for_good_p = np.zeros((N, K, 12))
    args = [(j, rot_images[j], t_images[j]) for j in range(N)]
    with Pool(16) as p:
        for j, j_allr0rdr0rd in tqdm(p.imap_unordered(
            partial(_build_r0_rd,
                    K=K,
                    n_l=n_l,
                    n_r=n_r,
                    all_Xp=all_Xp,
                    all_np=triangle_normals,
                    cos_theta_max=cos_theta_max
                    ),
                    args,
                    chunksize=1
        ), total=N) :
            r0_rd_for_good_p[j] = j_allr0rdr0rd
    return r0_rd_for_good_p

# Cette partie du code utilise r0_rd_for_good_p pour faire du raytracing sur les rayons
# calcules, dans le but d'identifier les faces reellement visibles depuis chaque vue

def _get_visible_faces(j, K, vertices, triangles, 
                       r0_rd_for_good_p) :
    """
    Utilise le raycasting pour determiner les faces visibles depuis une vue
    /! Ne regarde que si le centre de chaque triangle est visible
    **Output :**
        - j
        - are_triangles_visible : np array de shape (K, 2)
                                  la deuxieme dimension designe la droite ou la gauche
    """
    are_triangles_visible = np.zeros((K, 2), dtype=int)
   
    # -- Etape 1 -- filtration par extraction des faces dans la bonne direction : deja fait
    # -- Etape 2 -- Retro-projection du centre de chaque triangle dans la bonne direction : deja fait
        
    # -- Etape 3 -- Ray-tracing de chaque rayon visant une face
    # creation de la scene de raytracing
    raycasting_scene = o3d.t.geometry.RaycastingScene()
    vertices_tensor = o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32)
    triangles_tensor = o3d.core.Tensor(triangles, dtype=o3d.core.Dtype.UInt32)
    raycasting_scene.add_triangles(vertices_tensor, triangles_tensor)

    # gauche
    # mask pour ne garder que les triangles bien orientes
    for cam in ["l", "r"] :
        idx_cam = 0 if cam == "l" else 1
        rays_to_faces = r0_rd_for_good_p[j, :, :6] if cam == "l" else r0_rd_for_good_p[j, :, 6:]        
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
                are_triangles_visible[original_face_idx, idx_cam] = 1

    return j, are_triangles_visible

def _build_Mpj_cam(mesh, K, N, r0_rd_for_good_p) :
    """
    **Output** :
        - Mpj_cam : (K, N, 2) le tenseur de visibilite.
                    Mpj_cam[0] est la matrice de visibilite gauche, Mpj_cam[1] la droite
    """
    Mpj_cam = np.full((K, N, 2), fill_value=False, dtype=bool)
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    args = list(range(N))
    _f = partial(
                _get_visible_faces,
                K=K,
                vertices=vertices,
                triangles=triangles,
                r0_rd_for_good_p=r0_rd_for_good_p
            )
    with Pool(24) as p:
        for j, are_triangles_visible in tqdm(p.imap_unordered(
            _f,
            args,
            chunksize=1
        ), total=N) :
            Mpj_cam[:, j, 0] = are_triangles_visible[:, 0]
            Mpj_cam[:, j, 1] = are_triangles_visible[:, 1]
    return Mpj_cam

# Cette partie du code nettoie le mesh pour ne garder que les faces visibles. 
# Mpj_cam calcule precedemment est utilise

def _clean_mesh(original_mesh, old_Mpj_cam) :
    """
    Sert a nettoyer le mesh original, en supprimant les faces qui ne sont visibles sur aucune vue, 
    avec le critere qui a ete donne (cos_theta_max) dans build_r0_rd_for_good_p
    **Output :**
        - clean_mesh : le mesh ou les triangles non visibles sur les vues ont ete supprimes
        - clean_Mpj_cam : le nouveau tenseur des vues Mpj_cam
        - visible_faces_mask : le mask pour filtrer les nouveaux indices dans un tenseur
                               indexe sur les anciens indices
    """
    visible_faces_mask = np.any(old_Mpj_cam, axis=(1, 2))
    new_Mpj_cam = old_Mpj_cam[visible_faces_mask]
    triangles = np.asarray(original_mesh.triangles)
    vertices = np.asarray(original_mesh.vertices)
    new_triangles = triangles[visible_faces_mask]
    used_vertices_idx = np.unique(new_triangles)

    old_to_new_idx = np.zeros(vertices.shape[0], dtype=np.int32)
    old_to_new_idx[used_vertices_idx] = np.arange(len(used_vertices_idx))
    remapped_triangles = old_to_new_idx[new_triangles]

    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(vertices[used_vertices_idx])
    new_mesh.triangles = o3d.utility.Vector3iVector(remapped_triangles)
    new_mesh.compute_vertex_normals()
    return new_mesh, new_Mpj_cam, visible_faces_mask

def reconstruct_visible_mesh(mesh,
                             are_triangles_visible) :
    """ POUR LA VISUALISATION AU DEBUG
    Reconstruit a partrir du tableau des faces visibles, le mesh contenant uniquement les faces visibles 
    - are_triangle_visible : bool np array de shape (K,) 
    """
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    visible_faces_indices = np.where(are_triangles_visible == True)[0]
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

# Cette partie du code calcule le cout d'une vue donnee pour une face donnee.
# Ce cout est calcule uniquement sur les faces visibles. 
# ce cout depend du rayon emanant de la face vers la vue. Aussi, r0_rd_for_good_p 
# est utilise pour ce calcul 

@njit(error_model="numpy", cache=True)
def _cost_func(distance, cos_angle, epsilon=0.5) :
    """
    Donne le cout d'une vue regardant la cible depuis une distance donnee
    sous un angle donne. 
    /! fait un calcul simple sans tenir compte de la vraie visibilite. Utiliser Mpj
    **Input :**
        - distance : distance de la camera a la face
        - cos_angle : cosinus de l'angle entre la normale a la face et la normale a la camera
        - float_inf : +infty
    """
    # la camera regarde en direction de l'image donc cos_angle est dans [-1, 0[
    # si cos_angle vaut -1, la camera est pile en face de l'image
    # plus cos_angle est petit plus la vue est de qualite
    # donc plus cos_angle est grand plus la vue est mauvaise
    return distance * epsilon *(1.1+cos_angle)**2

@njit(error_model="numpy", cache=True)
def _cost_param(rdG, rdD, n_p, n_cam, delta=100) :
    """
    **Input :**
        - rdG, rdD : vecteurs directeurs des rayons vers les images gauche et droite
        - n_p : normale a la face dans le repere monde
        - n_cam : normale a la camera par rapport a laquelle le cout est calcule
                  dans le repere monde
        - delta : distance entre les deux cameras (en mm)
    **Output :**
        - distance : distance (en mm) entre la camera et la face
        - cos_angle : cosinus de l'angle entre la normale de la face et la normale 
                      de la camera
    """
    cos_alpha = np.dot(rdD, rdG)    # theta : angle entre l'objet et les deux cam (norm(rd) = 1)
    distance = (delta * cos_alpha) / (2 * (1 - cos_alpha**2))
    cos_angle = np.dot(n_cam, n_p)
    return distance, cos_angle

@njit(error_model="numpy", cache=True)
def _cost_from_view(args, K, Mpj_cam, r0_rd_for_good_p, normals, n_l, n_r, float_inf) :
    """
    Calcule le cout de chaque pixel visible pour la vue p (images droites et gauches)
    **Output :** :
        - cost_view_j : (K, 2)
    """
    j, rot_j, _ = args
    n_l_monde = rot_j.T @ n_l 
    n_r_monde = rot_j.T @ n_r # /! VERIFIER QUE C'EST LA BONNE FORMULE
    cost_view_j = np.full((K, 2), fill_value=float_inf)
    for p in range(K) :
        rdG = r0_rd_for_good_p[j, p, 3:6]
        rdD = r0_rd_for_good_p[j, p, 9:12]
        n_p = normals[p]
        if Mpj_cam[p, j, 0] == True : # gauche
            distance, cos_angle = _cost_param(rdG, rdD, n_p, n_l_monde)
            cost_view_j[p, 0] = _cost_func(distance, cos_angle)
        if Mpj_cam[p, j, 1] == True : # doite
            distance, cos_angle = _cost_param(rdG, rdD, n_p, n_r_monde)
            cost_view_j[p, 1] = _cost_func(distance, cos_angle)
    return j, cost_view_j

def _build_Wpj_cam(K, N, normals,
                   rot_images, t_images, n_l, n_r,
                   Mpj_cam, r0_rd_for_good_p,
                   float_inf) :
    """
    Construit le tenseur des couts individuels. Le mesh en entree est suppose nettoye, 
    ainsi que son tenseur des vues Mpj_cam et des rayons r0_rd_for_good_p.
    **Output :**
        - Wpj_cam : float np array (K, N, 2)
                    Wpj_cam[p, j, 0] donne le cout de la vue j gauche sur la face p
                    Wpj_cam[p, j, 1] de la vue droite
                    Si la face en question n'est pas visible d'apres Mpj_cam, le poids
                    sera egal a float_inf

    """
    Wpj_cam = np.full((K, N, 2), fill_value=float_inf, dtype=np.float64)
    args = [(j, rot_images[j], t_images[j]) for j in range(N)]
    with Pool(24) as p:
        for j, cost_view_j in tqdm(p.imap_unordered(
            partial(_cost_from_view,
                    K=K,
                    Mpj_cam=Mpj_cam,
                    r0_rd_for_good_p=r0_rd_for_good_p,
                    normals=normals,
                    n_l=n_l,
                    n_r=n_r,
                    float_inf=float_inf
            ), 
            args,
            chunksize=1
        ), total=N) :
            Wpj_cam[:, j, 0] =  cost_view_j[:, 0]
            Wpj_cam[:, j, 1] =  cost_view_j[:, 1]
    return Wpj_cam

# Le main qui utilise toutes les fonctions precedentes pour construire les tenseurs Mpj_cam et Wpj_cam

def clean_and_build_Mpj_Wpj_cam(original_mesh, 
                                N, rot_images, t_images, 
                                n_l, n_r, 
                                cos_theta_max, float_inf) :
    """
    Effectue simultanement la construction des matrices de visibilite et de cout M et W, et nettoie le mesh
    en supprimant les faces non visibles avec le critere cos_theta_max.
    **Output :**
        - mesh_clean : le mesh nettoye
        - Mpj_cam : bool np array (K, N, 2)
        - Wpj cam : float np array (K, N, 2)
    """
    original_mesh.compute_triangle_normals()
    original_triangles = np.asarray(original_mesh.triangles)
    original_vertices = np.asarray(original_mesh.vertices)
    original_triangle_normals = np.asarray(original_mesh.triangle_normals)
    original_K = len(original_triangles)

    print("Retroprojection du centre de chaque face vers chaque vue")
    original_r0_rd_for_good_p = build_r0_rd_for_good_p(original_K, N, original_vertices, 
                                                       original_triangles, original_triangle_normals, 
                                                       rot_images, t_images, n_l, n_r, cos_theta_max)
    
    print("Raycasting des rayons obtenus pour generer la matrice originale des vues")
    Mpj_cam = _build_Mpj_cam(original_mesh, original_K, N, original_r0_rd_for_good_p)

    print("Nettoyage du mesh et conservation des faces visibles")
    mesh_clean, Mpj_cam_clean, visible_faces_mask = _clean_mesh(original_mesh, Mpj_cam)

    mesh_clean.compute_triangle_normals()
    triangles = np.asarray(mesh_clean.triangles)
    triangle_normals = np.asarray(mesh_clean.triangle_normals)
    K = len(triangles)

    clean_r0_rd_for_good_p = original_r0_rd_for_good_p[:, visible_faces_mask, :]

    print("Construction du tenseur des couts individuels sur le mesh nettoye")
    Wpj_cam = _build_Wpj_cam(K, N, triangle_normals, rot_images, t_images, n_l, n_r, Mpj_cam_clean, clean_r0_rd_for_good_p, float_inf)

    return mesh_clean, Mpj_cam_clean, Wpj_cam

if __name__ == "__main__" :

    multiprocessing.set_start_method('spawn')

    from calib_luca import n_l, n_r

    N = 52

    # ouverture des transformations image
    rot_images = []
    t_images = []
    for j in range(N) :
        rot, t = get_image_data(j+1)
        rot_images.append(rot)
        t_images.append(t)
    rot_images = np.array(rot_images)
    t_images = np.array(t_images)

    original_mesh = o3d.io.read_triangle_mesh("ply/mesh_cailloux_luca_high.ply")
   
    mesh_clean, Mpj_cam, Wpj_cam = clean_and_build_Mpj_Wpj_cam(original_mesh, N, rot_images, t_images, n_r, n_l, -0.2, 1e9)
    
    np.save("tensors/Mpj_cam.npy", Mpj_cam)
    np.save("tensors/Wpj_cam.npy", Wpj_cam)


    # are_triangle_visible_l = Mpj_cam[:, 35, 0]
    # visible_mesh = reconstruct_visible_mesh(original_mesh, are_triangle_visible_l)
    # o3d.visualization.draw_geometries([visible_mesh])
    o3d.visualization.draw_geometries([mesh_clean])
    o3d.io.write_triangle_mesh("ply/LOW_CLEAN_MESH.ply", mesh_clean)
    print(f"mesh_clean.triangles : ({np.asarray(mesh_clean.triangles).shape}), \
          Mpj_cam : ({Mpj_cam.shape}), Wpj_cam : ({Wpj_cam.shape})")
    are_triangle_visible_l = Mpj_cam[:, 35, 0]
    visible_mesh = reconstruct_visible_mesh(mesh_clean, are_triangle_visible_l)
    o3d.visualization.draw_geometries([visible_mesh])