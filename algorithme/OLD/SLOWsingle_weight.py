import numpy as np
import open3d as o3d
from tqdm import tqdm
import calib_luca as param_calib
from utils import get_image_data
from backprojection import back_projeter

from numba import njit

@njit(error_model="numpy", cache=True)
def _cost_func(distance, cos_angle, float_inf, epsilon=0.5) :
    """
    Donne le cout d'une vue regardant la cible depuis une distance donnee
    sous un angle donne. 
    /!\ fait un calcul simple sans tenir compte de la vraie visibilite. Utiliser Mpj
    **Input :**
        - distance, cos_angle
        - float_inf
    """
    # la camera regarde en direction de l'image donc cos_angle est dans [-1, 0[
    # si cos_angle vaut -1, la camera est pile en face de l'image
    # plus cos_angle est petit plus la vue est de qualite
    # donc plus cos_angle est grand plus la vue est mauvaise
    return distance * 0.5*(1.1+cos_angle)**2


def _cost_face_cam(X_p, n_p, rot, t, cam, cost_func, float_inf) :
    """
    Donne le poids d'une face vue depuis une cam donnee
    - X_p : le centre de la face
    - n_p : la normale a la face (dans le repere monde)
    """
    # distance entre l'objet et le rig :
    _, r0G, rdG = back_projeter(X_p, rot, t, "l")
    _, r0D, rdD = back_projeter(X_p, rot, t, "r")
    delta = 100                     # distance entre les deux cam dans le rig (en mm)
    cos_theta = np.dot(rdD, rdG)    # theta : angle entre l'objet et les deux cam (norm(rd) = 1)
    distance = (delta * cos_theta) / (2 * (1 - cos_theta**2))
    # angle entre la face et la cam
    n_cam = param_calib.n_l if cam=="l" else param_calib.n_r
    n_monde = rot.T @ n_cam    # vecteur normal a la cam dans le repere monde
    cos_angle = np.dot(n_monde, n_p)

    return cost_func(distance, cos_angle, float_inf)

def build_Wpj(K, N, triangles, vertices, normals, Mpj, rot_images, t_images, cam="l", float_inf=1e20) :
    """
    Renvoie la matrice de cout Wpj
    Wpj = | +infty si la face p n'est pas visible depuis la vue j
          | le cout de la vue j pour la face p sinon (nombre positif) 
    """
    Wpj = np.full((K, N), float_inf)
    for p in tqdm(range(K)) :
        tri = triangles[p]
        X_p = vertices[tri].mean(axis=0)
        n_p = normals[p, :]
        for j in range(N) :
            if Mpj[p, j] == True :  # si la face est visible
                Wpj[p, j] = _cost_face_cam(X_p, n_p, rot_images[j], t_images[j], cam, _cost_func, float_inf)
    return Wpj
        


if __name__ == "__main__" :
    mesh = o3d.io.read_triangle_mesh("ply/mesh_cailloux_luca_CLEAN.ply")
    mesh.compute_triangle_normals()
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    triangle_normals = np.asarray(mesh.triangle_normals)
    K = len(triangles)
    N = 52

    rot_images = []
    t_images = []
    for j in range(N) :
        rot, t = get_image_data(j+1)
        rot_images.append(rot)
        t_images.append(t)
    rot_images = np.array(rot_images)
    t_images = np.array(t_images)
    Mpj = np.load("tensors/Mpj.npy")
    Wpj = build_Wpj(K, N, triangles, vertices, triangle_normals, Mpj, rot_images, t_images, "l", 1e20)
    print(Wpj.shape)
    print(Mpj[0, :])
    print(Wpj[0,:])
    np.save("tensors/Wpj.npy", Wpj)