import numpy as np
import calib_luca as param_calib
import json
import cv2
from numba import njit



@njit(error_model="numpy")
def inverse_3x3(A):
    """
    Inverse explicitement une matrice 3x3 en utilisant la regle de Sarrus.
    A : liste de listes 3x3 ou numpy 3x3
    Retourne : matrice inverse 3x3 sous forme de liste de listes
    """
    a, b, c = A[0]
    d, e, f = A[1]
    g, h, i = A[2]
    det = a*e*i + b*f*g + c*d*h - c*e*g - b*d*i - a*f*h    # Determinant par la regle de Sarrus
    if abs(det) < 1e-12:
        raise ValueError("La matrice n'est pas inversible")
    cofactor = [ # Comatrice (mineur-cofacteur)
        [(e*i - f*h), -(d*i - g*f), (d*h - e*g)],
        [-(b*i - h*c), (a*i - c*g), -(a*h - g*b)],
        [(b*f - e*c), -(a*f - d*c), (a*e - b*d)]
    ]
    adjugate = [[cofactor[j][i] for j in range(3)] for i in range(3)] # Transposee de la comatrice
    invA = [[adjugate[i][j] / det for j in range(3)] for i in range(3)]
    return np.array(invA)

@njit(error_model="numpy")
def sqrt_newton(x, tol=1e-10, max_iter=15):
    """
    - x : float strictement positif
    - tol : tolerance sur l'erreur relative
    - max_iter : nombre maximum d'iterations
    Retourne :
    - sqrt(x) approximee
    """
    if x < 0:
        raise ValueError("x doit être positif")
    if x == 0:
        return 0.0
    # Initial guess : bonne heuristique rapide
    guess = x if x >= 1 else 1.0
    for _ in range(max_iter):
        new_guess = 0.5 * (guess + x / guess)
        if abs(new_guess - guess) < tol * new_guess:
            return new_guess
        guess = new_guess
    return guess

@njit(error_model="numpy")
def fast_matvec_3x3(A, v):
    """
    Multiplie explicitement une matrice 3x3 (A) avec un vecteur 3 (v)
    """
    res = np.zeros(3)
    res[0] = A[0, 0]*v[0] + A[0, 1]*v[1] + A[0, 2]*v[2]
    res[1] = A[1, 0]*v[0] + A[1, 1]*v[1] + A[1, 2]*v[2]
    res[2] = A[2, 0]*v[0] + A[2, 1]*v[1] + A[2, 2]*v[2]
    return res

@njit(error_model="numpy")
def fast_trace_refract_ray(pixel_coord, air_K, distance, normal, thickness, eta_glass, eta_water):
    """
    assumes that `pixel_coord` was already undistorted
    """
    camera_rd = fast_matvec_3x3(inverse_3x3(air_K),np.array([pixel_coord[0], pixel_coord[1], 1]))
    camera_rd /= sqrt_newton((camera_rd**2).sum(), max_iter=10)
    # first intersection and refract, from inside the tube (air) to inside the flat port
    c = normal[0]*camera_rd[0] + normal[1]*camera_rd[1] + normal[2]*camera_rd[2]
    r = 1 / eta_glass
    glass_rd = r * camera_rd - (r * c - sqrt_newton(1 - r*r * (1 - c*c))) * normal
    glass_rd /= sqrt_newton((glass_rd**2).sum(), max_iter=10)
    camera_rd_times_normal = normal[0]*camera_rd[0] + normal[1]*camera_rd[1] + normal[2]*camera_rd[2]
    glass_ro = camera_rd * (distance * normal[2]) / camera_rd_times_normal
    # second intersection and refraction, from inside the flat port towards the water
    c = normal[0]*glass_rd[0] + normal[1]*glass_rd[1] + normal[2]*glass_rd[2]
    r = eta_glass / eta_water
    water_rd = r * glass_rd - (r * c - sqrt_newton(1 - r*r * (1 - c*c))) * normal
    water_rd /= sqrt_newton((water_rd**2).sum(), max_iter=10)
    glass_rd_times_normal = normal[0]*glass_rd[0] + normal[1]*glass_rd[1] + normal[2]*glass_rd[2]
    vec_temp = (np.array([0, 0, distance]) + thickness * normal - glass_ro)
    vec_temp_times_normal = normal[0]*vec_temp[0] + normal[1]*vec_temp[1] + normal[2]*vec_temp[2]
    water_ro = glass_ro + glass_rd * (vec_temp_times_normal / glass_rd_times_normal)
    return water_ro, water_rd

@njit(error_model="numpy")
def trace_refract_ray(pixel_coord, air_K, distance, normal, thickness, eta_glass, eta_water):
    """
    assumes that `pixel_coord` was already undistorted
    """
    camera_rd = np.linalg.inv(air_K) @ np.array([pixel_coord[0], pixel_coord[1], 1])
    camera_rd /= np.linalg.norm(camera_rd)
    # first intersection and refract, from inside the tube (air) to inside the flat port
    c = normal @ camera_rd
    r = 1 / eta_glass
    glass_rd = r * camera_rd - (r * c - np.sqrt(1 - r*r * (1 - c*c))) * normal
    glass_rd /= np.linalg.norm(glass_rd)
    glass_ro = camera_rd * (((np.array([0, 0, distance])) @ normal) / (camera_rd @ normal))
    # second intersection and refraction, from inside the flat port towards the water
    c = normal @ glass_rd
    r = eta_glass / eta_water
    water_rd = r * glass_rd - (r * c - np.sqrt(1 - r*r * (1 - c*c))) * normal
    water_rd /= np.linalg.norm(water_rd)
    water_ro = glass_ro + glass_rd * (((np.array([0, 0, distance]) + thickness * normal - glass_ro) @ normal) / (glass_rd @ normal))

    return water_ro, water_rd

@njit(error_model="numpy")
def r0_rd(Y, R=np.eye(3), t=np.zeros((3,)), cam="l") :
    """
    Donne r0 et rd pour un point Y = [a, b] dans l'image
    /!\ -- R et t sont la matrice de rotation et le vecteur de translation 
    du repere monde vers le repere gauche
    """
    t.reshape((3,))
    if cam == "l" :
        r0_cam, rd_cam = trace_refract_ray(Y, param_calib.air_K_l, param_calib.D_l, param_calib.n_l, param_calib.THICKNESS, param_calib.ETA_GLASS, param_calib.ETA_WATER)
        # passage du repere gauche au repere monde
        r0 = R.T @ r0_cam - R.T @ t
        rd = R.T @ rd_cam
        # r0 = R @ r0_cam + t
        # rd = R @ rd_cam
    elif cam == "r" :
        r0_cam, rd_cam = trace_refract_ray(Y, param_calib.air_K_r, param_calib.D_r, param_calib.n_r, param_calib.THICKNESS, param_calib.ETA_GLASS, param_calib.ETA_WATER)
        # passage du repere droit au repere monde
        R_DG = param_calib.R_DG
        t_DG = param_calib.t_DG
        r0 = R.T @ (R_DG.T @ r0_cam - R_DG.T @ t_DG) - R.T @ t
        rd = R.T @ R_DG.T @ rd_cam
    else :
        raise ValueError(f"Incorrect value : {cam}. 'cam' must be 'l' or 'r'")
    rd_norm = ((rd) ** 2).sum() ** 0.5
    return r0, rd/rd_norm

@njit(error_model="numpy")
def FASTr0_rd(Y, R=np.eye(3), t=np.zeros((3,)), cam="l") :
    """
    Donne r0 et rd pour un point Y = [a, b] dans l'image
    /!\ -- R et t sont la matrice de rotation et le vecteur de translation 
    du repere monde vers le repere gauche
    """
    t.reshape((3,))
    if cam == "l" :
        r0_cam, rd_cam = fast_trace_refract_ray(Y, param_calib.air_K_l, param_calib.D_l, param_calib.n_l, param_calib.THICKNESS, param_calib.ETA_GLASS, param_calib.ETA_WATER)
        # passage du repere gauche au repere monde
        r0 = fast_matvec_3x3(R.T, r0_cam) - R.T @ t
        rd = fast_matvec_3x3(R.T, rd_cam)
        # r0 = R @ r0_cam + t
        # rd = R @ rd_cam
    elif cam == "r" :
        r0_cam, rd_cam = fast_trace_refract_ray(Y, param_calib.air_K_r, param_calib.D_r, param_calib.n_r, param_calib.THICKNESS, param_calib.ETA_GLASS, param_calib.ETA_WATER)
        # passage du repere droit au repere monde
        R_DG = param_calib.R_DG
        t_DG = param_calib.t_DG
        r0 = R.T @ (fast_matvec_3x3(R_DG.T,r0_cam) - fast_matvec_3x3(R_DG.T,t_DG)) - fast_matvec_3x3(R.T,t)
        rd = fast_matvec_3x3(R.T, fast_matvec_3x3(R_DG.T, rd_cam))
    else :
        raise ValueError(f"Incorrect value : {cam}. 'cam' must be 'l' or 'r'")
    rd_norm = ((rd) ** 2).sum() ** 0.5
    return r0, rd/rd_norm

@njit(error_model="numpy")
def distance_X_to_D_r0_rd(X, r0, rd, decalage_erreur=0):
    """
    Calcule d(X,D) ou X(x,y,z) est un point du mesh 3D, et D = {r0 + t * rd | d in IR} rayon lumineux parametre par r0 et rd
    X, r0 et rd sont des tableaux np de 3 floats
    """
    x,y,z = X[0], X[1], X[2]
    r0x, r0y, r0z = r0[0], r0[1], r0[2]
    rdx, rdy, rdz = rd[0], rd[1], rd[2]
    
    return np.array([
        rdz*(y-r0y) - rdy*(z-r0z) - decalage_erreur,
        rdx*(z-r0z) - rdz*(x-r0x) - decalage_erreur,
        rdy*(x-r0x) - rdx*(y-r0y) - decalage_erreur ]).flatten()

def get_image_data(image_id=26):
    """
    /! image_id dans [0, N[
    Renvoie (rot, t) ou rot est la matrice de rotation de la camera, et t son vecteur de translation
    """
    image_id += 1
    # position de l'image
    with open("absolute_transforms_luca.json") as f :
        data = json.load(f)
        image_id=str(image_id)
        r = np.array(data["0"][image_id][0], dtype=np.float64)
        t = np.array(data["0"][image_id][1], dtype=np.float64)
        rot, _ = cv2.Rodrigues(r)
    return (rot, t)


def closest_point_to_two_lines(ro1, rd1, ro2, rd2):
    b = ro2 - ro1

    d1_cross_d2 = np.cross(rd1, rd2)
    cross_norm2 = d1_cross_d2[0] * d1_cross_d2[0] + d1_cross_d2[1] * d1_cross_d2[1] + d1_cross_d2[2] * d1_cross_d2[2]

    t1 = np.linalg.det(np.array([
        [b[0], rd2[0], d1_cross_d2[0]],
        [b[1], rd2[1], d1_cross_d2[1]],
        [b[2], rd2[2], d1_cross_d2[2]]
    ])) / np.maximum(0.00001, cross_norm2)

    t2 = np.linalg.det(np.array([
        [b[0], rd1[0], d1_cross_d2[0]],
        [b[1], rd1[1], d1_cross_d2[1]],
        [b[2], rd1[2], d1_cross_d2[2]]
    ])) / np.maximum(0.00001, cross_norm2)

    p1 = ro1 + t1 * rd1
    p2 = ro2 + t2 * rd2

    return (p1 + p2) / 2.0, np.linalg.norm(p2 - p1)

if __name__ == "__main__" :
    
    import time
    k_time = 100000

    time1 = time.time()
    for _ in range(k_time) :
        k = r0_rd(np.array([np.random.randint(0,3000) ,np.random.randint(0,2000)], dtype=np.float64))
    print(f"numpy r0_rd : {time.time()-time1}")

    time2 = time.time()
    for _ in range(k_time) :
        k = FASTr0_rd(np.array([np.random.randint(0,3000) ,np.random.randint(0,2000)], dtype=np.float64))
    print(f"python optimized r0_rd : {time.time()-time2}")

