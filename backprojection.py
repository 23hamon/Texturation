from utils import r0_rd, distance_X_to_D_r0_rd
import open3d as o3d
import numpy as np
import cv2

from scipy.optimize import minimize, least_squares
import time
import json

def back_projeter(X,
                  image_height,
                  image_width,
                  R_cam=np.eye(3),      # rotation de la camera
                  t_cam=np.zeros((3,)), # tranlation de la camera
                  max_cost=None) :
    """
    Renvoie Y, la projection inverse sur l'image d'un point X sur le mesh 3D
    **Input :**
        - X : numpy array [x, y, z], coordonnees dans $\mathbb{R^3}$
        - image_height, image_width : format de l'image
        - max_cost : cout maximal au dessus duquel rien ne sera renvoye (point hors de l'image)
    **Output :**
        - Y_best : numpy array [x, y], coordonnees en pixel du point sur l'image
        - r0 et rd, le rayon qui emane de Y
    """
    r0, rd = None, None
    def f(Y) :     # fonction a minimiser
        nonlocal r0, rd
        r0, rd = r0_rd(Y, R_cam, t_cam)
        cost = distance_X_to_D_r0_rd(X, r0, rd)
        return cost
    Y0 = np.array([image_width // 2, image_height // 2], dtype=np.float64)  # point de depart
    # minimisation
    res = least_squares(f, Y0, loss="linear", verbose=2)
    #res = minimize(f, Y0, method="Nelder-Mead", bounds=[(0, image_width), (0, image_height)])
    #bounds=([0,0], [image_width, image_height])
    if max_cost:
        if res.fun < max_cost :
            return res.x, r0, rd
        else : 
            return None
    else : 
        return res.x, r0, rd
    
if __name__ == "__main__" :
    

    mesh = o3d.io.read_triangle_mesh("fichiers_ply/mesh_cailloux_low.ply")

    # chargement de l'image
    image_id = 29
    # position de l'image
    with open("data/absolute_transforms.json") as f :
        data = json.load(f)
        t_rot = np.array(data["0"][str(image_id)]).flatten()
        rot,_ = cv2.Rodrigues(np.array(t_rot[:3], dtype=np.float64))
        t = np.array(t_rot[3:], dtype=np.float64)
    print(rot, t)
        
    image_path = f"downsampled/scene_l_00{str(image_id)}.jpeg"
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    points = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    idx = np.random.randint(len(points))
    X_test = points[idx]

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=3)
    sphere.translate(X_test)
    sphere.paint_uniform_color([1, 0, 0])

    Y_best, r0, rd = back_projeter(X_test, h, w, rot, t)
    Y_best = tuple(map(int, Y_best))  
    print(r0, rd)
    # tracer le rayon
    ligne = o3d.geometry.LineSet()
    ligne.points = o3d.utility.Vector3dVector([r0-1200*rd, r0 -500 * rd])
    ligne.lines = o3d.utility.Vector2iVector([[0, 1]])

    lignes_coins = []
    for x in [0, 2999] :
        for y in [0, 1999] :
            Y= np.array([x, y], dtype=np.float64)
            r0_coin, rd_coin = r0_rd(Y, rot, t)
            ligne_coin = o3d.geometry.LineSet()
            ligne_coin.points = o3d.utility.Vector3dVector([r0_coin-1200*rd_coin, r0_coin -500 * rd_coin])
            ligne_coin.lines = o3d.utility.Vector2iVector([[0, 1]])
            lignes_coins.append(ligne_coin)


    image_with_point = image.copy()

    cv2.circle(image_with_point, Y_best, radius=15, color=(0, 0, 255), thickness=-1)
    cv2.imwrite("Projection inverse.jpg", cv2.flip(image_with_point, 1))

    o3d.visualization.draw_geometries([mesh, sphere, ligne]+lignes_coins, window_name="Mesh avec point sélectionné")

    """
    print(f"Nombre de triangles : {len(triangles)}")
    n_it = 1000
    start_time = time.time()
    for _ in range(n_it):
        idx = np.random.randint(len(points))
        X_test = points[idx]
        Y_best, _, _ = back_projeter(X_test, h, w)
    total_time = time.time() - start_time
    print(f"temps de back_projection moyen: {total_time/n_it} sec")
    print(f"temps estimé total : {len(triangles) * total_time/n_it} sec")
    """