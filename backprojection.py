from utils import r0_rd, distance_X_to_D_r0_rd
import open3d as o3d
import numpy as np
import cv2

from scipy.optimize import minimize

def back_projeter(X,
                  image,
                  max_cost=None) :
    """
    Renvoie Y, la projection inverse sur l'image d'un point X sur le mesh 3D
    **Input :**
        - X : numpy array [x, y, z], coordonnees dans $\mathbb{R^3}$
        - image : objet image
        - max_cost : cout maximal au dessus duquel rien ne sera renvoye (point hors de l'image)
    **Output :**
        - Y_best : numpy array [x, y], coordonnees en pixel du point sur l'image
    """
    def f(Y) :     # fonction a minimiser
        r0, rd = r0_rd(Y)
        return distance_X_to_D_r0_rd(X, r0, rd)
    height, width = image.shape[:2] 
    Y0 = np.array([width // 2, height // 2], dtype=np.float64)  # point de depart
    # minimisation
    res = minimize(f, Y0, method="L-BFGS-B", bounds=[(0, width), (0, height)])
    if max_cost:
        if res.fun < max_cost :
            return res.x
        else : 
            return None
    else : 
        return res.x
    
if __name__ == "__main__" :

    mesh = o3d.io.read_triangle_mesh("fichiers_ply/mesh_cailloux.ply")
    image_path = "downsampled/scene_l_0026.jpeg"
    image = cv2.imread(image_path)

    points = np.asarray(mesh.vertices)
    idx = np.random.randint(len(points))
    X_test = points[idx]

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=3)
    sphere.translate(X_test)
    sphere.paint_uniform_color([1, 0, 0])

    Y_best = back_projeter(X_test, image)
    Y_best = tuple(map(int, Y_best))  
    image_with_point = image.copy()

    cv2.circle(image_with_point, Y_best, radius=15, color=(0, 0, 255), thickness=-1)
    cv2.imwrite("Projection inverse.jpg", cv2.flip(image_with_point, 1))

    o3d.visualization.draw_geometries([mesh, sphere], window_name="Mesh avec point sélectionné")

    