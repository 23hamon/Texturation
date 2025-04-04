import numpy as np
import open3d as o3d
import cv2
from scipy.optimize import minimize
from utils import r0_rd, distance_X_to_D_r0_rd

def back_projeter(X, image, max_cost=None):
    """
    Renvoie Y, la projection inverse sur l'image d'un point X sur le mesh 3D
    **Input :**
        - X : numpy array [x, y, z], coordonnees dans $\mathbb{R^3}$
        - image : objet image
        - max_cost : cout maximal au dessus duquel rien ne sera renvoye (point hors de l'image)
    **Output :**
        - Y_best : numpy array [x, y], coordonnees en pixel du point sur l'image
    """
    def f(Y):  # Fonction à minimiser
        r0, rd = r0_rd(Y)
        return distance_X_to_D_r0_rd(X, r0, rd)
    
    height, width = image.shape[:2]
    Y0 = np.array([width // 2, height // 2], dtype=np.float64)  # Point de départ
    # Minimisation
    res = minimize(f, Y0, method="L-BFGS-B", bounds=[(0, width), (0, height)])
    if max_cost:
        if res.fun < max_cost:
            return res.x
        else:
            return None
    else:
        return res.x


#boucle sur tout le maillage
mesh = o3d.io.read_triangle_mesh("fichiers_ply/mesh_cailloux_min.ply")
image_path = "downsampled/scene_l_0026.jpeg"
image = cv2.imread(image_path)

points = np.asarray(mesh.vertices)

# correspondances 3D-2D
correspondances = []

for i, X_test in enumerate(points):
    X_test = points[i]
    Y_best = back_projeter(X_test, image)
    correspondances.append((X_test, Y_best))

correspondances_np = np.array(correspondances, dtype=object)

for i, (vertex_3d, uv) in enumerate(correspondances[:10]):
    print(f"V {i}: 3D={vertex_3d} -> image_pixel={uv}")
