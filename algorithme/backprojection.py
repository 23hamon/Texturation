__author__ = "Jules Imbert"
__date__ = "2025-05-06"

from utils import distance_X_to_D_r0_rd, get_image_data, FASTr0_rd
import numpy as np
from scipy.optimize import least_squares

class CostFunction:
    def __init__(self, X, R_cam, t_cam, cam):
        self.X = X
        self.R_cam = R_cam
        self.t_cam = t_cam
        self.cam = cam
        self.r0 = None
        self.rd = None
    def __call__(self, Y):
        self.r0, self.rd = FASTr0_rd(Y, self.R_cam, self.t_cam, self.cam)
        return distance_X_to_D_r0_rd(self.X, self.r0, self.rd)

def back_projeter(X, R_cam=np.eye(3), t_cam=np.zeros((3,)), cam="l", max_cost=None):
    cost_func = CostFunction(X, R_cam, t_cam, cam)
    Y0 = np.array([1500., 1000.], dtype=np.float32)
    res = least_squares(cost_func, Y0, loss="linear", verbose=0, max_nfev=14)

    if max_cost and res.fun >= max_cost:
        return None

    return res.x, cost_func.r0, cost_func.rd


if __name__ == "__main__" :
    import open3d as o3d
    import time
    mesh = o3d.io.read_triangle_mesh("fichiers_ply/mesh_cailloux_luca_LOW.ply")
    image_id = 10
    rot,t = get_image_data(image_id)
    k_time = 1000
    time1 = time.time()
    for _ in range(k_time) :
        points = np.asarray(mesh.vertices)
        X = points[np.random.randint(len(points)) ]
        Y = back_projeter(X, rot, t, "l") 
    print(f"FAST r0_rd : {time.time()-time1}")
