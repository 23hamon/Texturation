import numpy as np
from utils import r0_rd, get_image_data
from scipy.optimize import least_squares
import cv2
import open3d as o3d

def intersect_wiki(r0G, rdG, r0D, rdD) :
    """ Renvoie X, le pseudo-point d'intersection entre les deux droites et delta la distance"""
    nG = rdG - (np.dot(rdG, rdD) / np.dot(rdD, rdD)) * rdD
    nD = rdD - (np.dot(rdG, rdD) / np.dot(rdG, rdG)) * rdG
    A = r0D - r0G
    bG = rdG * (np.dot(A, nG) / np.dot(rdG, nG)) + r0G
    bD = rdD * (-np.dot(A, nD) / np.dot(rdD, nD)) + r0D
    delta = np.linalg.norm(bD-bG)
    return ((bG + bD) / 2, delta)


def intersection_lr(ro1, rd1, ro2, rd2) :

    b = ro2 - ro1

    d1_cross_d2_x = rd1[1] * rd2[2] - rd1[2] * rd2[1]
    d1_cross_d2_y = rd1[2] * rd2[0] - rd1[0] * rd2[2]
    d1_cross_d2_z = rd1[0] * rd2[1] - rd1[1] * rd2[0]
    cross_norm2 = d1_cross_d2_x * d1_cross_d2_x + d1_cross_d2_y * d1_cross_d2_y + d1_cross_d2_z * d1_cross_d2_z
    cross_norm2 = np.maximum(0.0000001, cross_norm2)

    t1 = (b[0] * rd2[1] * d1_cross_d2_z + b[1] * rd2[2] * d1_cross_d2_x + rd2[0] * d1_cross_d2_y * b[2]
            - b[2] * rd2[1] * d1_cross_d2_x - b[0] * d1_cross_d2_y * rd2[2] - b[1] * rd2[0] * d1_cross_d2_z) / cross_norm2

    t2 = (b[0] * rd1[1] * d1_cross_d2_z + b[1] * rd1[2] * d1_cross_d2_x + rd1[0] * d1_cross_d2_y * b[2]
            - b[2] * rd1[1] * d1_cross_d2_x - b[0] * d1_cross_d2_y * rd1[2] - b[1] * rd1[0] * d1_cross_d2_z) / cross_norm2

    p1 = ro1 + t1 * rd1
    p2 = ro2 + t2 * rd2

    return (p1 + p2) / 2

def back_projeter_droite_gauche(X,
                                R_cam=np.eye(3),      # rotation de la camera
                                t_cam=np.zeros((3,)), # translation de la camera
                                w=3000, h=2000) :
    r01, rd1, r02, rd2 = None, None, None, None
    def f(Y_lr) :     # fonction a minimiser
        nonlocal r01, rd1, r02, rd2
        Y_l = Y_lr[:2]
        Y_r = Y_lr[2:]
        r01, rd1 = r0_rd(Y_l, R_cam, t_cam, "l")
        r02, rd2 = r0_rd(Y_r, R_cam, t_cam, "r")
        X_inter, delta = intersect_wiki(r01, rd1, r02, rd2)
        return np.concatenate((X - X_inter, np.array([delta]))).flatten() 
    #Y0 = np.array([image_width // 2, image_height // 2], dtype=np.float64)  # point de depart
    Y0 = np.array([0., 0., 0., 0.], dtype=np.float64)
    # minimisation
    res = least_squares(f, Y0, loss="linear", verbose=0, 
                        bounds=(np.array([0.,0.,0.,0.]), np.array([w, h, w, h])))
    Y_lr = res.x
    Y_l = Y_lr[:2]
    Y_r = Y_lr[2:]
    print(f"distance finale : {res.fun}")
    return Y_l, Y_r, r01, rd1, r02, rd2


# ---- fonction d'affichage de croix 3d
def create_cross(center, size=10, thickness=1, color=[0, 1, 0]):
    cross_parts = []

    # Axe X
    box_x = o3d.geometry.TriangleMesh.create_box(width=size, height=thickness, depth=thickness)
    box_x.translate(np.array(center) - np.array([size/2, thickness/2, thickness/2]))
    box_x.paint_uniform_color(color)
    cross_parts.append(box_x)

    # Axe Y
    box_y = o3d.geometry.TriangleMesh.create_box(width=thickness, height=size, depth=thickness)
    box_y.translate(np.array(center) - np.array([thickness/2, size/2, thickness/2]))
    box_y.paint_uniform_color(color)
    cross_parts.append(box_y)

    # Axe Z
    box_z = o3d.geometry.TriangleMesh.create_box(width=thickness, height=thickness, depth=size)
    box_z.translate(np.array(center) - np.array([thickness/2, thickness/2, size/2]))
    box_z.paint_uniform_color(color)
    cross_parts.append(box_z)

    return cross_parts

if __name__ == "__main__" :
    # Ouverture du mesh, des images, des points
    mesh = o3d.io.read_triangle_mesh("fichiers_ply/mesh_cailloux_luca_high.ply")
    mesh.paint_uniform_color([0.5, 0.5, 0.5])
    mesh.compute_vertex_normals()

    flip = False
    image_id = 26
    r,t = get_image_data(image_id)
    rot, _ =cv2.Rodrigues(r)
    print(f"r = {r}, t = {t}, \n rot = {rot}")

    image_path_l = f"downsampled/scene_l_00{str(image_id)}.jpeg"
    image_path_r = f"downsampled/scene_r_00{str(image_id)}.jpeg"
    image_l = cv2.imread(image_path_l)
    image_r = cv2.imread(image_path_r)
    h, w = image_l.shape[:2]
    print(f"w = {w}, h = {h}")
    points = np.asarray(mesh.vertices)
    
    # trace de X
    idx = np.random.randint(len(points))
    X = points[idx]
    print(f"point selectionne : X = {X} (id : {idx})")
    # sphere = o3d.geometry.TriangleMesh.create_sphere(radius=3)
    # sphere.translate(X)
    # sphere.paint_uniform_color([0,1,0])
    cross = create_cross(X, size=10, thickness=1, color=[1, 0, 0])  # croix rouge

    # retro-projection
    Y_l, Y_r, r0_l, rd_l, r0_r, rd_r = back_projeter_droite_gauche(X, rot, t, w, h)
    print(Y_l, Y_r)
    Y_l = tuple(map(int, Y_l))
    Y_r = tuple(map(int, Y_r))

    # tracer les rayons
    ligne_l = o3d.geometry.LineSet()
    ligne_l.points = o3d.utility.Vector3dVector([r0_l+1200*rd_l, r0_l +500 * rd_l])
    ligne_l.lines = o3d.utility.Vector2iVector([[0, 1]])
    ligne_l.colors = o3d.utility.Vector3dVector([[0, 0, 1]])
    ligne_r = o3d.geometry.LineSet()
    ligne_r.points = o3d.utility.Vector3dVector([r0_r+1200*rd_r, r0_r +500 * rd_r])
    ligne_r.lines = o3d.utility.Vector2iVector([[0, 1]])
    ligne_r.colors = o3d.utility.Vector3dVector([[1, 0, 0]])

    # tracer les coins
    lignes_coins_l = []
    lignes_coins_r = []
    for x in [0, 2999] :
        for y in [0, 1999] :
            Y= np.array([x, y], dtype=np.float64)
            r0_coin_l, rd_coin_l = r0_rd(Y, rot, t, "l")
            r0_coin_r, rd_coin_r = r0_rd(Y, rot, t, "r")
            ligne_coin_l = o3d.geometry.LineSet()
            ligne_coin_l.points = o3d.utility.Vector3dVector([r0_coin_l+1200*rd_coin_l, r0_coin_l +500 * rd_coin_l])
            ligne_coin_l.lines = o3d.utility.Vector2iVector([[0, 1]])
            ligne_coin_l.colors = o3d.utility.Vector3dVector([[0, 0, 1]])
            lignes_coins_l.append(ligne_coin_l)
            ligne_coin_r = o3d.geometry.LineSet()
            ligne_coin_r.points = o3d.utility.Vector3dVector([r0_coin_r+1200*rd_coin_r, r0_coin_r +500 * rd_coin_r])
            ligne_coin_r.lines = o3d.utility.Vector2iVector([[0, 1]])
            ligne_coin_r.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
            lignes_coins_r.append(ligne_coin_r)

    # dessiner sur les images
    image_with_point_l = image_l.copy()
    image_with_point_r = image_r.copy()

    def draw_cross(image, center, color=(0, 255, 0), size=10, thickness=2):
        x, y = map(int, center)
        cv2.line(image, (x - size, y), (x + size, y), color, thickness)  # ligne horizontale
        cv2.line(image, (x, y - size), (x, y + size), color, thickness)

    draw_cross(image_with_point_l, Y_l, color=(0, 255, 0))
    draw_cross(image_with_point_r, Y_r, color=(0, 255, 0))
    
    if flip :
        cv2.imwrite("fichiers_test/Projection inverse_l.jpg", cv2.flip(image_with_point_l, 1))
        cv2.imwrite("fichiers_test/Projection inverse_r.jpg", cv2.flip(image_with_point_r, 1))
    else :
        cv2.imwrite("fichiers_test/Projection inverse_l.jpg",image_with_point_l)
        cv2.imwrite("fichiers_test/Projection inverse_r.jpg", image_with_point_r)

    o3d.visualization.draw_geometries([mesh, ligne_l, ligne_r]+cross+lignes_coins_l+lignes_coins_r, window_name="Mesh avec point sélectionné")
