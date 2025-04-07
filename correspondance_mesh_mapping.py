import trimesh
import xatlas
import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from backprojection import back_projeter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


mesh = trimesh.load_mesh('fichiers_ply/mesh_cailloux_min.ply')
vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
xatlas.export("maillage_avec_uv.obj", mesh.vertices[vmapping], indices, uvs)
image = cv2.imread("downsampled/scene_l_0026.jpeg")


#                                                                   --- fcts ---

def generate_random_point(uv1, uv2, uv3):
    r1 = np.random.random()
    r2 = np.random.random()
    if r1 + r2 > 1:
        r1 = 1 - r1
        r2 = 1 - r2
    
    return r1, r2
    return (1 - r1 - r2) * uv1 + r1 * uv2 + r2 * uv3 #point avec barycentriques 

def bilinear_interpolate(image, x, y):
    x0 = int(np.floor(x))
    x1 = min(x0 + 1, image.shape[1] - 1)
    y0 = int(np.floor(y))
    y1 = min(y0 + 1, image.shape[0] - 1)
    wx = x - x0
    wy = y - y0
    top = (1 - wx) * image[y0, x0] + wx * image[y0, x1]
    bottom = (1 - wx) * image[y1, x0] + wx * image[y1, x1]
    return ((1 - wy) * top + wy * bottom).astype(np.uint8)




#                                                        --- main ---


# generation des points randoms 3d depuis les coordonnées (U,V) de points random du mapping
random_points_2d = []
random_points_3d = []

for face in indices:
    uv1, uv2, uv3 = uvs[face]
    for _ in range(3):
        r1, r2 = generate_random_point(uv1, uv2, uv3)
        pt_uv = (1 - r1 - r2) * uv1 + r1 * uv2 + r2 * uv3
        random_points_2d.append(pt_uv)

        vertices = mesh.vertices[vmapping[face]]

        pt_3d =  (1 - r1 - r2) * vertices[0] + r1 * vertices[1] + r2 * vertices[2]
        random_points_3d.append(pt_3d)




#                                                       --- verifications---


# --- verification de la correspondance UV -> 3D ---
colors = []

for i in range(3):
    uv = random_points_2d[i]
    pt_3d = random_points_3d[i]
    pixel = back_projeter(pt_3d, image, max_cost=None) #backprojection sur l'image

    if pixel is not None:
        X, Y = pixel
        color = bilinear_interpolate(image, X, Y) #obtention couleur pixel
        colors.append(color)
        
#         print(f"UV: {uv} => 3D: {pt_3d} => Pixel: ({pixel}) => Couleur: {color}")
#         plt.scatter(X, Y, color=color/255.0)  # Normalisation de la couleur pour l'affichage

#     else:
#         print("NOOOOO")


#on regarde les points random créés et le maillage
plt.figure(figsize=(8, 8))
triang = mtri.Triangulation(uvs[:, 0], uvs[:, 1], indices)
plt.triplot(triang, color='lightgray', linewidth=0.5)
face_id = 10 # on choisit une face à colorer pour bien vérifier entre mapping et mesh
uv_face = uvs[indices[face_id]]
plt.fill(uv_face[:, 0], uv_face[:, 1], color='blue', alpha=0.5, label="face choisie")

x_coords = [pt[0] for pt in random_points_2d]
y_coords = [pt[1] for pt in random_points_2d]
plt.scatter(x_coords, y_coords, c='red', s=5)

plt.gca().invert_yaxis()
plt.title("points UV générés")
plt.xlabel("u")
plt.ylabel("v")
plt.show()


# Charger le maillage
mesh = o3d.io.read_triangle_mesh('fichiers_ply/mesh_visible.ply')

mesh.paint_uniform_color([0.5, 0.5, 0.5])

# Nuage de points aléatoires
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(random_points_3d)
mesh.compute_vertex_normals()

# Visualisation
o3d.visualization.draw_geometries([mesh, point_cloud])
