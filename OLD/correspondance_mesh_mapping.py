


import trimesh
import xatlas
import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from backprojection import back_projeter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pyglet
from tqdm import tqdm
from skimage.draw import polygon
from PIL import Image


mesh = trimesh.load_mesh('fichiers_intermediaires/mesh_visible_low.ply')
vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
xatlas.export("fichiers_ply/maillage_avec_uv.obj", mesh.vertices[vmapping], indices, uvs)
image = cv2.imread("downsampled/scene_l_0026.jpeg")[..., ::-1]


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


def barycentric_coordinates(p, A, B, C):
    v0, v1, v2 = B - A, C - A, p - A #chaque cote du triange
    d00, d01, d11 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v1, v1)
    d20, d21 = np.dot(v2, v0), np.dot(v2, v1)

    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom 
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w 
    
    return v, w, u


#                                                        --- main ---

white_image = np.ones((512, 512, 3), dtype=np.uint8) * 255 
white_image_height, white_image_width = white_image.shape[:2]
for face in tqdm(indices):
    uv = uvs[face]
    r = uv[:, 1] * white_image.shape[0] # lignes parce que polygon veut  (y, x) et non pas (x, y) --> r stands for row
    c = uv[:, 0] * white_image.shape[1]  # colonnes
    rr, cc = polygon(r, c,white_image.shape[:2])  

    vertices = mesh.vertices[vmapping[face]]

    #pour chaque pixel, on calcule ses coordonnées barycentriques
    A = np.array([c[0], r[0]])
    B = np.array([c[1], r[1]])
    C = np.array([c[2], r[2]])

    for i in tqdm(range(len(rr))):
        p = np.array([cc[i], rr[i]])
        u, v, w = barycentric_coordinates(p, A, B, C)
        #on calcule le point 3D correspondant
        pt_3d = (1 - u - v) * vertices[0] + u * vertices[1] + v * vertices[2]
        pixel,_,_ = back_projeter(pt_3d, white_image_height, white_image_width, max_cost=None) 
        
        if pixel is not None:
            X, Y = pixel[0], pixel[1]
            color = bilinear_interpolate(image, X, Y)
            white_image[rr[i], cc[i]] = color 

        else:
            print("NOOOOO")


# Affichage
plt.imshow(white_image)
plt.axis('off')
plt.title("Faces remplies à partir du mapping 2D")
plt.show()

texture_map = np.save("fichiers_intermediaires/texture_map.npy", white_image)


texture_image = Image.fromarray(white_image)  
texture_image.save("texture_image2.png")










# # generation des points randoms 3d depuis les coordonnées (U,V) de points random du mapping
# random_points_2d = []
# random_points_3d = []
# print(len(indices))

# for face in tqdm(indices):
#     uv1, uv2, uv3 = uvs[face]
#     for _ in range(20):
#         r1, r2 = generate_random_point(uv1, uv2, uv3)
#         pt_uv = (1 - r1 - r2) * uv1 + r1 * uv2 + r2 * uv3
#         random_points_2d.append(pt_uv)

#         vertices = mesh.vertices[vmapping[face]]
#         pt_3d =  (1 - r1 - r2) * vertices[0] + r1 * vertices[1] + r2 * vertices[2]
#         random_points_3d.append(pt_3d)
# colors = []

# for i in tqdm(range(len(random_points_2d))):
#     uv = random_points_2d[i]
#     pt_3d = random_points_3d[i]
#     pixel = back_projeter(pt_3d, image_height, image_width, max_cost=None) #backprojection sur l'image

#     if pixel is not None:
#         point, _, _ = pixel
#         X, Y = point[0], point[1]
#         color = bilinear_interpolate(image, X, Y) #obtention couleur pixel
#         colors.append(color)

# print(colors)




# # colorisation ptn 3d

# print(random_points_3d)
# print(colors)

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(np.array(random_points_3d))
# pcd.colors = o3d.utility.Vector3dVector(np.array(colors) / 255)
# o3d.visualization.draw_geometries([pcd])
# o3d.io.write_point_cloud("fichiers_ply/output_mesh_visible.ply", pcd)

# #                                                       --- verifications---


        
# #         print(f"UV: {uv} => 3D: {pt_3d} => Pixel: ({pixel}) => Couleur: {color}")
# #         plt.scatter(X, Y, color=color/255.0)  # Normalisation de la couleur pour l'affichage

# #     else:
# #         print("NOOOOO")


# #on regarde les points random créés et le maillage

# plt.figure(figsize=(8, 8))
# triang = mtri.Triangulation(uvs[:, 0], uvs[:, 1], indices)
# plt.triplot(triang, color='lightgray', linewidth=0.5)
# face_id = 10 # on choisit une face à colorer pour bien vérifier entre mapping et mesh
# uv_face = uvs[indices[face_id]]
# plt.fill(uv_face[:, 0], uv_face[:, 1], color='blue', alpha=0.5, label="face choisie")

# x_coords = [pt[0] for pt in random_points_2d]
# y_coords = [pt[1] for pt in random_points_2d]
# plt.scatter(x_coords, y_coords, c='red', s=5)

# plt.gca().invert_yaxis()
# plt.title("points UV générés")
# plt.xlabel("u")
# plt.ylabel("v")
# plt.legend()
# plt.show()



# # # toutes les faces en blanc, sauf la 65 en rouge
# face_colors = np.ones((len(mesh.faces), 4)) * 255
# mesh.visual.face_colors = face_colors

#face à part pour comparaison
# vertices_id = mesh.vertices[vmapping[indices[face_id]]]
# faces_id = mesh.faces[indices[face_id]]
# mesh_id = trimesh.Trimesh(vertices=vertices_id, faces=[[0, 1, 2]],
#                           face_colors=[[255, 0, 0, 255]])

# edges = mesh.edges_unique
# lines = trimesh.load_path(mesh.vertices[edges])
# lines.colors = [[0, 0, 0, 255]] * len(lines.entities) 

# point_cloud = trimesh.points.PointCloud(random_points_3d, colors=[0, 0, 255, 255])

# scene = trimesh.Scene()
# scene.add_geometry(mesh)
# scene.add_geometry(lines)
# scene.add_geometry(point_cloud)
# scene.add_geometry(mesh_id)  # par-dessus

# scene.show()
