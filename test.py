import trimesh
import xatlas
import open3d as o3d
import numpy as np
from backprojection import back_projeter
import cv2
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


#on charge le maillage et le mapping
mesh = trimesh.load_mesh('fichiers_ply/mesh_visible.ply')
vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
xatlas.export("maillage_avec_uv.obj", mesh.vertices[vmapping], indices, uvs)
mapping = trimesh.load_mesh("maillage_avec_uv.obj")
image_path = "downsampled/scene_l_0026.jpeg"
image = cv2.imread(image_path)


#prendre points random (coord barycentriques)
def generate_random_point(uv1, uv2, uv3): #les uvs du maillage appartiennent à [0,1]
    r1 = np.random.random()
    r2 = np.random.random()
    if r1 + r2 > 1:
        r1 = 1 - r1
        r2 = 1 - r2
    point = (1 - r1 - r2) * uv1 + r1 * uv2 + r2 * uv3
    return point

def bilinear_interpolate(image, x, y):
    """
    Interpolation bilinéaire d'une image en un point (x, y) à coordonnées flottantes.
    Retourne un vecteur [B, G, R]
    """
    x0 = int(np.floor(x))
    x1 = min(x0 + 1, image.shape[1] - 1)
    y0 = int(np.floor(y))
    y1 = min(y0 + 1, image.shape[0] - 1)

    # poids
    wx = x - x0
    wy = y - y0

    top = (1 - wx) * image[y0, x0] + wx * image[y0, x1]
    bottom = (1 - wx) * image[y1, x0] + wx * image[y1, x1]
    value = (1 - wy) * top + wy * bottom

    return value.astype(np.uint8)


random_points_3d = [] #points 3d correspondants

#generation points aléatoires et correspondance en 3d
for face in indices:
    uv1 = uvs[face[0]]
    uv2 = uvs[face[1]]
    uv3 = uvs[face[2]]
    for _ in range(20):
        random_point_uv = generate_random_point(uv1, uv2, uv3)
        verts_id = vmapping[face] #on trouve le numéro des sommets de face dans le 3d
        vertices_3d = mesh.vertices[verts_id] #puis on trouve leurs coordonnées dans le mesh 3d
        lambda1, lambda2, lambda3 = 1 - random_point_uv[0] - random_point_uv[1], random_point_uv[0], random_point_uv[1] #on crée les poids des coord barycentrqiues
        if lambda1 + lambda2 + lambda3> 1:
            lambda1 = 1 - lambda1
            lambda2 = 1 - lambda2
            lambda3 = 1 - lambda3
    
        #on trouve les coord barycentriques du pt dans le mesh 3d
        point_3d = (lambda1 * vertices_3d[0] +
                    lambda2 * vertices_3d[1] +
                    lambda3 * vertices_3d[2])
        random_points_3d.append(point_3d)


point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(random_points_3d)
o3d.visualization.draw_geometries([point_cloud], window_name="Points générés en 3D")

#on récupère les picelx correspondant à chaque point random trouvé dans le maillage 3d

#ici, on a donc généré plein de points sur le maillage. Maintenant nous voulons récupérer les coordonnées de ce point dans le mesh 3D. 
pixels = []

for X in random_points_3d[:20]: 
    Y = back_projeter(X, image, max_cost=False)
    pixels.append(Y)
print(len(pixels))

#les pixels ont des coordonnées non entières : il faut donc interpoler pour connaitre leur couleur : 
#récupération des couleurs des pixels de l'image : 
colors = []

for i in range (len(pixels)) :
    X , Y = pixels[i]
    color = bilinear_interpolate(image, X, Y)
    colors.append(color)

print(colors)


# Afficher le mapping UV et les points tirés
plt.figure(figsize=(8, 8))
triangulation = mtri.Triangulation(uvs[:, 0], uvs[:, 1], indices)
plt.triplot(triangulation, color='lightgray', linewidth=0.5)

random_points_uv = np.array([generate_random_point(uvs[face[0]], uvs[face[1]], uvs[face[2]])
                             for face in indices for _ in range(2)])

plt.scatter(random_points_uv[:, 0], random_points_uv[:, 1], c='red', s=5)
plt.gca().invert_yaxis()
plt.title("Points UV générés")
plt.xlabel("u")
plt.ylabel("v")
plt.show()

mesh_o3d = o3d.geometry.TriangleMesh()
mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh.vertices)
mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces)
mesh_o3d.paint_uniform_color([0.7, 0.7, 0.7])
mesh_o3d.compute_vertex_normals()

o3d.visualization.draw_geometries([mesh_o3d, point_cloud])

image_copy = image.copy()
for x, y in pixels[:500]:  # max 500 points pour ne pas saturer
    x_int, y_int = int(round(x)), int(round(y))
    if 0 <= x_int < image.shape[1] and 0 <= y_int < image.shape[0]:
        cv2.circle(image_copy, (x_int, y_int), radius=2, color=(0, 0, 255), thickness=-1)

cv2.imshow("Pixels projetés sur image", image_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()

colors_np = np.array(colors) / 255.0  # Open3D attend des couleurs entre 0 et 1
point_cloud.colors = o3d.utility.Vector3dVector(colors_np)
o3d.visualization.draw_geometries([point_cloud], window_name="Point cloud coloré")

heatmap = np.zeros(image.shape[:2], dtype=np.uint8)
for x, y in pixels:
    x_int, y_int = int(round(x)), int(round(y))
    if 0 <= x_int < heatmap.shape[1] and 0 <= y_int < heatmap.shape[0]:
        heatmap[y_int, x_int] += 1

heatmap_color = cv2.applyColorMap(cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET)
cv2.imshow("Densité des projections", heatmap_color)
cv2.waitKey(0)
cv2.destroyAllWindows()

