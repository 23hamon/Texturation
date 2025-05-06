import trimesh
import numpy as np
import xatlas
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.optimize import least_squares
from skimage.draw import polygon
from backprojection import back_projeter
from utils import get_image_data
from tqdm import tqdm
from PIL import Image
from scipy.sparse import lil_matrix
np.set_printoptions(threshold=np.inf)
import cv2

def barycentric_coordinates(p, A, B, C):
    """
    Calcule les coordonnées barycentriques d'un point p par rapport à un triangle défini par A, B, et C.
    
    Arguments :
    - p : array, coordonnées du point à analyser.
    - A, B, C : arrays, coordonnées des trois sommets du triangle.
    
    Retourne :
    - v, w, u : float, coordonnées barycentriques du point p dans le triangle ABC.
    """
    v0, v1, v2 = B - A, C - A, p - A
    d00, d01, d11 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v1, v1)
    d20, d21 = np.dot(v2, v0), np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return v, w, u


def bilinear_interpolate(image, x, y):
    """
    Effectue une interpolation bilinéaire sur une image pour obtenir l'intensité d'un pixel à des coordonnées flottantes (x, y).
    
    Arguments :
    - image : array, l'image sur laquelle l'interpolation est effectuée.
    - x, y : float, coordonnées du point à interpoler dans l'image.
    
    Retourne :
    - array, intensité interpolée au pixel (x, y).
    """
    x0 = int(np.floor(x))
    x1 = min(x0 + 1, image.shape[1] - 1)
    y0 = int(np.floor(y))
    y1 = min(y0 + 1, image.shape[0] - 1)
    wx = x - x0
    wy = y - y0
    top = (1 - wx) * image[y0, x0] + wx * image[y0, x1]
    bottom = (1 - wx) * image[y1, x0] + wx * image[y1, x1]
    return ((1 - wy) * top + wy * bottom).astype(np.uint)


def find_new_face_id(faces, vmapping, indices):
    """
    Trouve les nouvelles faces en fonction des indices UV et des sommets du maillage.
    
    Arguments :
    - faces : list, les faces originales du maillage.
    - vmapping : array, le mappage des sommets du maillage.
    - indices : array, indices des UV des faces.
    
    Retourne :
    - ancienne_vers_nouvelle : dict, un dictionnaire mappant les anciennes faces aux nouvelles.
    """
    ancienne_vers_nouvelle = defaultdict(list)
    original_face_sets = {i: set(face) for i, face in enumerate(faces)}
    for new_face_id, face_uv in enumerate(indices):
        original_vertices = {vmapping[i] for i in face_uv}
        for old_face_id, vertex_set in original_face_sets.items():
            if original_vertices == vertex_set:
                ancienne_vers_nouvelle[old_face_id].append(new_face_id)
                break
    return ancienne_vers_nouvelle


def adjacent_vertices(mesh):
    """
    Crée un dictionnaire des sommets adjacents pour chaque sommet du maillage.
    
    Arguments :
    - mesh : objet, maillage avec des faces.
    
    Retourne :
    - L : dict, dictionnaire des sommets adjacents.
    """
    L = defaultdict(list)
    for face in mesh.faces:
        for i in range(3):
            for j in range(i + 1, 3):
                if face[j] not in L[face[i]]:
                    L[face[i]].append(face[j])
                if face[i] not in L[face[j]]:
                    L[face[j]].append(face[i])
    return L


def couple_vertex_view(M_final, indices):
    """
    Crée des couples de sommets et de vues à partir des faces et des vues finales.
    
    Arguments :
    - M_final : array, identifiant de la vue pour chaque face.
    - indices : array, indices des sommets des faces.
    
    Retourne :
    - M : list, couples (sommet, vue) à partir des faces et des vues.
    """
    M = set()
    for view in range(M_final.shape[0]):
        for face in range(len(indices)):
            if M_final[face] == view:
                for vertex in indices[face]:
                    M.add((vertex, view))
    return list(M)


def from_vertice_uv_to_pixel(vertex_uv_id, view_id, vmapping, mesh):
    """
    Convertit un identifiant de sommet UV en coordonnées de pixel dans une vue.
    
    Arguments :
    - vertex_uv_id : int, identifiant du sommet UV.
    - view_id : int, identifiant de la vue.
    - vmapping : array, mappage des sommets du maillage.
    - mesh : objet, maillage 3D.
    
    Retourne :
    - x, y : int, coordonnées du pixel correspondant dans la vue.
    """
    rot, t = get_image_data(view_id )
    vertex_mesh = mesh.vertices[vmapping[vertex_uv_id]]
    pixel = back_projeter(vertex_mesh, rot, t, "l")[0]
    return pixel[0], pixel[1]


def from_face_to_all_pixel(face_id, view_id, image_shape, indices, uvs, vmapping, mesh):
    """
    Convertit les coordonnées UV d'une face en un ensemble de pixels projetés dans une vue.
    
    Arguments :
    - face_id : int, identifiant de la face.
    - view_id : int, identifiant de la vue.
    - image_shape : tuple, dimensions de l'image.
    - indices : array, indices des sommets des faces.
    - uvs : array, coordonnées UV des faces.
    - vmapping : array, mappage des sommets du maillage.
    - mesh : objet, maillage 3D.
    
    Retourne :
    - pixels : list, liste des coordonnées de pixels dans la vue.
    """
    rot, t = get_image_data(view_id )
    face = indices[face_id]
    uv = uvs[face]
    r, c = uv[:, 1] * image_shape[0], uv[:, 0] * image_shape[1]
    rr, cc = polygon(r, c, image_shape[:2])
    pixels = []
    vertices = [mesh.vertices[vmapping[face[i]]] for i in range(3)]
    A, B, C = np.array([c[0], r[0]]), np.array([c[1], r[1]]), np.array([c[2], r[2]])
    for i in range(len(rr)):
        p = np.array([cc[i], rr[i]])
        u, v, w = barycentric_coordinates(p, A, B, C)
        point_3d = (1 - u - v) * vertices[0] + u * vertices[1] + v * vertices[2]
        pixel = back_projeter(point_3d, rot, t, "l")[0]
        if pixel is not None:
            X, Y = int(pixel[0]), int(pixel[1])
            pixels.append((X, Y))
    return pixels


def f_all_summit(view_id, Vjyxc, vmapping, mesh, channel_id=0):
    """
    Récupère l'intensité des pixels correspondant aux sommets du maillage pour une vue donnée.
    
    Arguments :
    - view_id : int, identifiant de la vue.
    - Vjyxc : array, images des vues.
    - vmapping : array, mappage des sommets du maillage.
    - mesh : objet, maillage 3D.
    - channel_id : int, canal de couleur (par défaut 0 pour rouge).
    
    Retourne :
    - intensities : array, intensités des pixels correspondants aux sommets du maillage dans la vue.
    """
    image = Vjyxc[view_id]
    h, w = image.shape[:2]
    intensities = np.zeros((h, w))
    for vertex_id in range(len(vmapping)):
        x, y = from_vertice_uv_to_pixel(vertex_id, view_id, vmapping, mesh)
        x, y = int(x), int(y)
        if 0 <= y < h and 0 <= x < w:
            intensities[y, x] = image[y, x, channel_id]
    return intensities


def intensity_all_views(views, channel_id, vertex_in_mesh, Vjyxc):
    """
    Calcule l'intensité pour tous les sommets et toutes les vues.
    
    Arguments :
    - views : dict, dictionnaire des vues.
    - channel_id : int, canal de couleur.
    - vertex_in_mesh : dict, mappage des sommets aux vues.
    - Vjyxc : array, images des vues.
    
    Retourne :
    - all_views_intensities : dict, intensité pour chaque sommet et chaque vue.
    """
    all_views_intensities = {}
    for view_id in tqdm(range(len(views)), desc="Calcul de intensity_all_views"):
        rot, t = get_image_data(view_id)
        for (vertex_id, vertex_mesh) in vertex_in_mesh[view_id]:
            pixel = back_projeter(vertex_mesh, rot, t, "l")[0]
            x, y = int(pixel[0]), int(pixel[1])
            all_views_intensities[(vertex_id, view_id)] = Vjyxc[view_id, y, x, channel_id]
    return all_views_intensities


def build_jacobian_sparsity(n_vars, smooth_idx_1, smooth_idx_2, same_idx_1, same_idx_2):
    """
    Crée une matrice creuse de Jacobien pour les contraintes de l'optimisation.
    
    Arguments :
    - n_vars : int, nombre de variables.
    - smooth_idx_1, smooth_idx_2 : listes d'indices des sommets pour les contraintes de lissage.
    - same_idx_1, same_idx_2 : listes d'indices des sommets pour les contraintes de même intensité.
    
    Retourne :
    - J : matrice creuse, structure de la matrice de Jacobien.
    """
    n_smooth = len(smooth_idx_1)
    n_same = len(same_idx_1)
    J = lil_matrix((n_smooth + n_same, n_vars))
    for i in range(n_smooth):
        idx_1, idx_2 = smooth_idx_1[i], smooth_idx_2[i]
        J[i, idx_1] = -1
        J[i, idx_2] = 1
    for i in range(n_same):
        idx_1, idx_2 = same_idx_1[i], same_idx_2[i]
        J[n_smooth + i, idx_1] = 1
        J[n_smooth + i, idx_2] = -1
    return J



def build_g_function(intensity_all, L, M, index_map, lambda_seam=1000):
    print("Préparation des résidus")
    smooth_idx_1, smooth_idx_2 = [], []
    same_idx_1, same_idx_2, delta_I = [], [], []
    for (i1, j) in M:
        for i2 in L.get(i1, []):
            if (i2, j) in index_map:
                smooth_idx_1.append(index_map[(i1, j)])
                smooth_idx_2.append(index_map[(i2, j)])

    for (i, j1) in M:
        for j2 in range(len(views)):
            if j1 != j2 and (i, j2) in index_map:
                idx1 = index_map[(i, j1)]
                idx2 = index_map[(i, j2)]
                I1 = int(intensity_all[(i, j1)])
                I2 = int(intensity_all[(i, j2)])
                same_idx_1.append(idx1)
                same_idx_2.append(idx2)
                delta_I.append((I2 - I1))
    J = build_jacobian_sparsity(len(M), smooth_idx_1, smooth_idx_2, same_idx_1, same_idx_2)
    def g(x):
        res_smooth = x[smooth_idx_1] - x[smooth_idx_2]
        res_seam = (x[same_idx_1] - x[same_idx_2]) - np.array(delta_I)
        return np.concatenate([res_smooth, (lambda_seam ** 0.5) * res_seam])
    print("Lancement de l'optimisation")
    x0 = np.zeros(len(M))
    res = least_squares(g, x0, method='trf', jac_sparsity=J)
    print("Optimisation terminée.")
    assert len(res.x) == len(M)
    return res.x


if __name__ == "__main__":
    # --- Initialisation des paramètres de base ---
    cam = "l"  # caméra à utiliser
    N = 52  # nombre d'images à traiter
    image_path = f"downsampled/scene_{cam}_"  # chemin des images

    # --- Chargement des images ---
    Vjyxc = [
        cv2.cvtColor(cv2.imread(image_path + f"{j:04d}.jpeg"), cv2.COLOR_BGR2RGB)
        for j in range(1, N + 1)
    ]
    Vjyxc = np.stack(Vjyxc, axis=0)  # shape (N, h, w, 3)
    h, w = Vjyxc[0].shape[:2]

    # Création du dictionnaire de vues
    views = {i: Vjyxc[i] for i in range(Vjyxc.shape[0])}

    # Chargement du tensor M_final
    M_final = np.load("tensors/M_final_l.npy")

    # --- Chargement et préparation du mesh ---
    mesh = trimesh.load("ply/mesh_cailloux_luca_CLEAN.ply")
    vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
    faces = mesh.faces
    vertices = mesh.vertices

    # Exportation du maillage paramétré
    xatlas.export("maillage_texture_entiere.obj", mesh.vertices[vmapping], indices, uvs)

    # --- Calcul des relations entre les vertices du mesh et les vues ---
    L = adjacent_vertices(mesh)
    M = couple_vertex_view(M_final, indices)
    index_map = { (i, j): idx for idx, (i, j) in enumerate(M)}
    M_set = set(M)  # accélère la recherche

    # Initialisation des dictionnaires pour les vertices à analyser
    vertex_to_analyse = {}
    vertex_in_mesh = {}

    # Mapping des vertices du mesh
    mapping_mesh = mesh.vertices[vmapping]

    # --- Traitement des vues et des intensités ---
    for view_id in tqdm(range(len(views))):
        vertex_to_analyse[view_id] = [vertex_id for vertex_id in range(len(vmapping)) if (vertex_id, view_id) in M_set]
        vertex_in_mesh[view_id] = []
        for vertex_id in vertex_to_analyse[view_id]:
            if ((vertex_id, mapping_mesh[vertex_id])) not in vertex_in_mesh[view_id]:
                vertex_in_mesh[view_id].append((vertex_id, mapping_mesh[vertex_id]))

    # Calcul des intensités pour chaque couleur (rouge, vert, bleu)
    red_intensity_all_views = intensity_all_views(views, 0, vertex_in_mesh, Vjyxc)
    blue_intensity_all_views = intensity_all_views(views, 2, vertex_in_mesh, Vjyxc)
    green_intensity_all_views = intensity_all_views(views, 1, vertex_in_mesh, Vjyxc)

    # --- Construction des fonctions g pour chaque couleur ---
    g_red = build_g_function(red_intensity_all_views, L, M, index_map)
    g_blue = build_g_function(blue_intensity_all_views, L, M, index_map)
    g_green = build_g_function(green_intensity_all_views, L, M, index_map)

    # Assemblage des valeurs g dans une matrice
    g = np.zeros((len(M), 3), dtype=np.float32)
    for i in range(len(M)):
        g[i, 0] = int(g_red[i])
        g[i, 1] = int(g_green[i])
        g[i, 2] = int(g_blue[i])

    # --- Initialisation des textures et masque UV ---
    image_width = 256
    image_height = 256
    map_texture = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255
    final_texture = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255
    uv_mask = np.zeros((image_height, image_width), dtype=np.uint8)  # masque pour savoir où la texture est appliquée

    # Dictionnaires pour stocker les couleurs
    colors_seamless = {}
    colors_texture = {}
    final_colors = {}

    # --- Traitement des faces du mesh ---
    for face_id in tqdm(range(len(indices)), desc="Traitement des faces"):
        sommets_uv = indices[face_id]
        view_id = M_final[face_id]
        image = Vjyxc[view_id]
        rot, t = get_image_data(view_id)
        face = indices[face_id]
        uv = uvs[face]

        # Collecte des couleurs pour chaque sommet de la face
        colors = []
        vertices = [
            mesh.vertices[vmapping[face[0]]],
            mesh.vertices[vmapping[face[1]]],
            mesh.vertices[vmapping[face[2]]]
        ]
        for vertex_id in sommets_uv:
            vertex_intensity = g[index_map[(vertex_id, view_id)]]
            colors.append(vertex_intensity)
        colors = np.array(colors)

        # Calcul des coordonnées UV pour la face
        uv = uvs[sommets_uv]
        r = uv[:, 1] * map_texture.shape[0]  # Lignes
        c = uv[:, 0] * map_texture.shape[1]  # Colonnes

        rr, cc = polygon(r, c, map_texture.shape[:2])

        # Calcul des coordonnées barycentriques pour chaque pixel
        A = np.array([c[0], r[0]])
        B = np.array([c[1], r[1]])
        C = np.array([c[2], r[2]])

        for x, y in zip(cc, rr):
            p = np.array([x, y])
            u, v, w = barycentric_coordinates(p, A, B, C)
            color = (1 - u - v) * colors[0] + u * colors[1] + v * colors[2]
        
            # Ajout de la couleur calculée dans la texture sans couture
            colors_seamless[(y, x)] = tuple(np.round(color).astype(int))

            pt_3d = (1 - u - v) * vertices[0] + u * vertices[1] + v * vertices[2]
        
            # Back-projection pour obtenir la couleur du pixel
            if view_id < 52:  
                pixel = back_projeter(pt_3d, rot, t, "l")[0]
            else:
                pixel = back_projeter(pt_3d, rot, t, "r")[0]
        
            if pixel is not None:
                px, py = int(pixel[0]), int(pixel[1])
            else:
                continue  # Si la back-projection échoue → on saute
        
            if 0 <= px < image.shape[1] and 0 <= py < image.shape[0]:
                intensity = image[py, px]
            else:
                continue

            # Mise à jour de la texture et des couleurs finales
            colors_texture[(y, x)] = np.clip(intensity, 0, 255)
            map_texture[y, x] = colors_texture[(y, x)].astype(np.uint8)

            final_colors[(y, x)] = intensity + color
            final_colors[(y, x)] = np.clip(intensity + color, 0, 255)
            final_texture[y, x] = final_colors[(y, x)].astype(np.uint8)

            uv_mask[y, x] = 1

    # --- Affichage des résultats ---
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(map_texture)
    axes[0].set_title("Texture originale")
    axes[0].axis('off')

    axes[1].imshow(final_texture)
    axes[1].set_title("Texture seamless")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

    # --- Enregistrement des textures ---
    texture_seamless = Image.fromarray(final_texture)
    texture_seamless.save("map_seamless.png")

    texture_map = Image.fromarray(map_texture)
    texture_map.save("map_original.png")

    # Sauvegarde des textures et du masque UV
    np.save("texture_seamless.npy", final_texture)
    np.save("texture_map.npy", map_texture)
    np.save("uv_mask.npy", uv_mask)