import open3d as o3d
import numpy as np
from utils import get_image_data
from visible_faces_from_view import build_Mpj


def clean_mesh(old_mesh, rot_images, t_images, cos_theta_max=0):
    """
    Supprime les faces du mesh qui ne sont visibles dans aucune vue.
    **Input**:
        - old_mesh (o3d.geometry.TriangleMesh): le mesh d'origine
    **Output** :
        o3d.geometry.TriangleMesh: un nouveau mesh avec seulement les faces visibles
    """
    old_Mpj = build_Mpj(old_mesh, rot_images, t_images, "l", cos_theta_max)

    visible_faces_mask = np.any(old_Mpj, axis=1)  # shape: (nb_faces,)
    triangles = np.asarray(old_mesh.triangles)
    vertices = np.asarray(old_mesh.vertices)

    new_triangles = triangles[visible_faces_mask]
    used_vertices_idx = np.unique(new_triangles)
    old_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices_idx)}
    remapped_triangles = np.array([[old_to_new_idx[idx] for idx in triangle] for triangle in new_triangles])

    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(vertices[used_vertices_idx])
    new_mesh.triangles = o3d.utility.Vector3iVector(remapped_triangles)

    new_mesh.compute_vertex_normals()
    return new_mesh


if __name__ == "__main__" :
    import multiprocessing

    multiprocessing.set_start_method('spawn')
    mesh = o3d.io.read_triangle_mesh("ply/mesh_cailloux_luca_LOW.ply")
    rot_images = []
    t_images = []
    N = 52
    for j in range(N) :
        rot, t = get_image_data(j+1)
        rot_images.append(rot)
        t_images.append(t)
    rot_images = np.array(rot_images)
    t_images = np.array(t_images)
    new_mesh = clean_mesh(mesh, rot_images, t_images)
    o3d.visualization.draw_geometries([new_mesh])
    o3d.io.write_triangle_mesh("ply/mesh_cailloux_luca_CLEAN.ply", new_mesh)
    new_Mpj = build_Mpj(new_mesh, rot_images, t_images, "l")
    np.save("tensors/Mpj.npy", new_Mpj)