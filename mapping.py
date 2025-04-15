import trimesh
import xatlas
import open3d as o3d


mesh = trimesh.load_mesh('fichiers_ply/mesh_cailloux_low.ply')
vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
xatlas.export("maillage_avec_uv.obj", mesh.vertices[vmapping], indices, uvs)


