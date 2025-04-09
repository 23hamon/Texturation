import trimesh
mesh = trimesh.load_mesh("maillage_avec_uv.obj")

print(mesh.faces[0])
print(mesh.vertices[0])