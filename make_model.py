import numpy as np
from stl import mesh
import polyhedron

poly = polyhedron.custom()

# Find all the vertices
vert_map = {}
vert_no = 0
for f in poly:
    for v in f[0]:
        if vert_map.setdefault(v, vert_no) == vert_no:
            vert_no += 1

vert = [None] * vert_no
for v, id in vert_map.items():
    # Scale up the coordinates. I don't know whether that is important.
    vert[id] = [3 * c  for c in v.value()]


vertices = np.array(vert)

# Now iterate over the faces, and triangulate each.
triangles = []
for vvv, _ in poly:
    for i in range(2, len(vvv)):
        triangles.append(
            [vert_map[vvv[0]], vert_map[vvv[i - 1]], vert_map[vvv[i]]]
        )


# Define the 12 triangles composing the cube
faces = np.array(triangles)

# Create the mesh
cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    for j in range(3):
        cube.vectors[i][j] = vertices[f[j], :]

# Save the mesh to file
cube.save("c:/users/sigma/documents/poly.stl")
