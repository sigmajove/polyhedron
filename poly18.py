import math
import numpy

# A number expected to be a floating point rounding error away from zero.
EPSILON = 1e-10


# Maintains small natural id for each Vertex, and the set of faces
# that use each Vertex.
class PointIntern:
    # Initializes the map to empty.
    def __init__(self):
        self.points = []

    # Finds or creates the id associated with a vertex.
    # vertex is a Vertex.
    # faces is an iterable of face ids.
    # The find function is fuzzy, accepting points that are close as equal.
    # The set of faces is the union of every face provided by any
    # call to insert.
    # Returns the id number of the Vertex.
    def insert(self, vertex, faces):
        for i, point in enumerate(self.points):
            if vertex.isclose(point[0]):
                for f in faces:
                    point[1].add(f)
                return i
        self.points.append((vertex, set(faces)))
        return len(self.points)


# dot product of two vectors
def dot_product(v0, v1):
    length = len(v0)
    assert length == len(v1)
    return sum(v0[i] * v1[i] for i in range(length))


ABS_TOL = 1.5e-14
REL_TOL = 1e-8


# A vertex is defined by (x, y, z) coordinates in three-space.
class Vertex:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def isclose(self, other):
        return (
            math.isclose(self.x, other.x, rel_tol=REL_TOL, abs_tol=ABS_TOL)
            and math.isclose(self.y, other.y, rel_tol=REL_TOL, abs_tol=ABS_TOL)
            and math.isclose(self.z, other.z, rel_tol=REL_TOL, abs_tol=ABS_TOL)
        )

    def __repr__(self):
        return str(self.value())

    # Returns a vector whose head is self and tail is other.
    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    # Returns the xyz coordinate as a tuple.
    def value(self):
        return (self.x, self.y, self.z)


def tripoint(plane0, plane1, plane2):
    try:
        x = Vertex(
            *numpy.linalg.solve(
                numpy.array([plane0[0:3], plane1[0:3], plane2[0:3]]),
                numpy.array([-plane0[3], -plane1[3], -plane2[3]]),
            ).tolist(),
        )
    except numpy.linalg.LinAlgError:
        x = None
    return x


# A Vector is defined by (dx, dy, dz) coordinates in three-space.
class Vector:
    def __init__(self, dx, dy, dz):
        self.dx = dx
        self.dy = dy
        self.dz = dz

    def __repr__(self):
        return f"Vector(dx={self.dx} dy={self.dy} dz={self.dz})"

    # Reverse the direction of the vector
    def reverse(self):
        self.dx = -self.dx
        self.dy = -self.dy
        self.dz = -self.dz

    # Return the point obtained by moving the vertex along the vector.
    def move(self, vertex):
        assert isinstance(vertex, Vertex)
        return (vertex.x + self.dx, vertex.y + self.dy, vertex.z + self.dz)

    # Returns the xyz coordinate as a tuple.
    def value(self):
        return (self.dx, self.dy, self.dz)

    def magnitude(self):
        return math.sqrt(
            self.dx * self.dx + self.dy * self.dy + self.dz * self.dz
        )

    # Returns a vector of magnitude 1 in the same direction as self.
    def normalize(self):
        m = self.magnitude()
        return Vector(self.dx / m, self.dy / m, self.dz / m)

    # Return self multiplied by scalar s
    def scale(self, s):
        return Vector(self.dx * s, self.dy * s, self.dz * s)

    # Return the sum self and v
    def __add__(self, v):
        assert isinstance(v, Vector)
        return Vector(self.dx + v.dx, self.dy + v.dy, self.dz + v.dz)


def cross_product(v0, v1):
    x0, y0, z0 = v0.value()
    x1, y1, z1 = v1.value()
    dx = y0 * z1 - z0 * y1
    dy = z0 * x1 - x0 * z1
    dz = x0 * y1 - y0 * x1
    return Vector(dx, dy, dz)


# Returns the vector from vertices e0 to e1
def delta_vector(e0, e1):
    assert isinstance(e0, Vertex)
    assert isinstance(e1, Vertex)
    return Vector(e1.x - e0.x, e1.y - e0.y, e1.z - e0.z)


# Given three consecutive verticies on a face,
# returns whether their order is counterclockwise
def is_counterclockwise(v0, v1, v2, face):
    return (
        dot_product(
            cross_product(delta_vector(v0, v1), delta_vector(v1, v2)).value(),
            face[0:3],
        )
        > 0
    )


# The result is a list of tuples, one for each face, in the same order in
# which the half-spaces were passed in. The index of the half space will
# be the face-id of the corresponding face.
# The ith tuple will be of the form (v, f, n, c), where
#   v is the list of vertices of that face in counterclockwise order
#   f is a list of face-ids that are adjacent to the face (in what order?)
#   n is normal vector perpendicular to the face, pointing outward.
#   c is the 3d coordinates of the point where the face touches the sphere.
def make_polygon(half_spaces):
    # Find all the potential vertices, using the brute force approach
    # of intersecting all possible triples of spaces. Not all of these
    # points will be on the surface of the polyhedron.
    points = PointIntern()
    for i in range(0, len(half_spaces)):
        half_i = half_spaces[i]
        for j in range(i + 1, len(half_spaces)):
            half_j = half_spaces[j]
            for k in range(j + 1, len(half_spaces)):
                half_k = half_spaces[k]
                vertex = tripoint(half_i, half_j, half_k)
                if vertex is not None:
                    points.insert(vertex, (i, j, k))

    # A list of the (vertex, face-set) pairs that are actual vertices of
    # the polyhederon.
    kept_points = []

    # More brute force. Iterate over all the points. For each, iterate
    # over all the half spaces. Only keep the points that are contained
    # in every half-space.
    for vertex, faces in points.points:
        keep = True
        for i, h in enumerate(half_spaces):
            # if i is one of the faces that defines the point, we don't
            # need to perform any computation, which would be inaccurate
            # anyway.
            if i in faces:
                continue

            # Check if the vertex is contained with the half-space h,
            # allowing for a little inaccuracy. If not, then we don't
            # keep the vertex.
            if dot_product(h[0:3], vertex.value()) + h[3] > EPSILON:
                keep = False
                break
        if keep:
            kept_points.append((vertex, faces))

    # We have all the vertices. Now we need to find all the edges.

    # face_edges is an array over all the half-spaces, which correspond
    # to faces of the polyhedron. For each we maintain a list of edges,
    # where each edge is represent by pair of vertices followed by the
    # index of the other face that contains that edge.

    # More brute force. We iterate over all pairs of (vertex, face-set) pairs.
    face_edges = [[] for _ in range(len(half_spaces))]
    for i in range(len(kept_points)):
        v_i, f_i = kept_points[i]
        for j in range(i + 1, len(kept_points)):
            v_j, f_j = kept_points[j]
            faces = f_i.intersection(f_j)
            match len(faces):
                case 2:
                    # Convert the intersection to a list of two face-ids.
                    f = list(faces)

                    # At this point, we just record two verticies.
                    # We don't try to order them. That will come later.
                    face_edges[f[0]].append((v_i, v_j, f[1]))
                    face_edges[f[1]].append((v_i, v_j, f[0]))
                case _:
                    pass

    # Now we have enough information to determine each face.
    result = []
    for z, fe in enumerate(face_edges):
        v0, v1, f = fe[0]
        vertices = None
        for v2, v3, g in fe[1:]:
            if v2 == v1:
                if is_counterclockwise(v0, v1, v3, half_spaces[z]):
                    vertices = [v0, v1, v3]
                    faces = [f, g]
                else:
                    vertices = [v3, v1, v0]
                    faces = [g, f]
                break
            if v3 == v1:
                if is_counterclockwise(v0, v1, v2, half_spaces[z]):
                    vertices = [v0, v1, v2]
                    faces = [f, g]
                else:
                    vertices = [v2, v1, v0]
                    faces = [g, f]
                break
        if not vertices:
            raise RuntimeError("face failure")
        while True:
            next_v = None
            for v0, v1, f in fe:
                if {v0, v1} == {vertices[-1], vertices[-2]}:
                    continue
                if v0 == vertices[-1]:
                    next_v = v1
                    break
                if v1 == vertices[-1]:
                    next_v = v0
                    break
            if next_v is None:
                raise RuntimeError("face failure")
            faces.append(f)
            if next_v == vertices[0]:
                break
            vertices.append(next_v)
            if len(vertices) > len(fe):
                print(f"Error on face {z+1}")
                raise RuntimeError("face failure")

        # Unit vector pointing out of the half space
        normal = Vector(*half_spaces[z][0:3]).normalize()

        result.append((vertices, faces, normal, half_spaces[z][0:3]))
    return result


def poly18():
    # These parameters define an 18-sided polyhedron whose faces have equal
    # area within a tenth of a percent.
    phi = 0.67401686035084829
    weights = [
        0987.57259746581451054226,
        1035.38120169441140205890,
        0954.69453935047556569771,
        1036.79718497324597592524,
        0985.34222651605227838445,
        1019.27739583333345763094,
        0981.56770833333337122895,
        0988.71962500000006457412,
        1013.32433333333335667703,
    ]

    # Here is where we define the somewhat nonintuitve numbering system
    # for the 18 faces. The faces are defined in pairs with each
    # member of the pair antipodal.
    # The first 10 faces wrap around the North and South pole, spaced
    # not quite equally-- that is what the weights are for. Phi is the
    # angle of these faces with respect to their pole.
    # The second 8 faces wrap around the equator, not quite evenly spaced,
    # as defined by the weights.
    coords = []
    total = float(sum(weights[:5]))
    for i in range(5):
        angle = 2.0 * math.pi * sum(weights[: i + 1]) / total
        coords.append((angle, phi))

    total = float(sum(weights[5:]))
    equator = math.pi / 2
    for i in range(5, len(weights)):
        angle = math.pi * sum(weights[5 : i + 1]) / total
        coords.append((angle, equator))

    # Convert from Spherical coordinates to Cartesian coordinates.
    halves = []
    for theta, phi in coords:
        sin_phi = math.sin(phi)
        x = math.cos(theta) * sin_phi
        y = math.sin(theta) * sin_phi
        z = math.cos(phi)

        # The pairs of faces are antipodal.
        # The -1 appears because we use the unit sphere.
        halves.append((x, y, z, -1.0))
        halves.append((-x, -y, -z, -1.0))

    return make_polygon(halves)


def display():
    for i, l in enumerate (poly18()):
        print (f"id = {i}")
        print (f"verticies =")
        for v in l[0]:
            print (f"    {v}")
        print (f"adjacent faces = {l[1]}")
        print (f"normal_vector = {l[2]}")
        print (f"center point = {l[3]}")
        print ("===========")
