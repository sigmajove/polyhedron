import cairo
import math
import sys

# A number expected to be a floating point rounding error away from zero.
EPSILON = 1e-10


# A half-space is defined by the typle (a, b, c, d)
# (x, y, z) is in the half-space iff a * x + b * y + c * z + d <= 0
# (a, b, c) defines the orientation of the plane that is the border.
# It is a vector perpendicular to the border plane, pointing out of the
# half space. The value d defines the position of the border plane.


# dot product of two vectors
def dot_product(v0, v1):
    length = len(v0)
    assert length == len(v1)
    return sum(v0[i] * v1[i] for i in range(length))


# Determines whether a point is within a half-space
def within(point, half_space):
    return dot_product(half_space[0:3], point) <= -half_space[3]


# We use these objects rather than enum values. They are less error-prone
# than string literals.
class Token:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


KEEP = Token("keep")
DELETE = Token("delete")
NEW_E0 = Token("new_e0")
NEW_E1 = Token("new_e1")

LEFT = Token("left")
RIGHT = Token("right")


ABS_TOL = 1.5e-14
# A vertex is defined by (x, y, z) coordinates in three-space.
class Vertex:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def isclose(self, other):
        return (
            math.isclose(self.x, other.x, abs_tol = ABS_TOL)
            and math.isclose(self.y, other.y, abs_tol = ABS_TOL)
            and math.isclose(self.z, other.z, abs_tol = ABS_TOL)
        )

    def __repr__(self):
        return point_image(self.value())

    # Returns the xyz coordinate as a tuple.
    def value(self):
        return (self.x, self.y, self.z)


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


# Returns the vector from vertices e0 to e1
def delta_vector(e0, e1):
    assert isinstance(e0, Vertex)
    assert isinstance(e1, Vertex)
    return Vector(e1.x - e0.x, e1.y - e0.y, e1.z - e0.z)


# Return the distance between two points.
# May be in 2d or 3d
def distance(p0, p1):
    assert len(p0) == len(p1)
    total = 0.0
    for a, b in zip(p0, p1):
        delta = a - b
        total += delta * delta
    return math.sqrt(total)


def cross_product(v0, v1):
    x0, y0, z0 = v0.value()
    x1, y1, z1 = v1.value()
    dx = y0 * z1 - z0 * y1
    dy = z0 * x1 - x0 * z1
    dz = x0 * y1 - y0 * x1
    return Vector(dx, dy, dz)


def two_planes(plane1, plane2):
    # Find the line that is the intersection of the two planes.
    # The line is represented by point on the line and a vector
    # parallel to the line.

    a1, b1, c1, d1 = plane1
    a2, b2, c2, d2 = plane2

    # The vector [dx, dy, dz] is the cross product of the normal vectors
    # of the two planes.
    dx = b1 * c2 - c1 * b2
    dy = c1 * a2 - a1 * c2
    dz = a1 * b2 - b1 * a2

    # Find a point on the intersection of the two planes.
    # We set one of x, y, or z to zero. Then we have two linear
    # equations in two variables, which we can solve straightforwardly.
    # We pick the variable to set to zero that maximizes the
    # magnitude of the divisor in the computation of the point.
    adx = abs(dx)
    adz = abs(dy)
    avz = abs(dz)
    m = max(adx, adz, avz)
    if m < EPSILON:
        # The two planes are (nearly) parallel; there is no intersection.
        return (None, None)
    if m == adx:
        x = 0.0
        y = (c1 * d2 - c2 * d1) / dx
        z = (b2 * d1 - b1 * d2) / dx
    elif m == adz:
        x = (c2 * d1 - c1 * d2) / dy
        y = 0.0
        z = (a1 * d2 - a2 * d1) / dy
    else:  # m == avz
        x = (b1 * d2 - b2 * d1) / dz
        y = (a2 * d1 - a1 * d2) / dz
        z = 0.0

    # Return (point, vector)
    return ((x, y, z), Vector(dx, dy, dz))


# An Edge of a face of a polyhedron.
# e0 -> e1 are counterclockwise around the face.
# e0 and e1 can be either a Vertex or a Vector
# Both e0 and e1 cannot both be vectors.
class Edge:
    def __init__(self, e0, e1, id):
        assert type(e0) in (Vertex, Vector)
        assert type(e1) in (Vertex, Vector)
        assert type(e0) is not Vector or type(e1) is not Vector
        self.e0 = e0
        self.e1 = e1
        self.id = id

        self.action = KEEP
        # This attribute is either KEEP, DELETE, NEW_E0, or NEW_E1

    def __repr__(self):
        return f"(e0={self.e0} e1={self.e1} action={self.action})"


# Given a plane and a line defined by a point and vector,
# returns the Vertex where the line crosses the plane.
# Returns None if the line is (nearly) parallel to the plane.
def plane_line(plane, point, vector):
    if not isinstance(plane, tuple):
        raise RuntimeError(f"Bad plane {plane}")
    assert len(plane) == 4
    assert isinstance(point, tuple)
    assert len(point) == 3
    assert isinstance(vector, tuple)
    assert len(vector) == 3
    a, b, c, d = plane
    px, py, pz = point
    vx, vy, vz = vector

    denom = dot_product(plane[0:3], vector)
    if abs(denom) < EPSILON:
        # The line is (nearly) parallel to the plane.
        # There is no interesection.
        return None
    t = -(a * px + b * py + c * pz + d) / denom

    x = px + t * vx
    y = py + t * vy
    z = pz + t * vz
    return Vertex(x, y, z)


def trim_vertex_vertex(edge, half):
    assert isinstance(edge.e0, Vertex)
    assert isinstance(edge.e1, Vertex)
    ev0 = edge.e0.value()
    ev1 = edge.e1.value()

    code = int(within(ev0, half)) + 2 * int(within(ev1, half))
    match code:
        case 0:
            # Neither endpoint is in the half-space
            edge.action = DELETE
        case 1 | 2:
            # One endpoint is in the space and the other isn't.
            # We must create a new endpoint.

            # The vector from e0 to e1
            vector = [ev1[i] - ev0[i] for i in range(3)]

            # The new vertex
            new_vertex = plane_line(half, edge.e0.value(), vector)

            # Replace the endpoint not in the half-space.
            if code == 1:
                edge.e1 = new_vertex
                edge.action = NEW_E1
            else:
                edge.e0 = new_vertex
                edge.action = NEW_E0
        case 3:
            # Both endpoints are in the half-space.
            edge.action = KEEP


def trim_vertex_vector(edge, half):
    assert isinstance(edge.e0, Vertex)
    assert isinstance(edge.e1, Vector)
    vertex = edge.e0
    vector = edge.e1

    # Test if the vector points into or out of the half space.
    # negative means into
    # positive maeans out of
    # zero means the vector is parallel to the border.
    vector_dir = dot_product(half[0:3], vector.value())

    # Test if point is within the half-space
    # negative means inside the half-space
    # positive means outside the half-space
    # zero means on the border plane
    point_within = dot_product(half[0:3], vertex.value()) + half[3]

    # We use fuzzy comparisons to avoid making tiny changes
    # due to roundoff errors. If both vector_dir and point_within are very
    # close to zero, it means the vector is parallel to and very near the
    # border plane. The conditions of both of the following if statements
    # will be satsified. We err on the side of keeping the edge.
    if vector_dir <= EPSILON and point_within <= EPSILON:
        # The half-line is entirely inside the half-space
        edge.action = KEEP
        return

    if vector_dir >= -EPSILON and point_within >= -EPSILON:
        # The half-line is entirely out of the half-space
        edge.action = DELETE
        return

    assert abs(point_within) > EPSILON
    assert abs(vector_dir) > EPSILON

    # Compute the point where half space border interesects the line.
    # Since the half-line is partially in the space, the new point
    # must be on the half-line.
    t = -point_within / vector_dir
    new_vertex = Vertex(
        vertex.x + t * vector.dx,
        vertex.y + t * vector.dy,
        vertex.z + t * vector.dz,
    )

    if vector_dir < 0:
        # The vector points into the half-space.
        edge.action = NEW_E0
        edge.e0 = new_vertex
    else:
        # The vector points out the half-space.
        edge.action = NEW_E1
        edge.e1 = new_vertex


def trim_vector_vertex(edge, half):
    assert isinstance(edge.e0, Vector)
    assert isinstance(edge.e1, Vertex)
    vector = edge.e0
    vertex = edge.e1

    # Test if the vector points into or out of the half space.
    # negative means into
    # positive maeans out of
    # zero means the vector is parallel to the border.
    vector_dir = dot_product(half[0:3], vector.value())

    # Test if point is within the half-space
    # negative means inside the half-space
    # positive means outside the half-space
    # zero means on the border plane
    point_within = dot_product(half[0:3], vertex.value()) + half[3]

    # We use fuzzy comparisons to avoid making tiny changes
    # due to roundoff errors. If both vector_dir and point_within are very
    # close to zero, it means the vector is parallel to and very near the
    # border plane. The conditions of both of the following if statements
    # will be satsified. We err on the side of keeping the edge.
    if vector_dir >= -EPSILON and point_within <= EPSILON:
        # The half-line is entirely inside the half-space
        edge.action = KEEP
        return

    if vector_dir <= EPSILON and point_within >= -EPSILON:
        # The half-line is entirely out of the half-space
        edge.action = DELETE
        return

    assert abs(point_within) > EPSILON
    assert abs(vector_dir) > EPSILON

    # Compute the point where half space border interesects the line.
    # Since the half-line is partially in the space, the new point
    # must be on the half-line.
    t = -point_within / vector_dir
    new_vertex = Vertex(
        vertex.x + t * vector.dx,
        vertex.y + t * vector.dy,
        vertex.z + t * vector.dz,
    )

    if vector_dir < 0:
        # The vector points into the half-space.
        edge.action = NEW_E0
        edge.e0 = new_vertex
    else:
        # The vector points out the half-space.
        edge.action = NEW_E1
        edge.e1 = new_vertex


def trim(edge, half):
    (
        trim_vector_vertex
        if isinstance(edge.e0, Vector)
        else (
            trim_vertex_vector
            if isinstance(edge.e1, Vector)
            else trim_vertex_vertex
        )
    )(edge, half)


# Returns whethere two planes are (nearly) parallel.
def is_parallel(h1, h2):
    a1, b1, c1, _ = h1
    a2, b2, c2, _ = h2

    return (
        max(
            abs(a1 * b2 - a2 * b1),
            abs(b1 * c2 - b2 * c1),
            abs(c1 * a2 - c2 * a1),
        )
        < EPSILON
    )


# The point that is the intersection of three planes.
# of which are parallel.
def tripoint(plane0, plane1, plane2):
    p10, vec10 = two_planes(plane0, plane1)
    if p10 is None or vec10 is None:
        return None
    return plane_line(plane2, p10, vec10.value())


# Brute force way of finding a face
def brute_force(in_half, rest):
    result = []
    for i, h in enumerate(rest):
        for j in range(i + 1, len(rest)):
            point = tripoint(in_half, rest[i][1], rest[j][1])
            if point is None:
                continue
            keep = True
#            for k in range(0, len(rest)):
#                if k == i or k == j:
#                    continue
#                if not within(point.value(), rest[k][1]):
#                    keep = False
#                    break
            if keep:
                result.append((point, rest[i][0], rest[j][0]))
    return result


def starter(plane0, idplane1, idplane2):
    id1, plane1 = idplane1
    id2, plane2 = idplane2

    print("Start", plane_image(plane0))
    print("slash1", plane_image(plane1))
    print("slash2", plane_image(plane2))

    p10, vec10 = two_planes(plane0, plane1)
    _, vec20 = two_planes(plane2, plane0)
    apex_vertex = plane_line(plane2, p10, vec10.value())

    if not within(vec10.move(apex_vertex), plane2):
        vec10.reverse()
    if not within(vec20.move(apex_vertex), plane1):
        vec20.reverse()

    if dot_product(cross_product(vec10, vec20).value(), plane0[:3]) > 0:
        vec10.reverse()
        return [Edge(vec10, apex_vertex, id1), Edge(apex_vertex, vec20, id2)]
    else:
        vec20.reverse()
        return [Edge(vec20, apex_vertex, id2), Edge(apex_vertex, vec10, id1)]


def show_edges(title, edges):
    print(f"========== {title}")
    for e in edges:
        print(e)
    print(f"==========")


# Find the face in in_half formed by the interesection with all the
# other half-planes.
def get_face(in_half, rest):
    print("get face")
    assert len(rest) >= 2
    rest = list(filter(lambda r: not is_parallel(in_half, r[1]), rest))

    # Permute rest so that rest[0] and rest[1] are not parallel.
    found = None
    for i in range(1, len(rest)):
        if not is_parallel(rest[0][1], rest[i][1]):
            found = i
            break
    assert found is not None
    rest[1], rest[found] = rest[found], rest[1]

    edges = starter(in_half, rest[0], rest[1])
    show_edges("starter", edges)
    for r in rest[2:]:
        name = plane_image(r[1])
        edges = add_half(edges, r, in_half, name == "f3")
        show_edges(f"Add half {name}", edges)

    return edges


# Slices a face, keeping just the part in the half-plane
def add_half(edges, idhalf, face_plane, trace):
    id, half = idhalf
    # Iterate over all the edges, and decide what to do
    # with each.
    for e in edges:
        trim(e, half)

    if trace:
        show_edges("after trim", edges)

    # Remove all the edges whose action is DELETE
    edges = list(filter(lambda e: e.action is not DELETE, edges))

    # Find all the edges containing newly created vertices.
    pairs = list(filter(lambda e: e.action in (NEW_E0, NEW_E1), edges))
    match len(pairs):
        case 0:
            pass
        case 1:
            edge = pairs[0]
            old_vector = delta_vector(edge.e0, edge.e1)
            _, new_vector = two_planes(face_plane, half)
            if (
                dot_product(
                    face_plane[:3],
                    cross_product(old_vector, new_vector).value(),
                )
                > 0
            ) ^ (edge.action is NEW_E0):
                new_vector.reverse()
            if edge.action is NEW_E0:
                edges.append(Edge(new_vector, edge.e0, id))
            else:
                edges.append(Edge(edge.e1, new_vector, id))

        case 2:
            e0 = None
            e1 = None
            for edge in pairs:
                if edge.action is NEW_E0:
                    e1 = edge.e0
                else:
                    e0 = edge.e1
            assert e0 is not None and e1 is not None
            if not e0.isclose(e1):
                edges.append(Edge(e0, e1, id))

        case _:
            raise RuntimeError(f"{len(pairs)} new vertices")

    result = []
    for edge in edges:
        if (
            isinstance(edge.e0, Vertex)
            and isinstance(edge.e1, Vertex)
            and edge.e0.isclose(edge.e1)
        ):
            continue
        edge.action = KEEP
        result.append(edge)
    return result


# Cope with lack of accuracy in Vertex computations.
class VertexTable:
    def __init__(self):
        self.table = []

    def insert(self, key, value):
        assert isinstance(key, Vertex)
        for k, _ in self.table:
            if key.isclose(k):
                raise RuntimeError("duplicate key")
        self.table.append((key, value))

    def find(self, key):
        assert isinstance(key, Vertex)
        for k, v in self.table:
            if key.isclose(k):
                return v
        raise RuntimeError("key not found")


class PointIntern:
    def __init__(self):
        self.points = []

    def insert(self, vertex, faces):
        for i, point in enumerate(self.points):
            if vertex.isclose(point[0]):
                for f in faces:
                    point[1].add(f)
                return i
        self.points.append((vertex, set(faces)))
        return len(self.points)



def brute_faces(half_spaces):
    points = PointIntern()
    for i in range(0, len(half_spaces)):
        half_i = half_spaces[i]
        for j in range(i+1, len(half_spaces)):
            half_j = half_spaces[j]
            for k in range(j+1, len(half_spaces)):
                half_k = half_spaces[k]
                vertex = tripoint(half_i, half_j, half_k)
                if vertex is not None:
                    points.insert(vertex, (i, j, k))
    
    # Delete all the points that are outside the polyhedron.
    result = []
    for vertex, faces in points.points:
        keep = True
        for i, h in enumerate(half_spaces):
            if i in faces:
                continue

            if dot_product(h[0:3], vertex.value()) + h[3] > EPSILON:
                keep = False
                break
        if keep:
            result.append((vertex, faces))

    return result



def find_faces(half_planes):
    # Assign an id to each half-plane
    half_planes = list(enumerate(half_planes))

    faces = []
    for i in range(len(half_planes)):
        face = get_face(half_planes[i][1], half_planes[:i] + half_planes[i:])
        edict = VertexTable()
        n = len(face)
        for edge in face:
            assert isinstance(edge.e0, Vertex)
            assert isinstance(edge.e1, Vertex)
            edict.insert(edge.e0, edge)
        e = face[0]
        start = e
        sorted = [e]
        while True:
            e = edict.find(e.e1)
            if e is start:
                break
            sorted.append(e)
            assert len(sorted) <= n
        assert len(sorted) == n

        show_edges("Sorted edges", sorted)
        faces.append(sorted)
    return faces


# p0 and p1 are the last 2-d points drawn.
# e0 and e1 are consecutive edges
# determine and return p2, the next 2-d point.
def next_point(p0, p1, edge0, edge1):
    # Get the sin and cos of the angle between the two edges.
    print("next_point")
    print("p0", p0)
    print("p1", p1)
    print("edge0", edge0)
    print("edge1", edge1)

    vec0 = delta_vector(edge0.e1, edge0.e0)
    vec1 = delta_vector(edge1.e0, edge1.e1)
    print("vec0", vec0)
    print("vec1", vec1)
    cos_a = dot_product(vec0.value(), vec1.value()) / (
        vec0.magnitude() * vec1.magnitude()
    )
    print("cos_a", cos_a)
    sin_a = math.sqrt(1.0 - cos_a * cos_a)

    # Get the sin and cos of the angle of the line segment from p0 to p1
    d = distance(p0, p1)
    sin_b = (p1[1] - p0[1]) / d
    cos_b = (p1[0] - p0[0]) / d
    print("sin_b", sin_b)
    print("cos_b", cos_b)

    # Compute the sin and cos of a-b
    sin_diff = sin_a * cos_b - cos_a * sin_b
    cos_diff = cos_a * cos_b + sin_a * sin_b
    print("sin_diff", sin_diff)
    print("cos_diff", cos_diff)
    print("=====================")

    x = p1[0] - d * cos_diff
    y = p1[1] + d * sin_diff

    return (x, y)


def plot(faces, face_id, from_id, ctx):
    print("Plotting face", face_id)
    face = faces[face_id]
    if from_id is None:
        p0 = (0.0, 0.0)
        ctx.move_to(*p0)
        print("Move to", p0)
        p1 = (delta_vector(face[0].e0, face[0].e1).magnitude(), 0.0)
        face[0].plotted = p1
        ctx.line_to(*p1)
        print("line to", p1)
        joined = face
        prev_edge = face[0]
        for edge in face[1:]:
            next_p = next_point(p0, p1, prev_edge, edge)
            prev_edge = edge
            ctx.line_to(*next_p)
            print("line to", next_p)
            edge.plotted = next_p
            p0 = p1
            p1 = next_p
        ctx.stroke()
    else:
        gappy = None
        for i, e in enumerate(face):
            if e.id == from_id:
                gappy = i
                break
        assert gappy is not None
        joined = face[gappy + 1 :] + face[:gappy]

        from_face = faces[from_id]
        found = None
        pred_edge = from_face[-1]
        for edge in from_face:
            if edge.id == face_id:
                found = edge
                break
            pred_edge = edge
        assert found is not None
        p0 = found.plotted
        p1 = pred_edge.plotted
        ctx.move_to(*p1)
        prev_edge = face[gappy]
        prev_edge.plotted = p1
        for edge in joined:
            next_p = next_point(p0, p1, prev_edge, edge)
            prev_edge = edge
            print(next_p)
            ctx.line_to(*next_p)
            edge.plotted = next_p
            p0 = p1
            p1 = next_p
        ctx.stroke()
    for f in face:
        assert hasattr(f, "plotted")


def flatten(faces):
    with cairo.RecordingSurface(cairo.CONTENT_COLOR_ALPHA, None) as rs:
        ctx = cairo.Context(rs)
        ctx.scale(100.0, -100.0)
        ctx.set_line_width(0.05)
        ctx.set_source_rgba(0, 0, 0, 1)

        visited = len(faces) * [False]

        def dfs(from_id, i):
            nonlocal visited

            visited[i] = True
            face = faces[i]
            plot(faces, i, from_id, ctx)
            for edge in face:
                if not visited[edge.id]:
                    dfs(i, edge.id)
                    break

        dfs(None, 0)

        x, y, width, height = rs.ink_extents()
        surface = cairo.SVGSurface("output.svg", width, height)
        ccc = cairo.Context(surface)
        ccc.set_source_surface(rs, -x, -y)
        ccc.paint()
        surface.finish()
        surface.flush()


def main():
    x_hi = [1.0, 0.0, 0.0, -1.0]
    x_lo = [-1.0, 0.0, 0.0, -1.0]
    y_hi = [0.0, 1.0, 0.0, -1.0]
    y_lo = [0.0, -1.0, 0.0, -1.0]
    z_hi = [0, 0.0, 1.0, -1.0]
    z_lo = [0.0, 0.0, -1.0, -1.0]

    faces = find_faces([x_lo, x_hi, y_lo, y_hi, z_lo, z_hi])
    for i, f in enumerate(faces):
        adj = [e.id for e in f]
        print(f"{i}: {adj}")
    flatten(faces)


def make_octa_face(p0, p1, p2):
    c = cross_product(delta_vector(p1, p2), delta_vector(p1, p0)).value()
    result = (*c, -dot_product(p0.value(), c))
    assert within((0, 0, 0), result)
    return result


NW = Vertex(-1, 1, 0)
NE = Vertex(1, 1, 0)
SW = Vertex(-1, -1, 0)
SE = Vertex(1, -1, 0)
TOP = Vertex(0, 0, math.sqrt(2.0))
BOTTOM = Vertex(0, 0, -math.sqrt(2.0))
f0 = make_octa_face(SE, NE, TOP)
f1 = make_octa_face(NE, NW, TOP)
f2 = make_octa_face(NW, SW, TOP)
f3 = make_octa_face(SW, SE, TOP)
f4 = make_octa_face(NE, SE, BOTTOM)
f5 = make_octa_face(NW, NE, BOTTOM)
f6 = make_octa_face(SW, NW, BOTTOM)
f7 = make_octa_face(SE, SW, BOTTOM)


# Get rid of negative zero
def normalize(plane):
    result = tuple(0 if x == 0 else x for x in plane)
    return result


PLANES = {
    normalize(f0): "f0",
    normalize(f1): "f1",
    normalize(f2): "f2",
    normalize(f3): "f3",
    normalize(f4): "f4",
    normalize(f5): "f5",
    normalize(f6): "f6",
    normalize(f7): "f7",
}


def plane_image(f):
    return PLANES[normalize(f)]


def octa():
    halves = [f0, f1, f2, f3, f4, f5, f6, f7]
    faces = brute_faces(halves)
    for f in faces:
        print(f)
    return
    for i, f in enumerate(faces):
        print ("======face", i)
        for g in f:
            print ("vertex", g[0])
            print ("connecting", g[1:])
    return
    for i, f in enumerate(faces):
        adj = [e.id for e in f]
        print(f"{i}: {adj}")
    flatten(faces)


NWU = (-1, +1, +1)
NEU = (+1, +1, +1)
SWU = (-1, -1, +1)
SEU = (+1, -1, +1)
NWD = (-1, +1, -1)
NED = (+1, +1, -1)
SWD = (-1, -1, -1)
SED = (+1, -1, -1)
POINTS = {
    NWU: "NWU",
    NEU: "NEU",
    SWU: "SWU",
    SEU: "SEU",
    NWD: "NWD",
    NED: "NED",
    SWD: "SWD",
    SED: "SED",
}


def point_image(point):
    result = POINTS.get(point, None)
    if result is None:
        result = str(point)
    return result


if __name__ == "__main__":
    octa()
