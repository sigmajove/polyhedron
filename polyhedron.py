import cairo
import collections
import json
import math
import numpy
import struct
from collections.abc import Iterable

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

    # Returns a vector of magnitude 1 in the same direction as self.
    def normalize(self):
        m = self.magnitude()
        return Vector(self.dx / m, self.dy / m, self.dz / m)

    # Return self multiplied by scalar s
    def scale(self, s):
        return Vector(self.dx * s, self.dy * s, self.dz * s)

    # Return the sum self and v
    def add(self, v):
        assert isinstance(v, Vector)
        return Vector(self.dx + v.dx, self.dy + v.dy, self.dz + v.dz)


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
    if not isinstance(plane, Iterable):
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


def test_result(x, plane0, plane1, plane2):
    if x is None:
        return
    print(
        *(
            dot_product(x.value(), p[0:3]) + p[3]
            for p in (plane0, plane1, plane2)
        )
    )


# The point that is the intersection of three planes.
# of which are parallel.
def tripoint_old(plane0, plane1, plane2):
    p10, vec10 = two_planes(plane0, plane1)
    if p10 is None or vec10 is None:
        result = None
    else:
        result = plane_line(plane2, p10, vec10.value())

    print ("me", result)
    test_result(result, plane0, plane1, plane2)
    tripoint2(plane0, plane1, plane2)
    print("=====================")
    return result


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
    # print ("np", x)
    # test_result(x, plane0, plane1, plane2)
    return x


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

    def key(self, vertex):
        for i in range(len(self.points)):
            if self.points[i][0] == vertex:
                return i
        return len(self.points)


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


# Returns the area of a polygon with the given vertices in order.
# Assumes all the vertices are coplanar.
def area(vertices):
    v0 = vertices[0]
    vecs = [(delta_vector(v0, v)) for v in vertices[1:]]
    return 0.5 * sum(
        cross_product(vecs[i - 1], vecs[i]).magnitude()
        for i in range(1, len(vecs))
    )


def test():
    p0 = Vertex(0, 0, 40)
    p1 = Vertex(10, 0, 40)
    p2 = Vertex(10, 10, 40)
    print(area((p0, p1, p2)))
    print(area((p1, p2, p0)))
    print(area((p2, p0, p1)))


pole_area = 0
barrel_area = 0


def brute_faces(half_spaces):
    global pole_area
    global barrel_area
    points = PointIntern()
    for i in range(0, len(half_spaces)):
        half_i = half_spaces[i]
        for j in range(i + 1, len(half_spaces)):
            half_j = half_spaces[j]
            for k in range(j + 1, len(half_spaces)):
                half_k = half_spaces[k]
                vertex = tripoint(half_i, half_j, half_k)
                if vertex is not None:
                    zzz = points.insert(vertex, (i, j, k))
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

    # Locate all edges.
    face_edges = [[] for _ in range(len(half_spaces))]
    for i in range(len(result)):
        v_i, f_i = result[i]
        for j in range(i + 1, len(result)):
            v_j, f_j = result[j]
            faces = f_i.intersection(f_j)
            match len(faces):
                case 2:
                    f = list(faces)
                    face_edges[f[0]].append((v_i, v_j, f[1]))
                    face_edges[f[1]].append((v_i, v_j, f[0]))
                case _:
                    pass

    result = []
    pole_area = 0
    barrel_area = 0
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
                print (f"Error on face {z+1}")
                raise RuntimeError("face failure")
        face_area = area(vertices)
        if len(result) <= 9:
            pole_area += face_area
        else:
            barrel_area += face_area
        result.append((vertices, faces))
    pole_area /= 10
    barrel_area /= 8
    return result


def find_vertex(this_face, other_face, poly):
    v, f = poly[this_face]
    return v[f.index(other_face)]


class better_pattern:
    def __init__(self, poly):
        self.poly = poly
        self.is_drawn = len(poly) * [False]
        self.plotted = {}
        self.candidates = []

    def make_pattern(self):
        with cairo.RecordingSurface(cairo.CONTENT_COLOR_ALPHA, None) as rs:
            ctx = cairo.Context(rs)
            ctx.scale(100.0, -100.0)
            ctx.set_line_width(0.05)
            ctx.set_source_rgba(0, 0, 0, 1)
            ctx.set_font_size(0.5)
            ctx.select_font_face(
                "Arial", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL
            )
            self.ctx = ctx

            self.do_plot()

            x, y, width, height = rs.ink_extents()
            surface = cairo.SVGSurface(
                "C:/Users/sigma/Documents/polygon.svg", width, height
            )
            ccc = cairo.Context(surface)
            ccc.set_source_surface(rs, -x, -y)
            ccc.paint()
            surface.finish()
            surface.flush()

    def draw_and_append(self, this_face, from_face):
        # print(f"Drawing {this_face}")
        plot_face(self.poly, this_face, from_face, self.plotted, self.ctx)
        self.is_drawn[this_face] = True
        for adj in self.poly[this_face][1]:
            if not self.is_drawn[adj]:
                edge_length = delta_vector(
                    find_vertex(this_face, adj, self.poly),
                    find_vertex(adj, this_face, self.poly),
                ).magnitude()
                self.candidates.append((edge_length, this_face, adj))
        # print("Candidates now")
        # for length, this_f, adj_f in self.candidates:
        #    print(f"{length}: {this_f} -> {adj_f}")

    def do_plot(self):
        self.draw_and_append(0, None)
        while self.candidates:
            max_edge = None
            for i in range(len(self.candidates)):
                edge_length = self.candidates[i][0]
                if max_edge is None or edge_length > max_edge:
                    max_edge = edge_length
                    next_i = i
            _, from_face, this_face = self.candidates[next_i]
            self.candidates = list(
                filter(lambda c: c[2] != this_face, self.candidates)
            )
            self.draw_and_append(this_face, from_face)


def centroid(points):
    n = len(points)
    return (sum(p[0] for p in points) / n, sum(p[1] for p in points) / n)


def plot_face(poly, this_face, from_face, plotted, ctx):
    vertices, faces = poly[this_face]
    assert len(vertices) == len(faces)
    assert len(vertices) >= 3

    if from_face is None:
        # Find the coordinates of all the vertices of the face.
        plist = [
            (0.0, 0.0),
            (delta_vector(vertices[0], vertices[1]).magnitude(), 0.0),
        ]
        for i in range(2, len(vertices)):
            plist.append(
                find_next_point(*vertices[i - 2 : i + 1], plist[-2], plist[-1])
            )

        assert len(vertices) == len(plist)
        for v, c in zip(vertices, plist):
            plotted[(this_face, v)] = c

        ctx.move_to(*plist[0])
        for c in plist[1:]:
            ctx.line_to(*c)
        ctx.close_path()
        ctx.stroke()

    else:
        pos = faces.index(from_face)

        # Reorder the vertices so that vertices[0] and vertices[1]
        # are also in from_face.
        vertices = vertices[pos:] + vertices[:pos]

        plist = [plotted[(from_face, vertices[i])] for i in range(2)]
        for i in range(2, len(vertices)):
            plist.append(
                find_next_point(*vertices[i - 2 : i + 1], plist[-2], plist[-1])
            )
        assert len(vertices) == len(plist)
        for v, c in zip(vertices, plist):
            plotted[(this_face, v)] = c

        ctx.move_to(*plist[1])
        for c in plist[2:]:
            ctx.line_to(*c)
        ctx.line_to(*plist[0])
        ctx.stroke()

    # Label the face
    text = f"{1+this_face}"
    x_bearing, y_bearing, width, height, x_advance, y_advance = (
        ctx.text_extents(text)
    )
    ctx.save()
    cen = centroid(plist)
    ctx.translate(*cen)
    ctx.scale(1, -1)
    ctx.move_to(-width / 2, height / 2)
    ctx.show_text(text)
    ctx.restore()


# To map the 3d coordinates of the vertices to 2d, we define an
# auxiliary attribute "plotted" to each Vertex to store its 2d
# coordinates. If v0, v1, and v2 are three successive (counterclockwise)
# vertices of a face in 3-space, and v0, and v1 have been plotted,
# determine the 2-d coordinates of v2, and store them as the "plotted"
# attribute.
def find_next_point(v0, v1, v2, p0, p1):
    # Get the sin and cos of the angle between the angle defined
    # by v0, v1, v2 in 3-space
    vec01 = delta_vector(v0, v1)
    vec12 = delta_vector(v1, v2)
    cos_a = dot_product(vec01.value(), vec12.value()) / (
        vec01.magnitude() * vec12.magnitude()
    )
    sin_a = math.sqrt(1.0 - cos_a * cos_a)

    # Get the sin and cos of the angle of the line segment from p0 to p1
    d = distance(p0, p1)
    sin_b = (p1[1] - p0[1]) / d
    cos_b = (p1[0] - p0[0]) / d

    # Compute the sin and cos of a+b
    sin_sum = sin_a * cos_b + cos_a * sin_b
    cos_sum = cos_a * cos_b - sin_a * sin_b

    d = vec12.magnitude()
    x = p1[0] + d * cos_sum
    y = p1[1] + d * sin_sum

    return (x, y)


# p0 and p1 are the last 2-d points drawn.
# e0 and e1 are consecutive edges
# determine and return p2, the next 2-d point.
def next_point(p0, p1, edge0, edge1):
    # Get the sin and cos of the angle between the two edges.

    vec0 = delta_vector(edge0.e1, edge0.e0)
    vec1 = delta_vector(edge1.e0, edge1.e1)
    cos_a = dot_product(vec0.value(), vec1.value()) / (
        vec0.magnitude() * vec1.magnitude()
    )
    sin_a = math.sqrt(1.0 - cos_a * cos_a)

    # Get the sin and cos of the angle of the line segment from p0 to p1
    d = distance(p0, p1)
    sin_b = (p1[1] - p0[1]) / d
    cos_b = (p1[0] - p0[0]) / d

    # Compute the sin and cos of a-b
    sin_diff = sin_a * cos_b - cos_a * sin_b
    cos_diff = cos_a * cos_b + sin_a * sin_b

    x = p1[0] - d * cos_diff
    y = p1[1] + d * sin_diff

    return (x, y)


def plot(faces, face_id, from_id, ctx):
    face = faces[face_id]
    if from_id is None:
        p0 = (0.0, 0.0)
        ctx.move_to(*p0)
        p1 = (delta_vector(face[0].e0, face[0].e1).magnitude(), 0.0)
        face[0].plotted = p1
        ctx.line_to(*p1)
        joined = face
        prev_edge = face[0]
        for edge in face[1:]:
            next_p = next_point(p0, p1, prev_edge, edge)
            prev_edge = edge
            ctx.line_to(*next_p)
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


# Make a pattern for a cube
def cube():
    x_hi = [1.0, 0.0, 0.0, -1.0]
    x_lo = [-1.0, 0.0, 0.0, -1.0]
    y_hi = [0.0, 1.0, 0.0, -1.0]
    y_lo = [0.0, -1.0, 0.0, -1.0]
    z_hi = [0, 0.0, 1.0, -1.0]
    z_lo = [0.0, 0.0, -1.0, -1.0]

    plot_pattern(brute_faces([x_lo, x_hi, y_lo, y_hi, z_lo, z_hi]))


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
    plot_pattern(brute_faces(halves))


def tweak():
    with open("points18.json", "r") as file:
        halves = json.load(file)
    for h in halves:
        h.append(-1.0)

    adjust_min = True
    while True:
        poly = brute_faces(halves)
        min_area = None
        max_area = None
        for i in range(0, len(poly), 2):
            vertices = poly[i][0]
            a = area(vertices)
            if min_area is None or a < min_area:
                min_area = a
                min_index = i
            if max_area is None or a > max_area:
                max_area = a
                max_index = i
        ratio = 1 + (max_area / min_area - 1.0) / 10
        if ratio < 1.0000001:
            better_pattern(poly).make_pattern()
            break
        if adjust_min:
            halves[min_index][3] /= ratio
            halves[min_index + 1][3] /= ratio
        else:
            halves[max_index][3] *= ratio
            halves[max_index + 1][3] *= ratio
        adjust_min = not adjust_min


def nudge(halves):
    poly = brute_faces(halves)

    # Compute the area of every face
    areas = [area(v) for v, _ in poly]

    # Iterate over every edge
    max_ratio = None
    for i, (vertices, faces) in enumerate(poly):
        for f in faces:
            ratio = areas[i] / areas[f]
            if max_ratio is None or ratio > max_ratio:
                max_ratio = ratio
                max_i = i
                max_f = f
    print("max_ratio", max_ratio, "at", max_i, max_f)
    nudge_value = 1 + (max_ratio - 1.0) / 10.0

    # Nudge halves[f] towards halves[i] along the great circle
    # that connects the two points.
    # See https://www.johndcook.com/blog/2021/11/26/great-circle-equation/
    vec_i = Vector(*halves[i][0:3])
    vec_f = Vector(*halves[f][0:3])
    z = cross_product(vec_i, vec_f)

    # Get the angle from vec_i to vec_f
    sin_theta = z.magnitude()
    cos_theta = dot_product(vec_i.value(), vec_f.value())
    theta = math.atan2(sin_theta, cos_theta)

    u = cross_product(z, vec_i).normalize()

    # Increase theta a little
    t = theta / nudge_value
    nudged_f = vec_i.scale(math.cos(t)).add(u.scale(math.sin(t)))

    halves[f][0:3] = nudged_f.value()


def read_halves(filename):
    halves = []
    buffer = []
    with open(filename, "rb") as file:
        while True:
            bytes = file.read(8)
            if not bytes:
                break
            buffer.append((struct.unpack("d", bytes))[0])
            if len(buffer) == 4:
                halves.append(tuple(buffer))
                buffer.clear()
    assert len(buffer) == 0
    return halves


def main():
    halves = []
    buffer = []
    with open("c:/users/sigma/documents/halves.bin", "rb") as file:
        while True:
            bytes = file.read(8)
            if not bytes:
                break
            buffer.append((struct.unpack("d", bytes))[0])
            if len(buffer) == 4:
                halves.append(tuple(buffer))
                buffer.clear()
    assert len(buffer) == 0
    assert len(halves) == 18
    poly = brute_faces(halves)
    better_pattern(poly).make_pattern()


def attempt(phi):
    coords = []
    #           1     3     5     7     9
    weights = [8390, 8670, 8030, 8760, 8200]
    total = float(sum(weights))
    for i in range(5):
        angle = 2.0 * math.pi * sum(weights[: i + 1]) / total
        coords.append((angle, phi))
    equator = math.pi / 2
    #          11    13    15     17
    weights = [950, 920, 930, 950]
    total = float(sum(weights))
    for i in range(4):
        angle = math.pi * sum(weights[: i + 1]) / total
        coords.append((angle, equator))

    # Convert to Cartesian
    halves = []
    for theta, phi in coords:
        sin_phi = math.sin(phi)
        x = math.cos(theta) * sin_phi
        y = math.sin(theta) * sin_phi
        z = math.cos(phi)
        halves.append((x, y, z, -1.0))
        halves.append((-x, -y, -z, -1.0))

    poly = brute_faces(halves)
    return poly


def custom():
    left = 35.0 * math.pi / 180.0
    right = 40 * math.pi / 180.0
    attempt(left)
    attempt(right)
    for i in range(20):
        midpoint = 0.5 * (left + right)
        poly = attempt(midpoint)
        if pole_area > barrel_area:
            right = midpoint
        else:
            left = midpoint

    for i, p in enumerate(poly):
        print(f"Face {i+1} has {len(p[0])} sides")
    areas = [area(p[0]) for p in poly]
    average = sum(areas) / len(poly)
    signif = [1, 3, 5, 7, 9, 11, 13, 15, 17]
    for i in signif:
        diff = areas[i - 1] - average
        print(f"Face {i} {'+' if diff > 0 else '-'}{round(10000*abs(diff))}")
    # better_pattern(poly).make_pattern()
    return poly


def debug_model():
    halves = read_halves("c:/users/sigma/documents/bad_model.bin")
    better_pattern(brute_faces(halves)).make_pattern()


def topper():
    coords = []
    phi = 40.0 * math.pi / 180.0
    for i in range(0, 18, 2):
        coords.append((i * 2.0 * math.pi / 18.0, phi))
    phi = math.pi - phi
    for i in range(1, 19, 2):
        coords.append((i * 2.0 * math.pi / 18.0, phi))

    # Convert to cartesian coordinates
    halves = []
    for theta, phi in coords:
        sin_phi = math.sin(phi)
        x = math.cos(theta) * sin_phi
        y = math.sin(theta) * sin_phi
        z = math.cos(phi)
        halves.append((x, y, z, -1.0))
        halves.append((-x, -y, z, -1.0))
    better_pattern(brute_faces(halves)).make_pattern()


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
    debug_model()
