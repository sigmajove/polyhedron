import cairo
import collections
import copy
import json
import math
import numpy
import struct
from collections.abc import Iterable
from fontTools.ttLib import TTFont
from fontTools.pens.boundsPen import BoundsPen

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

    # Returns a vector whose head is self and tail is other.
    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

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
    def __add__(self, v):
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

    print("me", result)
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


def make_polygon(half_spaces):
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
                print(f"Error on face {z+1}")
                raise RuntimeError("face failure")
        face_area = area(vertices)
        if len(result) <= 9:
            pole_area += face_area
        else:
            barrel_area += face_area

        # Unit vector pointing out of the half space
        normal = Vector(*half_spaces[z][0:3]).normalize()

        result.append((vertices, faces, normal))
    pole_area /= 10
    barrel_area /= 8
    return result


def find_vertex(this_face, other_face, poly):
    v, f = poly[this_face]
    return v[f.index(other_face)]


def start_draw():
    rs = cairo.RecordingSurface(cairo.CONTENT_COLOR_ALPHA, None)
    ctx = cairo.Context(rs)
    ctx.scale(1, -1)
    ctx.set_line_width(0.05)
    ctx.set_source_rgba(0, 0, 0, 1)
    return (rs, ctx)


def finish_draw(rs, ctx, filename):
    x, y, width, height = rs.ink_extents()
    surface = cairo.SVGSurface(
        f"C:/Users/sigma/Documents/{filename}.svg", width, height
    )
    ccc = cairo.Context(surface)
    ccc.set_source_surface(rs, -x, -y)
    ccc.paint()
    surface.flush()
    surface.finish()
    del ccc
    del ctx
    rs.finish()


class print_faces:
    def __init__(self, poly):
        self.poly = poly
        self.first = True
        self.y_top = 0.0

        rs = cairo.RecordingSurface(cairo.CONTENT_COLOR_ALPHA, None)
        ctx = cairo.Context(rs)
        ctx.scale(100.0, -100.0)
        ctx.set_line_width(0.02)
        ctx.set_source_rgba(0, 0, 0, 1)
        self.rs = rs
        self.ctx = ctx

    def finish(self):
        x, y, width, height = self.rs.ink_extents()
        surface = cairo.SVGSurface(
            "C:/Users/sigma/Documents/faces.svg", width, height
        )
        ccc = cairo.Context(surface)
        ccc.set_source_surface(self.rs, -x, -y)
        ccc.paint()
        self.rs.finish()
        surface.finish()
        surface.flush()
        del ccc
        del self.ctx

    def find_vertex(self, faces):
        result = None
        for f in faces:
            vertices = self.poly[f][0]
            v_set = set(v for v in vertices)
            if result is None:
                result = v_set
            else:
                result = result.intersection(v_set)
        assert len(result) == 1
        self.vertex = next(iter(result))

    def rotate_face(self, i):
        vertices = self.poly[i][0]
        apex = vertices.index(self.vertex)
        prev = vertices[(apex - 1) % len(vertices)]
        next = vertices[(apex + 1) % len(vertices)]
        vx0 = Vertex(*prev.value())
        vx1 = Vertex(*self.vertex.value())
        vx2 = Vertex(*next.value())

        y_axis = (
            ((vx1 - vx0).normalize() + (vx1 - vx2).normalize())
            .scale(0.5)
            .normalize()
        )

        x_axis = cross_product(y_axis, self.poly[i][2])
        flat = []
        min_y = math.inf
        max_y = -math.inf
        for vx in vertices:
            x = dot_product(x_axis.value(), vx.value())
            y = dot_product(y_axis.value(), vx.value())
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y
            flat.append([x, y])

        # Shift left/right so the x coordinate of the apex is zero
        align = flat[apex][0]
        updown = self.y_top - max_y
        for f in flat:
            f[0] -= align
            f[1] += updown

        self.ctx.move_to(*flat[0])
        for xy in flat[1:]:
            self.ctx.line_to(*xy)
        self.ctx.close_path()
        self.ctx.stroke()
        self.first = False

        self.y_top -= max_y - min_y + 0.25

    def poles(self, faces):
        self.find_vertex(faces)
        for f in faces:
            self.rotate_face(f)

    def barrel(self, i):
        print("Barrel")
        vertices = self.poly[i][0]
        prev = vertices[-1]
        top = None
        bottom = None
        for v in vertices:
            if prev.z > 0 and v.z < 0:
                assert top is None
                top = copy.deepcopy(v)
                top_length = (v - prev).magnitude()
                top.z = 0.0

            if prev.z < 0 and v.z > 0:
                assert bottom is None
                bottom = copy.deepcopy(v)
                bottom_length = (v - prev).magnitude()
                bottom.z = 0.0
            prev = v
        assert bottom is not None
        assert top is not None

        y_axis = (top - bottom).normalize()
        x_axis = cross_product(y_axis, self.poly[i][2])

        flat = []
        min_y = math.inf
        max_y = -math.inf
        for vx in vertices:
            x = dot_product(x_axis.value(), vx.value())
            y = dot_product(y_axis.value(), vx.value())
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y
            flat.append([x, y])

        updown = self.y_top - max_y

        tx = dot_product(x_axis.value(), top.value())
        ty = dot_product(y_axis.value(), top.value()) + updown

        bx = dot_product(x_axis.value(), bottom.value())
        by = dot_product(y_axis.value(), bottom.value()) + updown

        for f in flat:
            f[1] += updown

        self.ctx.move_to(*flat[0])
        for xy in flat[1:]:
            self.ctx.line_to(*xy)
        self.ctx.close_path()
        self.ctx.stroke()
        self.first = False

        self.y_top -= max_y - min_y + 0.25


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

    plot_pattern(make_polygon([x_lo, x_hi, y_lo, y_hi, z_lo, z_hi]))


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
    plot_pattern(make_polygon(halves))


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
    poly = make_polygon(halves)
    better_pattern(poly).make_pattern()


def attempt(phi, weights):
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

    # Convert to Cartesian
    halves = []
    for theta, phi in coords:
        sin_phi = math.sin(phi)
        x = math.cos(theta) * sin_phi
        y = math.sin(theta) * sin_phi
        z = math.cos(phi)
        halves.append((x, y, z, -1.0))
        halves.append((-x, -y, -z, -1.0))

    poly = make_polygon(halves)
    return poly


def iteration(weights):
    left = 35.0 * math.pi / 180.0
    right = 40 * math.pi / 180.0
    attempt(left, weights)
    attempt(right, weights)
    for _ in range(20):
        midpoint = 0.5 * (left + right)
        poly = attempt(midpoint, weights)
        if pole_area > barrel_area:
            right = midpoint
        else:
            left = midpoint
    return poly


def two_thirds(oncurve, ctrl):
    return ((oncurve[0] + 2 * ctrl[0]) / 3.0, (oncurve[1] + 2 * ctrl[1]) / 3.0)


class AdjustPen:
    def __init__(self, x_min, y_min):
        self.x_min = x_min
        self.y_min = y_min

    def move_to(self, p):
        x, y = p
        print(
            (
                f"                     "
                f"self.pen.move_to(({x-self.x_min:.2f}, {y-self.y_min:.2f}))"
            )
        )

    def line_to(self, p):
        x, y = p
        print(
            (
                f"                     "
                f"self.pen.line_to(({x-self.x_min:.2f}, {y-self.y_min:.2f}))"
            )
        )

    def curve_to(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        print(
            (
                f"                     "
                f"self.pen.curve_to(({x1-self.x_min:.2f}, {y1-self.y_min:.2f}), "
                f"({x2-self.x_min:.2f}, {y2-self.y_min:.2f})) "
            )
        )

    def close_path(self):
        print((f"                     " f"self.pen.close_path()"))


class BoundingBoxPen:
    def __init__(self):
        self.reset()

    def reset(self):
        self.scale = 1.0
        self.min_x = math.inf
        self.max_x = -math.inf
        self.min_y = math.inf
        self.max_y = -math.inf
        self.last_p = None

    def scale_point(self, p):
        return p

    def record(self, p):
        if p[0] > self.max_x:
            self.max_x = p[0]
        if p[0] < self.min_x:
            self.min_x = p[0]
        if p[1] > self.max_y:
            self.max_y = p[1]
        if p[1] < self.min_y:
            self.min_y = p[1]
        self.last_p = p

    def moveTo(self, p):
        self.move_to(p)

    def move_to(self, p):
        self.record(self.scale_point(p))

    def lineTo(self, p):
        self.line_to(p)

    def line_to(self, p):
        self.record(self.scale_point(p))

    def bounds(self):
        return (self.min_x, self.max_x, self.min_y, self.max_y)

    def curve_to(self, p2, p1):
        # https://www.shadertoy.com/view/lsyfWc
        p0 = self.last_p
        assert p0 is not None

        # extremes
        mi = [min(a, b) for a, b in zip(p0, p2)]
        ma = [max(a, b) for a, b in zip(p0, p2)]

        # if p1 is outside the current bbox/hull
        if any(p1[i] < mi[i] or p1[i] > ma[i] for i in range(2)):
            # See https://iquilezles.org/articles/bezierbbox/
            # p = (1-t)^2*p0 + 2(1-t)t*p1 + t^2*p2
            # dp/dt = 2(t-1)*p0 + 2(1-2t)*p1 + 2t*p2 =
            #     t*(2*p0-4*p1+2*p2) + 2*(p1-p0)
            # dp/dt = 0 -> t*(p0-2*p1+p2) = (p0-p1);

            for i in range(2):
                t = (p0[i] - p1[i]) / (p0[i] - 2 * p1[i] + p2[i])
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0
                s = 1.0 - t
                q = s * s * p0[i] + 2 * s * t * p1[i] + t * t * p2[i]

                if q < mi[i]:
                    mi[i] = q
                if q > ma[i]:
                    ma[i] = q

        if ma[0] > self.max_x:
            self.max_x = ma[0]
        if mi[0] < self.min_x:
            self.min_x = mi[0]
        if ma[1] > self.max_y:
            self.max_y = ma[1]
        if mi[1] < self.min_y:
            self.min_y = mi[1]

        self.last_p = p1

    def qCurveTo(self, *args):
        assert len(args) == 2 or len(args) == 3
        scaled = [self.scale_point(a) for a in args]
        if len(scaled) == 2:
            self.curve_to(scaled[0], scaled[1])
        else:
            implicit = tuple(
                0.5 * (scaled[0][i] + scaled[1][i]) for i in range(2)
            )
            self.curve_to(scaled[0], implicit)
            self.curve_to(scaled[1], scaled[2])

    def closePath(self):
        pass

    def close_path(self):
        pass


class CairoPen:
    def __init__(self, rs, ctx):
        self.rs = rs
        self.ctx = ctx
        self.x_offset = 0

    def set_trim_x(self, xmin):
        self.trim_x = xmin

    def advance_x(self, dx):
        self.x_offset += dx

    def scale_point(self, p):
        return (
            p[0] - self.trim_x + self.x_offset,
            p[1],
        )

    def moveTo(self, p0):
        self.ctx.move_to(*self.scale_point(p0))

    def lineTo(self, p1):
        self.ctx.line_to(*self.scale_point(p1))

    def oneCurve(self, on, off):
        ct1 = two_thirds(self.ctx.get_current_point(), off)
        ct2 = two_thirds(on, off)
        self.ctx.curve_to(*ct1, *ct2, *on)

    def qCurveTo(self, *args):
        assert len(args) == 2 or len(args) == 3
        scaled = [self.scale_point(a) for a in args]
        if len(scaled) == 2:
            self.oneCurve(scaled[1], scaled[0])
        else:
            implicit = tuple(
                0.5 * (scaled[0][i] + scaled[1][i]) for i in range(2)
            )
            self.oneCurve(implicit, scaled[0])
            self.oneCurve(scaled[2], scaled[1])

    def closePath(self):
        self.ctx.close_path()
        self.ctx.stroke()


class RecordingPen:
    def __init__(self, rs, ctx):
        self.rs = rs
        self.ctx = ctx

    def set_trim_x(self, xmin):
        self.trim_x = xmin

    def scale_point(self, p):
        return (p[0] - self.trim_x, p[1])

    def moveTo(self, p0):
        xxx = self.scale_point(p0)
        self.ctx.move_to(*xxx)
        print(
            (
                f"                         "
                f"self.pen.move_to(({xxx[0]:.2f}, {xxx[1]:.2f}))"
            )
        )

    def lineTo(self, p1):
        xxx = self.scale_point(p1)
        self.ctx.line_to(*xxx)
        print(
            (
                f"                         "
                f"self.pen.line_to(({xxx[0]:.2f}, {xxx[1]:.2f}))"
            )
        )

        self.ctx.line_to(*self.scale_point(p1))

    def oneCurve(self, on, off):
        print(
            (
                f"                         "
                f"self.pen.curve_to(({on[0]:.2f}, {on[1]:.2f}),"
                f"({off[0]:.2f}, {off[1]:.2f})),"
            )
        )
        ct1 = two_thirds(self.ctx.get_current_point(), off)
        ct2 = two_thirds(on, off)
        self.ctx.curve_to(*ct1, *ct2, *on)

    def qCurveTo(self, *args):
        assert len(args) == 2 or len(args) == 3
        scaled = [self.scale_point(a) for a in args]
        if len(scaled) == 2:
            self.oneCurve(scaled[1], scaled[0])
        else:
            implicit = tuple(
                0.5 * (scaled[0][i] + scaled[1][i]) for i in range(2)
            )
            self.oneCurve(implicit, scaled[0])
            self.oneCurve(scaled[2], scaled[1])

    def closePath(self):
        print((f"                         " "self.pen.close_path()"))

        self.ctx.close_path()
        self.ctx.stroke()


SPACING = 100 
class Font:
    def __init__(self, pen):
        self.pen = pen

    def d0(self):
        self.pen.move_to((706.00, 747.00))
        self.pen.curve_to((666.50, 1135.00), (706.00, 1009.00))
        self.pen.curve_to((500.00, 1261.00), (627.00, 1261.00))
        self.pen.curve_to((332.00, 1135.00), (373.00, 1261.00))
        self.pen.curve_to((291.00, 747.00), (291.00, 1009.00))
        self.pen.curve_to((332.00, 364.50), (291.00, 497.00))
        self.pen.curve_to((500.00, 232.00), (373.00, 232.00))
        self.pen.curve_to((666.50, 364.50), (627.00, 232.00))
        self.pen.curve_to((706.00, 747.00), (706.00, 497.00))
        # self.pen.close_path()
        self.pen.line_to((1000.00, 747.00))
        self.pen.curve_to((888.50, 194.00), (1000.00, 388.00))
        self.pen.curve_to((500.00, 0.00), (777.00, 0.00))
        self.pen.curve_to((111.50, 194.00), (223.00, 0.00))
        self.pen.curve_to((0.00, 747.00), (0.00, 388.00))
        self.pen.curve_to((111.50, 1301.00), (0.00, 1106.00))
        self.pen.curve_to((500.00, 1496.00), (223.00, 1496.00))
        self.pen.curve_to((888.50, 1301.00), (777.00, 1496.00))
        self.pen.curve_to((1000.00, 747.00), (1000.00, 1106.00))
        self.pen.close_path()
        return (1000.00, 1496.00)

    def d1(self):
        self.pen.move_to((0.00, 1036.00))
        self.pen.line_to((0.00, 1230.00))
        self.pen.curve_to((189.00, 1248.00), (135.00, 1236.00))
        self.pen.curve_to((329.00, 1324.00), (275.00, 1267.00))
        self.pen.curve_to((385.00, 1428.00), (366.00, 1363.00))
        self.pen.curve_to((396.00, 1496.00), (396.00, 1467.00))
        self.pen.line_to((633.00, 1492.00))
        self.pen.line_to((633.00, 0.00))
        self.pen.line_to((341.00, 0.00))
        self.pen.line_to((341.00, 1036.00))
        self.pen.close_path()
        return (633.00, 1496.00)

    def d2(self):
        self.pen.move_to((67.00, 321.00))
        self.pen.curve_to((355.00, 628.00), (128.00, 466.00))
        self.pen.curve_to((610.00, 830.00), (552.00, 769.00))
        self.pen.curve_to((699.00, 1038.00), (699.00, 925.00))
        self.pen.curve_to((648.00, 1191.00), (699.00, 1130.00))
        self.pen.curve_to((502.00, 1252.00), (597.00, 1252.00))
        self.pen.curve_to((325.00, 1155.00), (372.00, 1252.00))
        self.pen.curve_to((293.00, 977.00), (298.00, 1099.00))
        self.pen.line_to((16.00, 977.00))
        self.pen.curve_to((83.00, 1276.00), (23.00, 1162.00))
        self.pen.curve_to((488.00, 1496.00), (197.00, 1496.00))
        self.pen.curve_to((854.00, 1365.50), (718.00, 1496.00))
        self.pen.curve_to((990.00, 1028.00), (990.00, 1238.00))
        self.pen.curve_to((894.00, 742.00), (990.00, 867.00))
        self.pen.curve_to((687.00, 557.00), (831.00, 659.00))
        self.pen.line_to((573.00, 476.00))
        self.pen.curve_to((426.50, 366.00), (466.00, 400.00))
        self.pen.curve_to((360.00, 251.00), (387.00, 332.00))
        self.pen.line_to((993.00, 251.00))
        self.pen.line_to((993.00, 0.00))
        self.pen.line_to((0.00, 0.00))
        self.pen.curve_to((67.00, 321.00), (4.00, 192.00))
        self.pen.close_path()
        return (993.00, 1496.00)

    def d3(self):
        self.pen.move_to((280.00, 481.00))
        self.pen.curve_to((308.00, 337.00), (280.00, 394.00))
        self.pen.curve_to((497.00, 232.00), (360.00, 232.00))
        self.pen.curve_to((643.50, 289.50), (581.00, 232.00))
        self.pen.curve_to((706.00, 455.00), (706.00, 347.00))
        self.pen.curve_to((590.00, 646.00), (706.00, 598.00))
        self.pen.curve_to((382.00, 673.00), (524.00, 673.00))
        self.pen.line_to((382.00, 877.00))
        self.pen.curve_to((576.00, 904.00), (521.00, 879.00))
        self.pen.curve_to((671.00, 1074.00), (671.00, 946.00))
        self.pen.curve_to((622.50, 1209.00), (671.00, 1157.00))
        self.pen.curve_to((486.00, 1261.00), (574.00, 1261.00))
        self.pen.curve_to((337.50, 1197.00), (385.00, 1261.00))
        self.pen.curve_to((292.00, 1026.00), (290.00, 1133.00))
        self.pen.line_to((26.00, 1026.00))
        self.pen.curve_to((63.00, 1231.00), (30.00, 1134.00))
        self.pen.curve_to((173.00, 1388.00), (98.00, 1316.00))
        self.pen.curve_to((306.00, 1466.00), (229.00, 1439.00))
        self.pen.curve_to((495.00, 1496.00), (383.00, 1496.00))
        self.pen.curve_to((830.50, 1385.50), (703.00, 1496.00))
        self.pen.curve_to((958.00, 1097.00), (958.00, 1278.00))
        self.pen.curve_to((882.00, 881.00), (958.00, 969.00))
        self.pen.curve_to((782.00, 806.00), (834.00, 826.00))
        self.pen.curve_to((894.00, 739.00), (821.00, 806.00))
        self.pen.curve_to((1003.00, 463.00), (1003.00, 638.00))
        self.pen.curve_to((875.50, 139.50), (1003.00, 279.00))
        self.pen.curve_to((498.00, 0.00), (748.00, 0.00))
        self.pen.curve_to((70.00, 201.00), (190.00, 0.00))
        self.pen.curve_to((0.00, 481.00), (7.00, 308.00))
        self.pen.close_path()
        return (1003.00, 1496.00)

    def d4(self):
        y0 = 0
        y1 = 348
        y2 = 605
        y3 = 1200
        y4 = 1496
        x0 = 0
        x1 = 270
        x2 = 533
        x3 = 632
        x4 = 890
        x5 = 1120
        self.pen.move_to((x3, y1))

        self.pen.line_to((x0, y1))
        self.pen.line_to((x0, y2))
        self.pen.line_to((x2, y4))
        self.pen.line_to((x4, y4))
        self.pen.line_to((x4, y2))
        self.pen.line_to((x5, y2))
        self.pen.line_to((x5, y1))
        self.pen.line_to((x4, y1))
        self.pen.line_to((x4, y0))
        self.pen.line_to((x3, y0))
        self.pen.line_to((x3, y1))

        self.pen.line_to((x3, y2))
        self.pen.line_to((x3, y3))
        self.pen.line_to((x1, y2))
        self.pen.line_to((x3, y2))

        self.pen.close_path()
        return (1120.00, 1496.00)

    def d5(self):
        self.pen.move_to((284.00, 424.00))
        self.pen.curve_to((349.00, 280.50), (301.00, 331.00))
        self.pen.curve_to((489.00, 230.00), (397.00, 230.00))
        self.pen.curve_to((650.50, 304.50), (595.00, 230.00))
        self.pen.curve_to((706.00, 492.00), (706.00, 379.00))
        self.pen.curve_to((654.00, 679.50), (706.00, 603.00))
        self.pen.curve_to((492.00, 756.00), (602.00, 756.00))
        self.pen.curve_to((402.00, 743.00), (440.00, 756.00))
        self.pen.curve_to((301.00, 654.00), (335.00, 719.00))
        self.pen.line_to((45.00, 666.00))
        self.pen.line_to((147.00, 1496.00))
        self.pen.line_to((946.00, 1496.00))
        self.pen.line_to((946.00, 1225.00))
        self.pen.line_to((353.00, 1225.00))
        self.pen.line_to((301.00, 908.00))
        self.pen.curve_to((404.00, 965.00), (367.00, 951.00))
        self.pen.curve_to((555.00, 988.00), (466.00, 988.00))
        self.pen.curve_to((869.00, 867.00), (735.00, 988.00))
        self.pen.curve_to((1003.00, 515.00), (1003.00, 746.00))
        self.pen.curve_to((874.00, 156.00), (1003.00, 314.00))
        self.pen.curve_to((488.00, 0.00), (745.00, 0.00))
        self.pen.curve_to((148.00, 109.00), (281.00, 0.00))
        self.pen.curve_to((0.00, 424.00), (15.00, 220.00))
        self.pen.close_path()
        return (1003.00, 1496.00)

    def d6(self):
        self.pen.move_to((286, 495))

        self.pen.curve_to((349.00, 304.00), (286.00, 378.00))
        self.pen.curve_to((509.00, 230.00), (412.00, 230.00))
        self.pen.curve_to((658.50, 301.50), (604.00, 230.00))
        self.pen.curve_to((713.00, 487.00), (713.00, 373.00))
        self.pen.curve_to((651.00, 681.50), (713.00, 614.00))
        self.pen.curve_to((499.00, 749.00), (589.00, 749.00))
        self.pen.curve_to((370.00, 705.00), (426.00, 749.00))
        self.pen.curve_to((286.00, 495.00), (286.00, 640.00))
        self.pen.line_to((0, 689))

        self.pen.curve_to((14.00, 959.00), (0.00, 855.00))
        self.pen.curve_to((111.00, 1267.00), (39.00, 1144.00))
        self.pen.curve_to((273.50, 1436.00), (173.00, 1372.00))
        self.pen.curve_to((514.00, 1496.00), (374.00, 1496.00))
        self.pen.curve_to((836.00, 1396.50), (716.00, 1496.00))
        self.pen.curve_to((971.00, 1121.00), (956.00, 1293.00))
        self.pen.curve_to((660.00, 1198.00), (687.00, 1156.00))
        self.pen.curve_to((521.00, 1266.00), (614.00, 1266.00))
        self.pen.curve_to((323.00, 1110.00), (382.00, 1266.00))
        self.pen.curve_to((279.00, 856.00), (291.00, 1024.00))
        self.pen.curve_to((402.00, 948.00), (332.00, 919.00))
        self.pen.curve_to((562.00, 977.00), (472.00, 977.00))
        self.pen.curve_to((878.50, 846.00), (755.00, 977.00))
        self.pen.curve_to((1002.00, 511.00), (1002.00, 715.00))
        self.pen.curve_to((881.00, 153.00), (1002.00, 308.00))
        self.pen.curve_to((505.00, 0.00), (760.00, 0.00))
        self.pen.curve_to((101.00, 227.00), (231.00, 0.00))
        self.pen.curve_to((0.00, 689.00), (0.00, 406.00))
        self.pen.line_to((286, 495))
        self.pen.close_path()
        return (1002.00, 1496.00)

    def d7(self):
        self.pen.move_to((1028.00, 1244.00))
        self.pen.curve_to((850.00, 1019.50), (964.00, 1181.00))
        self.pen.curve_to((659.00, 686.00), (736.00, 858.00))
        self.pen.curve_to((549.00, 356.00), (598.00, 551.00))
        self.pen.curve_to((500.00, 0.00), (500.00, 161.00))
        self.pen.line_to((204.00, 0.00))
        self.pen.curve_to((460.00, 847.00), (217.00, 426.00))
        self.pen.curve_to((680.00, 1246.00), (617.00, 1108.00))
        self.pen.line_to((0.00, 1246.00))
        self.pen.line_to((4.00, 1496.00))
        self.pen.line_to((1028.00, 1496.00))
        self.pen.close_path()
        return (1028.00, 1496.00)

    def d8(self):
        self.pen.move_to((505.00, 232.00))
        self.pen.curve_to((658.50, 291.00), (603.00, 232.00))
        self.pen.curve_to((714.00, 457.00), (714.00, 350.00))
        self.pen.curve_to((657.50, 625.50), (714.00, 568.00))
        self.pen.curve_to((505.00, 683.00), (601.00, 683.00))
        self.pen.curve_to((352.50, 625.50), (409.00, 683.00))
        self.pen.curve_to((296.00, 457.00), (296.00, 568.00))
        self.pen.curve_to((351.50, 291.00), (296.00, 350.00))
        self.pen.curve_to((505.00, 232.00), (407.00, 232.00))
        self.pen.close_path()
        self.pen.move_to((218.00, 808.00))
        self.pen.curve_to((81.50, 959.50), (113.00, 878.00))
        self.pen.curve_to((50.00, 1112.00), (50.00, 1041.00))
        self.pen.curve_to((169.00, 1381.50), (50.00, 1270.00))
        self.pen.curve_to((505.00, 1496.00), (288.00, 1496.00))
        self.pen.curve_to((841.00, 1381.50), (722.00, 1496.00))
        self.pen.curve_to((960.00, 1112.00), (960.00, 1270.00))
        self.pen.curve_to((928.50, 959.50), (960.00, 1041.00))
        self.pen.curve_to((792.00, 818.00), (897.00, 878.00))
        self.pen.curve_to((953.00, 659.00), (899.00, 758.00))
        self.pen.curve_to((1007.00, 438.00), (1007.00, 560.00))
        self.pen.curve_to((871.50, 126.50), (1007.00, 255.00))
        self.pen.curve_to((493.00, 0.00), (736.00, 0.00))
        self.pen.curve_to((125.00, 126.50), (250.00, 0.00))
        self.pen.curve_to((0.00, 438.00), (0.00, 255.00))
        self.pen.curve_to((55.50, 659.00), (0.00, 560.00))
        self.pen.curve_to((218.00, 808.00), (111.00, 758.00))
        self.pen.close_path()
        self.pen.move_to((505.00, 901.00))
        self.pen.curve_to((639.50, 951.00), (592.00, 901.00))
        self.pen.curve_to((687.00, 1080.00), (687.00, 1001.00))
        self.pen.curve_to((639.50, 1214.50), (687.00, 1166.00))
        self.pen.curve_to((505.00, 1263.00), (592.00, 1263.00))
        self.pen.curve_to((370.50, 1214.50), (419.00, 1263.00))
        self.pen.curve_to((322.00, 1080.00), (322.00, 1166.00))
        self.pen.curve_to((370.50, 951.00), (322.00, 1001.00))
        self.pen.curve_to((505.00, 901.00), (419.00, 901.00))
        self.pen.close_path()
        return (1007.00, 1496.00)

    def d9(self):
        self.pen.move_to((484.00, 1496.00))
        self.pen.curve_to((938.00, 1205.00), (815.00, 1496.00))
        self.pen.curve_to((1008.00, 768.00), (1008.00, 1039.00))
        self.pen.curve_to((941.00, 329.00), (1008.00, 505.00))
        self.pen.curve_to((471.00, 0.00), (813.00, 0.00))
        self.pen.curve_to((178.00, 90.50), (308.00, 0.00))
        self.pen.curve_to((29.00, 372.00), (48.00, 187.00))
        self.pen.line_to((313.00, 372.00))
        self.pen.curve_to((367.00, 268.00), (323.00, 308.00))
        self.pen.curve_to((484.00, 228.00), (411.00, 228.00))
        self.pen.curve_to((682.00, 384.00), (625.00, 228.00))
        self.pen.curve_to((721.00, 635.00), (713.00, 470.00))
        self.pen.curve_to((638.00, 560.00), (682.00, 586.00))
        self.pen.curve_to((441.00, 512.00), (558.00, 512.00))
        self.pen.curve_to((134.00, 631.50), (268.00, 512.00))
        self.pen.curve_to((0.00, 976.00), (0.00, 751.00))
        self.pen.curve_to((134.50, 1353.50), (0.00, 1209.00))
        self.pen.curve_to((484.00, 1496.00), (269.00, 1496.00))
        self.pen.close_path()
        self.pen.move_to((614.00, 778.00))
        self.pen.curve_to((711.00, 993.00), (711.00, 840.00))
        self.pen.curve_to((653.50, 1188.00), (711.00, 1116.00))
        self.pen.curve_to((496.00, 1260.00), (596.00, 1260.00))
        self.pen.curve_to((371.00, 1219.00), (423.00, 1260.00))
        self.pen.curve_to((289.00, 1002.00), (289.00, 1155.00))
        self.pen.curve_to((341.50, 808.50), (289.00, 873.00))
        self.pen.curve_to((503.00, 744.00), (394.00, 744.00))
        self.pen.curve_to((614.00, 778.00), (562.00, 744.00))
        self.pen.close_path()
        return (1008.00, 1496.00)

    def draw(self, digit):
        match digit:
            case 0:
                return self.d0()
            case 1:
                return self.d1()
            case 2:
                return self.d2()
            case 3:
                return self.d3()
            case 4:
                return self.d4()
            case 5:
                return self.d5()
            case 6:
                return self.d6()
            case 7:
                return self.d7()
            case 8:
                return self.d8()
            case 9:
                return self.d9()
            case 10:
                x1, y1 = self.d1()
                x1 += SPACING
                self.pen.advance(x1)
                x2, y2 = self.d0()
                self.pen.advance(-x1)
                return (x1 + x2, max(y1, y2))
            case 11:
                x1, y1 = self.d1()
                x1 += SPACING
                self.pen.advance(x1)
                x2, y2 = self.d1()
                self.pen.advance(-x1)
                return (x1 + x2, max(y1, y2))
            case 12:
                x1, y1 = self.d1()
                x1 += SPACING
                self.pen.advance(x1)
                x2, y2 = self.d2()
                self.pen.advance(-x1)
                return (x1 + x2, max(y1, y2))
            case 13:
                x1, y1 = self.d1()
                x1 += SPACING
                self.pen.advance(x1)
                x2, y2 = self.d3()
                self.pen.advance(-x1)
                return (x1 + x2, max(y1, y2))
            case 14:
                x1, y1 = self.d1()
                x1 += SPACING
                self.pen.advance(x1)
                x2, y2 = self.d4()
                self.pen.advance(-x1)
                return (x1 + x2, max(y1, y2))
            case 15:
                x1, y1 = self.d1()
                x1 += SPACING
                self.pen.advance(x1)
                x2, y2 = self.d5()
                self.pen.advance(-x1)
                return (x1 + x2, max(y1, y2))
            case 16:
                x1, y1 = self.d1()
                x1 += SPACING
                self.pen.advance(x1)
                x2, y2 = self.d6()
                self.pen.advance(-x1)
                return (x1 + x2, max(y1, y2))
            case 17:
                x1, y1 = self.d1()
                x1 += SPACING
                self.pen.advance(x1)
                x2, y2 = self.d7()
                self.pen.advance(-x1)
                return (x1 + x2, max(y1, y2))
            case 18:
                x1, y1 = self.d1()
                x1 += SPACING
                self.pen.advance(x1)
                x2, y2 = self.d8()
                self.pen.advance(-x1)
                return (x1 + x2, max(y1, y2))
                
            case _:
                raise RuntimeError(f"(Bad digit {digit}")


def measure_font():
    b = BoundingBoxPen()
    f = Font(b)
    for i in range(10):
        print(f"            case {i}:")
        b.reset()
        f.draw(i)
        x_min, x_max, y_min, y_max = b.bounds()
        a = Font(AdjustPen(x_min, y_min))
        a.draw(i)
        print(
            (
                "                     return "
                f"({x_max-x_min:.2f}, {y_max-y_min:.2f})"
            )
        )


def font_stuff():
    font = TTFont(
        "c:/Users/sigma/AppData/Local/Microsoft/Windows\
/Fonts/Helvetica-Bold.ttf"
    )

    rs, ctx = start_draw()
    cairo_pen = RecordingPen(rs, ctx)

    glyphSet = font.getGlyphSet()
    for i, s in enumerate(
        (
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
        )
    ):
        print("            case", i, ":")
        glyph = glyphSet[s]
        bounding_pen = BoundingBoxPen()
        glyph.draw(bounding_pen)
        # print("y bounds", bounding_pen.min_y, bounding_pen.max_y)
        cairo_pen.set_trim_x(bounding_pen.min_x)
        glyph.draw(cairo_pen)
        x_width = bounding_pen.max_x - bounding_pen.min_x
        # cairo_pen.advance_x(x_width + 150)

    finish_draw(rs, ctx, "font")


class SketchPen:
    def __init__(self, ctx):
        self.ctx = ctx
        self.x_offset = 0

    def advance(self, dx):
        self.x_offset += dx

    def adjust(self, p):
        return (p[0] + self.x_offset, p[1])

    def move_to(self, p0):
        self.ctx.move_to(*self.adjust(p0))

    def line_to(self, p1):
        self.ctx.line_to(*self.adjust(p1))

    def curve_to(self, on, off):
        on_adj = self.adjust(on)
        off_adj = self.adjust(off)
        ct1 = two_thirds(self.ctx.get_current_point(), off_adj)
        ct2 = two_thirds(on_adj, off_adj)
        self.ctx.curve_to(*ct1, *ct2, *on_adj)

    def close_path(self):
        self.ctx.close_path()
        self.ctx.fill()


class WidthPen:
    def __init__(self):
        self.reset()

    def reset(self):
        self.max_x = -math.inf
        self.min_x = math.inf
        self.max_y = -math.inf
        self.min_y = math.inf

    def bounds(self):
        print(
            (
                f"min_x={self.min_x:.2f} "
                f"max_x={self.max_x:.2f} "
                f"min_y={self.min_y:.2f} "
                f"max_y={self.max_y:.2f}"
            )
        )
        return (self.min_x, self.max_x)

    def check_x(self, p):
        x, y = p
        if x > self.max_x:
            self.max_x = x
        if x < self.min_x:
            self.min_x = x
        if y > self.max_y:
            self.max_y = y
        if y < self.min_y:
            self.min_y = y

    def move_to(self, p0):
        self.check_x(p0)

    def line_to(self, p1):
        self.check_x(p1)

    def curve_to(self, on, off):
        self.check_x(on)

    def close_path(self):
        pass


def draw_font():
    rs, ctx = start_draw()
    s = SketchPen(ctx)
    f = Font(s)
    for i in range(1, 19):
        x, y = f.draw(i)
        s.advance(x)

    finish_draw(rs, ctx, "sketch")


# These weights define a polyhedron whose faces have equal area
# within a tenth of a percent.
WEIGHTS = [
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


def plot_faces():
    poly = iteration(WEIGHTS)
    p = print_faces(poly)
    p.poles([0, 2, 4, 6, 8])
    p.poles([1, 3, 5, 7, 9])
    for i in (10, 12, 14, 16, 11, 13, 15, 17):
        p.barrel(i)
    p.finish()


def test_rigid():
    x = Rigid((0, 3), (10, 3), (4, 0), (15, 3))
    assert x.move((0, 3)) == (10, 3)
    assert x.move((4, 0)) == (15, 3)
    assert x.move((2, 1.5)) == (12.5, 3)


class Rigid:
    # Define a rigid transformation that maps
    # from0 to to0 and from1 to to1
    # The distance from from0 to from1 must equal
    # the distance from to0 to to1
    def __init__(self, from0, to0, from1, to1):
        df = distance(from0, from1)
        dt = distance(to0, to1)
        if not math.isclose(
            df,
            dt,
            rel_tol=REL_TOL,
            abs_tol=ABS_TOL,
        ):
            print("Fail df", df, "dt", dt)
            assert False

        self.origin = from0
        self.dx = to0[0] - from0[0]
        self.dy = to0[1] - from0[1]

        v0 = self.normvec(from1, from0)
        v1 = self.normvec(to1, to0)

        # Compute the cos and sin of the angle between the two
        # vectors using the dot product and cross product.
        self.cos_t = v0[0] * v1[0] + v0[1] * v1[1]
        self.sin_t = v0[0] * v1[1] - v0[1] * v1[0]

    # The normal vector in the direction from f to to t.
    # f and t must be different points.
    def normvec(pos, t, f):
        dx = t[0] - f[0]
        dy = t[1] - f[1]
        r = math.sqrt(dx * dx + dy * dy)
        return (dx / r, dy / r)

    # Translate p
    def move(self, p):
        v0 = p[0] - self.origin[0]
        v1 = p[1] - self.origin[1]
        r = math.sqrt(v0 * v0 + v1 * v1)

        if r < EPSILON:
            # Alternatively, we could use from1 as the origin.
            return (p[0] + self.dx, p[1] + self.dy)

        cos_u = v0 / r
        sin_u = v1 / r

        cos_x = self.cos_t * cos_u - self.sin_t * sin_u
        sin_x = self.sin_t * cos_u + sin_u * self.cos_t

        return (
            self.origin[0] + r * cos_x + self.dx,
            self.origin[1] + r * sin_x + self.dy,
        )


def print_edges(faces, filename):
    rs, ctx = start_draw()

    edges = []
    for f in faces:
        if f is not None:
            prev = f[-1]
            for v in f:
                edges.append((prev, v))
                prev = v

    while edges:
        e = edges.pop(0)
        ctx.move_to(*e[0])
        v = e[1]
        while True:
            ctx.line_to(*v)
            found = False
            for i, e in enumerate(edges):
                if e[0] == v:
                    v = e[1]
                    found = True
                    break
                if e[1] == v:
                    v = e[0]
                    found = True
                    break
            if found:
                edges.pop(i)
            else:
                ctx.stroke()
                break
    finish_draw(rs, ctx, "pattern")


class CustomPattern:
    def __init__(self):
        self.poly = iteration(WEIGHTS)
        self.flattened = [None] * len(self.poly)
        self.moved = [None] * len(self.poly)

        # Compute all the flattened faces.
        self.poles([0, 2, 4, 6, 8])
        self.poles([1, 3, 5, 7, 9])
        for i in (10, 12, 14, 16, 11, 13, 15, 17):
            self.barrel(i)
        for f in self.flattened:
            assert f is not None

    def poles(self, faces):
        v = self.find_vertex(faces)
        for f in faces:
            self.rotate_pole(f, v)

    def barrel(self, i):
        vertices = self.poly[i][0]
        prev = vertices[-1]
        top = None
        bottom = None
        for v in vertices:
            if prev.z > 0 and v.z < 0:
                assert top is None
                top = copy.deepcopy(v)
                top_length = (v - prev).magnitude()
                top.z = 0.0

            if prev.z < 0 and v.z > 0:
                assert bottom is None
                bottom = copy.deepcopy(v)
                bottom_length = (v - prev).magnitude()
                bottom.z = 0.0
            prev = v
        assert bottom is not None
        assert top is not None

        y_axis = (top - bottom).normalize()
        x_axis = cross_product(y_axis, self.poly[i][2])

        flat = []
        min_y = math.inf
        max_y = -math.inf
        for vx in vertices:
            x = dot_product(x_axis.value(), vx.value())
            y = dot_product(y_axis.value(), vx.value())
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y
            flat.append([x, y])

        self.flattened[i] = flat

    def find_vertex(self, faces):
        result = None
        for f in faces:
            vertices = self.poly[f][0]
            v_set = set(v for v in vertices)
            if result is None:
                result = v_set
            else:
                result = result.intersection(v_set)
        assert len(result) == 1
        return next(iter(result))

    def rotate_pole(self, i, v):
        vertices = self.poly[i][0]
        apex = vertices.index(v)
        prev = vertices[(apex - 1) % len(vertices)]
        next = vertices[(apex + 1) % len(vertices)]
        vx0 = Vertex(*prev.value())
        vx1 = Vertex(*v.value())
        vx2 = Vertex(*next.value())

        y_axis = (
            ((vx1 - vx0).normalize() + (vx1 - vx2).normalize())
            .scale(0.5)
            .normalize()
        )

        x_axis = cross_product(y_axis, self.poly[i][2])
        flat = []
        min_y = math.inf
        max_y = -math.inf
        for vx in vertices:
            x = dot_product(x_axis.value(), vx.value())
            y = dot_product(y_axis.value(), vx.value())
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y
            flat.append([x, y])

        # Translate so the coordinate of the apex is (0, 0)
        align = flat[apex][0]
        for f in flat:
            f[0] -= align
            f[1] -= max_y

        self.flattened[i] = flat

    # src and dst are face numbers that have an edge in common.
    # moved[src] has been already been computed.
    # Compute moved[dst]
    def attach(self, src, dst):
        if self.moved[src] is not None:
            return
        print(f"Attach src={src} to dst={dst}")
        # Find the common edge.
        src_edge = self.poly[src][1].index(dst)
        src_vtx = self.flattened[src]
        src_p0 = src_vtx[src_edge]
        src_p1 = src_vtx[(src_edge + 1) % len(src_vtx)]

        dst_edge = self.poly[dst][1].index(src)

        dst_vtx = self.moved[dst]

        dst_p0 = dst_vtx[(dst_edge + 1) % len(dst_vtx)]
        dst_p1 = dst_vtx[dst_edge]

        translate = Rigid(src_p0, dst_p0, src_p1, dst_p1)
        moved = []
        for p in src_vtx:
            if p == src_p0:
                m = dst_p0
            elif p == src_p1:
                m = dst_p1
            else:
                m = translate.move(p)
            moved.append(m)
        self.moved[src] = moved

    def main(self):
        for i in range(len(self.poly)):
            print(f"{i}: {', '.join(str(f) for f in self.poly[i][1])}")
        for i in range(len(self.poly)):
            v = self.poly[i][0]
            z = self.flattened[i]
            assert len(v) == len(z)
            for j in range(len(v)):
                prev = (j - 1) % len(v)
                dv = distance(v[prev].value(), v[j].value())
                dz = distance(z[prev], z[j])
                if not math.isclose(dv, dz):
                    print("Distance error", dv, dz)
                    assert False

        x = [1, 3, 5, 7, 9]
        self.moved[x[0]] = copy.deepcopy(self.flattened[x[0]])
        for i in range(1, len(x)):
            self.attach(x[i], x[i - 1])
        self.attach(9, 1)
        self.attach(11, 1)
        self.attach(13, 1)
        self.attach(15, 3)
        self.attach(17, 5)
        self.attach(4, 11)
        self.attach(6, 4)
        self.attach(8, 6)
        self.attach(0, 8)
        self.attach(2, 0)
        self.attach(10, 0)
        self.attach(12, 0)
        self.attach(14, 2)
        self.attach(16, 2)

        if False:
            self.attach(10, 0)
            self.attach(12, 0)
            self.attach(14, 2)
            self.attach(15, 3)
            self.attach(16, 2)
        print_edges(self.moved, "custom")


def custom():
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

    poly = iteration(weights)
    areas = [area(p[0]) for p in poly]
    best_score = min(areas) / max(areas)
    print(f"Score {best_score}")
    mod = 0.001
    for _ in range(0):
        best_adj = None

        for i in range(0, len(weights)):
            for sign in range(-1, 2, 2):
                adj = weights.copy()
                adj[i] += math.copysign(mod, sign)
                poly = iteration(adj)
                areas = [area(p[0]) for p in poly]
                adj_score = min(areas) / max(areas)
                if adj_score > best_score:
                    best_score = adj_score
                    best_adj = adj

        if best_adj is None:
            mod = mod * 0.5
            print(f"Mod is now {mod}")
        else:
            print(f"Score {best_score}")
            weights = best_adj

    print("Best weights")
    for w in weights:
        print(format(w, ".20f"))

    print("Result")
    for p in poly:
        print(p)
    better_pattern(poly).make_pattern()


def debug_model():
    halves = read_halves("c:/users/sigma/documents/bad_model.bin")
    better_pattern(make_polygon(halves)).make_pattern()


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
    better_pattern(make_polygon(halves)).make_pattern()


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
    # plot_faces()
    # font_stuff()
    # CustomPattern().main()
    draw_font()
    # measure_font()
