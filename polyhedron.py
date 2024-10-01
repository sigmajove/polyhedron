import cairo
import math

# A number expected to be a floating point rounding error away from zero.
EPSILON = 1e-10


# Determines whether a point is within a half-space
def within(point, half_space):
    x, y, z = point
    a, b, c, d = half_space
    return a * x + b * y + c * z + d <= 0.0


# dot product of two vectors
def dot_product(v0, v1):
    length = len(v0)
    assert length == len(v1)
    return sum(v0[i] * v1[i] for i in range(length))


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


# A vertex is defined by (x, y, z) coordinates in three-space.
class Vertex:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

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

    # Move the tuple (x, y, z) along the vector.
    def move(self, point):
        x, y, z = point
        return (x + self.dx, y + self.dy, z + self.dz)

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
# returns the point where the line crosses the plane.
# Returns None if the line is (nearly) parallel to the plane.
def plane_line(plane, point, vector):
    if not isinstance(plane, list):
        raise RuntimeError(f"Bad plane {plane}")
    assert len(plane) == 4
    assert isinstance(point, tuple)
    assert len(point) == 3
    assert isinstance(vector, tuple)
    assert len(vector) == 3
    a, b, c, d = plane
    px, py, pz = point
    vx, vy, vz = vector

    denom = a * vx + b * vy + c * vz
    if abs(denom) < EPSILON:
        # The line is (nearly) parallel to the plane.
        # There is no interesection.
        return None
    t = -(a * px + b * py + c * pz + d) / denom

    x = px + t * vx
    y = py + t * vy
    z = pz + t * vz
    return (x, y, z)


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

            # The new point
            new_point = Vertex(*plane_line(half, edge.e0.value(), vector))

            # Replace the endpoint not in the half-space.
            if code == 1:
                edge.e1 = new_point
                edge.action = NEW_E1
            else:
                edge.e0 = new_point
                edge.action = NEW_E0
        case 3:
            # Both endpoints are in the half-space.
            edge.action = KEEP


def trim_half_line(edge, half):
    e0_is_vector = isinstance(edge.e0, Vector)
    if e0_is_vector:
        assert isinstance(edge.e1, Vertex)
        vector = edge.e0.value()
        vertex = edge.e1
    else:
        assert isinstance(edge.e0, Vertex)
        assert isinstance(edge.e1, Vector)
        vertex = edge.e0
        vector = edge.e1.value()
    point = vertex.value()

    new_point = plane_line(half, point, vector)
    if new_point is None:
        # The line is parallel to the plane.
        # Either we keep the entire edge or delete the entire edge.
        edge.action = KEEP if within(point, half) else DELETE
        return
    new_vertex = Vertex(*new_point)

    if (
        dot_product(delta_vector(vertex, new_vertex).value(), vector) > 0
    ) ^ e0_is_vector:
        # The point is on the half line
        if e0_is_vector:
            edge.e0 = new_vertex
            edge.action = NEW_E0
        else:
            edge.e1 = new_vertex
            edge.action = NEW_E1
    else:
        # The half-space does not intersect the half-line
        edge.action = KEEP


def trim(edge, half):
    if isinstance(edge.e0, Vertex) and isinstance(edge.e1, Vertex):
        trim_vertex_vertex(edge, half)
    else:
        trim_half_line(edge, half)


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


def starter(plane0, idplane1, idplane2):
    id1, plane1 = idplane1
    id2, plane2 = idplane2

    p10, vec10 = two_planes(plane0, plane1)
    _, vec20 = two_planes(plane2, plane0)
    apex_point = plane_line(plane2, p10, vec10.value())
    apex_vertex = Vertex(*apex_point)

    if not within(vec10.move(apex_point), plane2):
        vec10.reverse()
    if not within(vec20.move(apex_point), plane1):
        vec20.reverse()

    if dot_product(cross_product(vec10, vec20).value(), plane0[:3]) > 0:
        vec10.reverse()
        return [Edge(vec10, apex_vertex, id1), Edge(apex_vertex, vec20, id2)]
    else:
        vec20.reverse()
        return [Edge(vec20, apex_vertex, id2), Edge(apex_vertex, vec10, id1)]


# Find the face in in_half formed by the interesection with all the
# other half-planes.
def get_face(in_half, rest):
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
    for r in rest[2:]:
        edges = add_half(edges, r, in_half)

    return edges


# Slices a face, keeping just the part in the half-plane
def add_half(edges, idhalf, face_plane):
    id, half = idhalf
    # Iterate over all the edges, and decide what to do
    # with each.
    for e in edges:
        trim(e, half)

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
            edges.append(Edge(e0, e1, id))

        case _:
            raise RuntimeError(f"{len(pairs)} new vertices")

    for e in edges:
        e.action = KEEP
    return edges


def find_faces(half_planes):
    # Assign an id to each half-plane
    half_planes = list(enumerate(half_planes))

    faces = []
    for i in range(len(half_planes)):
        face = get_face(half_planes[i][1], half_planes[:i] + half_planes[i:])
        edict = {}
        n = len(face)
        for edge in face:
            assert isinstance(edge.e0, Vertex)
            assert isinstance(edge.e1, Vertex)
            edict[edge.e0] = edge
        e = face[0]
        start = e
        sorted = [e]
        while True:
            e = edict[e.e1]
            if e is start:
                break
            sorted.append(e)
            assert len(sorted) <= n
        assert len(sorted) == n

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
    c = cross_product(delta_vector(p1, p0), delta_vector(p1, p2)).value()
    return [*c, -dot_product(p0.value(), c)]


def octa():
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
    halves = [f0, f1, f2, f3, f4, f5, f6, f7]
    faces = find_faces(halves)
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
