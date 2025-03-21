from font import Font
from poly18 import Vertex
from poly18 import Vector
from poly18 import cross_product
from poly18 import dot_product
from poly18 import poly18
from scipy.interpolate import BSpline
from stl import mesh
import copy
import math
import random
import svg_writer
import triangle
import numpy as np
import rotate_edge

# Maps the internal numbering used by poly18 for the faces to the
# numbers we want inscribed on each face.
LABELS = (
    +3,  # 0
    16,  # 1
    12,  # 2
    +7,  # 3
    18,  # 4
    +1,  # 5
    +5,  # 6
    14,  # 7
    10,  # 8
    +9,  # 9
    +2,  # 10
    17,  # 11
    13,  # 12
    +6,  # 13
    +4,  # 14
    15,  # 15
    +8,  # 16
    11,  # 17
)


# The Euclidean distance between two points
def distance(p, q):
    assert len(p) == len(q)
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p, q)))


def midpoint(p, q):
    return (0.5 * (p[0] + q[0]), 0.5 * (p[1] + q[1]))


# Given three(x, y) coordinates which are not colinear,
# returns the center of the circle that passes through all three.
def circumcenter(p1, p2, p3):
    # Unpack the coordinates
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    # Calculate the perpendicular bisector parameters
    a = 2 * (x2 - x1)
    b = 2 * (y2 - y1)
    c = x2**2 + y2**2 - x1**2 - y1**2
    d = 2 * (x3 - x1)
    e = 2 * (y3 - y1)
    f = x3**2 + y3**2 - x1**2 - y1**2

    # Calculate the circumcenter coordinates
    x = (c * e - f * b) / (a * e - b * d)
    y = (c * d - a * f) / (b * d - a * e)

    return (x, y)


def circle_through_points(x1, y1, x2, y2, x3, y3):
    s1 = x1 * x1 + y1 * y1
    s2 = x2 * x2 + y2 * y2
    s3 = x3 * x3 + y3 * y3

    y12 = y1 - y2
    y23 = y2 - y3
    y31 = y3 - y1
    det = 2 * (x1 * y23 + x2 * y31 + x3 * y12)

    # Calculate center coordinates
    x0 = (s1 * y23 + s2 * y31 + s3 * y12) / det
    y0 = (s1 * (x3 - x2) + s2 * (x1 - x3) + s3 * (x2 - x1)) / det

    # Calculate radius squared
    r_squared = (x1 - x0) ** 2 + (y1 - y0) ** 2

    return (x0, y0, r_squared)


def in_circle(a, b, c, p):
    x0, y0, r_squared = circle_through_points(*a, *b, *c)
    dist = (p[0] - x0) ** 2 + (p[1] - y0) ** 2
    return dist <= r_squared


def is_convex_quad(a, b, c, d):
    prev2 = c
    prev = d
    sign = None
    for p in [a, b, c, d]:
        dx0 = prev[0] - prev2[0]
        dy0 = prev[1] - prev2[1]
        dx1 = p[0] - prev[0]
        dy1 = p[1] - prev[1]

        cross = dx0 * dy1 - dy0 * dx1
        if abs(cross) < 1e-4:
            return False
        s = -1 if cross < 0 else 1
        if sign is None:
            sign = s
        else:
            if sign != s:
                return False
        prev2 = prev
        prev = p

    return True


def check_triangulation(vertices, triangles, neighbors, tag):
    with svg_writer.SVGWriter(tag, 25, 0.005) as ctx:
        for tr0, n in enumerate(neighbors):
            for v, tr1 in enumerate(n):
                xxx = triangles[tr0]
                apex0 = xxx[v]  # point index
                o0, o1 = (1, 2) if v == 0 else (0, 2) if v == 1 else (0, 1)
                other0 = xxx[o0]  # point index
                other1 = xxx[o1]  # point index
                if tr1 >= 0:
                    yyy = triangles[tr1]
                    apex1 = None
                    for p in yyy:
                        if p != other0 and p != other1:
                            if apex1 is not None:
                                raise RuntimeError("too many apex1")
                            apex1 = p
                    if apex1 is None:
                        raise RuntimeError("cant find apex1")
                    # I now have point indices
                    # apex0, apex1, other0, other1
                    # translate them to coordinates
                    e0 = vertices[other0]
                    e1 = vertices[other1]
                    t0 = vertices[apex0]
                    t1 = vertices[apex1]
                    if is_convex_quad(e0, t0, e1, t1) and in_circle(
                        e0, t0, e1, t1
                    ):
                        print(f"Found non-Delauney edge in {tag}")
                        print(f"bad edge {e0}, {e1}")
                        ctx.set_source_rgb(1, 0, 0)
                        ctx.set_line_width(0.01)
                    else:
                        ctx.set_source_rgb(0, 0, 0)
                        ctx.set_line_width(0.005)
                    ctx.move_to(*e0)
                    ctx.line_to(*e1)
                    ctx.stroke()


TRIANGLE_COUNTER = 0


class Triangle:
    def __init__(self, p, q, r):
        global TRIANGLE_COUNTER
        self.p0 = p
        self.p1 = q
        self.p2 = r
        self.id = TRIANGLE_COUNTER
        TRIANGLE_COUNTER += 1

    def area(self):
        v1 = self.p1 - self.p0
        v2 = self.p2 - self.p0
        c = cross_product(v1, v2)
        return c.magnitude()


class DummyPen:
    def __init__(self):
        pass

    def move_to(self, p):
        pass

    def line_to(self, p):
        pass

    def curve_to(self, p, q):
        pass

    def cubic(self, p, q, r):
        pass

    def add_steiner(self, p):
        pass

    def advance(self, x):
        pass

    def close_path(self, hole=False, inner=None):
        pass


pass
STEP = 0.025

# The depth of the indentation numbers
DEPTH = 0.1


# Returns a sorted tuple
def normalize(a, b):
    return (a, b) if a <= b else (b, a)


class DigitPen:
    def __init__(self, faceno, border, scale, x_offset, y_offset, zilla):
        self.faceno = faceno
        self.zilla = zilla
        self.upper = []
        self.lower = []

        self.manual_steiner = []

        # Record the constructor parameters.
        # We use the border to determine whether random Steiner points
        # are valid.
        assert len(border) > 2
        self.border = [tuple(p) for p in border]

        # Scale and offset are used to map numbers in the font
        # to coordinates that match the border.
        self.scale = scale
        self.x_offset = x_offset
        self.y_offset = y_offset

        # The points that get passed to Triangulate
        self.points = []

        # A bunch of segments of the form((x1, y1), (x2, y2))
        # that are the constrained edges used by CDT
        self.segments = []

        # The sequence of points being accumulated between
        # move_to and close_path
        self.current = [self.border[0]]

        # Find the location of a point just inside the border.
        b0 = self.border[0]
        b1 = self.border[1]
        dist = distance(b0, b1)
        mid = midpoint(b0, b1)
        c = (b1[0] - b0[0]) / dist
        s = (b1[1] - b0[1]) / dist
        small = 0.01
        sx = mid[0] - s * small
        sy = mid[1] + c * small
        self.upper.append((sx, sy))

        for p in border[1:]:
            self.interpolate(p)
        self.interpolate(self.border[0])

        bad = 0
        for p in self.current:
            back = Vertex(*self.zilla.rotate_back((p[0], p[1], 0), faceno))
            if self.zilla.find_join_point(back) is None:
                bad += 1
        if bad:
            print(f"border has {bad} of {len(self.current)} bad points")

        # note self.current begins and ends with border[0]
        # Don't use ClosePath to mark the inner point because
        # it will adjust it.
        self.close_path(hole=False, inner=None)

    def move_to(self, p):
        self.current = [self.adjust(p)]

    def line_to(self, p):
        self.interpolate(self.adjust(p))

    # Adds evenly spaced points on a line from current[-1] to r1
    # Uses raw(border - relative) coordinates.
    def interpolate(self, r1):
        r0 = self.current[-1]
        n = math.ceil(distance(r0, r1) / (10 * STEP))
        dx = r1[0] - r0[0]
        dy = r1[1] - r0[1]
        for i in range(1, n):
            mx = r0[0] + (i * dx) / n
            my = r0[1] + (i * dy) / n
            self.current.append((mx, my))
        self.current.append(r1)

    # Quadratic B-spline
    def curve_to(self, on, off):
        self.b_spline(self.current[-1], self.adjust(off), self.adjust(on))

    # Cubic B-spline
    def cubic(self, a, b, c):
        self.b_spline(
            self.current[-1], self.adjust(a), self.adjust(b), self.adjust(c)
        )

    def b_spline(self, *args):
        k = len(args) - 1
        spline = BSpline(
            np.concatenate(
                [
                    np.zeros(k),
                    np.linspace(0, 1, 2),
                    np.ones(k),
                ]
            ),
            np.array(args),
            k,
        )

        # Approximate the length of the curve using eight segments.
        length = 0
        prev = self.current[-1]
        for x, y in spline(np.linspace(0, 1, 8))[1:]:
            p = (float(x), float(y))
            length += distance(p, prev)
            prev = p

        # Choose a number of segments so that each segment will
        # have length approximately STEP
        num_segments = math.ceil(length / STEP)

        counter = 0
        for x, y in spline(np.linspace(0, 1, num_segments))[1:]:
            p = (float(x), float(y))
            prev = self.current[-1]
            self.current.append(p)

            # On every third segment, add two Steiner points on either side of
            # the segment.
            # The pattern is O O X O O
            if counter == 2:
                # Add Steiner points
                dist = distance(p, prev)
                mid = midpoint(p, prev)

                # The Steiner points at the ends of vectors of length
                # 6 * STEP, extending from the midpoint of the segment
                # perpendicular to the segment.
                cos = (p[0] - prev[0]) / dist
                sin = (p[1] - prev[1]) / dist
                parm = 2
                self.add_raw_steiner(
                    (mid[0] - sin * parm * STEP, mid[1] + cos * parm * STEP)
                )
                self.add_raw_steiner(
                    (mid[0] + sin * parm * STEP, mid[1] - cos * parm * STEP)
                )
            counter += 1
            if counter == 4:
                counter = 0

    def close_path(self, hole=False, inner=None):
        if inner is not None:
            (self.lower if hole else self.upper).append(self.adjust(inner))

        if len(self.current) >= 2:
            assert self.current[0] == self.current[-1]
            base = len(self.points)
            self.points.extend(self.current[:-1])
            before = len(self.segments)
            for i in range(0, len(self.current) - 2):
                self.segments.append((base + i, base + i + 1))
            self.segments.append((base + len(self.current) - 2, base))
            after = len(self.segments)
        self.current = []

    def dump_data(self, i):
        filename = f"data{i:02d}"
        segpoints = set()
        for s in self.segments:
            segpoints.add(s[0])
            segpoints.add(s[1])
        with svg_writer.SVGWriter(filename, 25, 0.005) as ctx:
            for i, p in enumerate(self.points):
                if i in segpoints:
                    ctx.set_source_rgb(0, 0, 0)
                else:
                    ctx.set_source_rgb(1, 0, 0)
                ctx.arc(*p, 0.01, 0, 2 * math.pi)
                ctx.fill()
            ctx.set_source_rgb(0, 0, 0)
            for s in self.segments:
                ctx.move_to(*self.points[s[0]])
                ctx.line_to(*self.points[s[1]])
                ctx.stroke()

            for h in self.lower:
                ctx.set_source_rgb(0, 1, 0)
                ctx.arc(*h, 0.01, 0, 2 * math.pi)
                ctx.fill()

            for h in self.upper:
                ctx.set_source_rgb(0, 0, 1)
                ctx.arc(*h, 0.01, 0, 2 * math.pi)
                ctx.fill()

        print(f"Wrote {filename}")

    def triangulate(self, lower, tag=None):
        result = triangle.triangulate(
            {
                "vertices": self.points,
                "segments": self.segments,
                "holes": self.lower if lower else self.upper,
            },
            opts="pn",
        )

        # The results returned by triangle.triangulate use weird numpy
        # types for the coordinates. To avoid surprises, convert
        # everything to native Pyghon types.
        points = [(float(p[0]), float(p[1])) for p in result["vertices"]]
        triangles = [
            [int(t[0]), int(t[1]), int(t[2])] for t in result["triangles"]
        ]
        neighbors = [
            [int(n[0]), int(n[1]), int(n[2])] for n in result["neighbors"]
        ]

        # Delaunay triangulation works pretty well, but sometimes it
        # can produce triangles that Blender's 3D print toolkit flags as
        # being too skinny. Locate and repair skinny triangle, where possible.
        for i in range(len(triangles)):
            rotate_edge.repair_if_skinny(points, triangles, neighbors, i)

        # Make sure rotate_edge didn't mess up the neighbors.
        rotate_edge.check_neighbors(triangles, neighbors)

        # Find all the boundary edges.
        # These are found in triangles with missing neighbors.
        boundaries = []
        for tid, n in enumerate(neighbors):
            for i, neigh in enumerate(n):
                if neigh < 0:
                    tri = triangles[tid]
                    # List the vertices of the edge in counterclockwise order.
                    match i:
                        case 0:
                            edge = (tri[1], tri[2])
                        case 1:
                            edge = (tri[2], tri[0])
                        case 2:
                            edge = (tri[0], tri[1])
                        case _:
                            raise RuntimeError(
                                "Triangle has more than three neighbors"
                            )
                    if lower:
                        # Use clockwise order
                        edge = (edge[1], edge[0])
                    boundaries.append(edge)
        return (points, triangles, boundaries)

    def advance(self, dx):
        self.x_offset += dx

    # Maps the coordinate system used internally by Font to the
    # one we are using to construct the mesh.
    def adjust(self, p):
        return (
            (p[0] + self.x_offset) * self.scale,
            (p[1] + self.y_offset) * self.scale,
        )

    def dump(self, i):
        filename = f"tile{i:02d}"
        with svg_writer.SVGWriter(filename, 25, 1) as ctx:
            ctx.set_line_width(0.001)
            points, triangles, _ = self.triangulate(lower=False)

            ctx.set_source_rgb(0, 0, 0)
            for t in triangles:
                ctx.move_to(*points[t[0]])
                ctx.line_to(*points[t[1]])
                ctx.line_to(*points[t[2]])
                ctx.close_path()
                ctx.stroke()

            points, triangles, _ = self.triangulate(lower=True)

            ctx.set_source_rgb(0, 0, 0)
            for t in triangles:
                ctx.move_to(*points[t[0]])
                ctx.line_to(*points[t[1]])
                ctx.line_to(*points[t[2]])
                ctx.close_path()
                ctx.set_source_rgba(1, 0, 0, 0.6)
                ctx.fill()
                ctx.set_source_rgb(0, 0, 0)

                ctx.move_to(*points[t[0]])
                ctx.line_to(*points[t[1]])
                ctx.line_to(*points[t[2]])
                ctx.close_path()
                ctx.stroke()

            ctx.set_source_rgb(1, 0, 0)
            for s in self.manual_steiner:
                ctx.arc(*s, 0.005, 0, 2 * math.pi)
                ctx.fill()

            print(f"Wrote out {filename}")

    def make_mesh(self, i):
        upper_points, upper_triangles, upper_boundary = self.triangulate(
            lower=False, tag=f"upper{i:02d}"
        )
        lower_points, lower_triangles, lower_boundary = self.triangulate(
            lower=True, tag=f"lower{i:02d}"
        )

        num_triangles = len(upper_triangles) + len(lower_triangles)

        # The combined points will be upper + lower
        def to_lower(p):
            return p + len(upper_points)

        combined_points = []
        for p in upper_points:
            combined_points.append((p[0], p[1], -DEPTH))
        for p in lower_points:
            combined_points.append((p[0], p[1], 0))

        mesh_triangles = [t for t in upper_triangles]
        for t in lower_triangles:
            mesh_triangles.append(tuple([to_lower(p) for p in t]))

        for e in set(lower_boundary) & set(upper_boundary):
            #  u0 -- u1
            #  |    / |
            #  |   /  |
            #  |  /   |
            #  | /    |
            #  l0 --- l1
            u0 = e[0]
            u1 = e[1]
            l0 = to_lower(e[0])
            l1 = to_lower(e[1])
            mesh_triangles.append((u0, l0, u1))
            mesh_triangles.append((l0, l1, u1))
            num_triangles += 2

        # Translate the 2d triangulation back to its
        # original 3d coordinates.
        for j, p in enumerate(combined_points):
            combined_points[j] = self.zilla.correct(
                self.zilla.rotate_back(p, i)
            )

        for p in combined_points:
            if not isinstance(p, Vertex):
                print(f"Bad point {type(p)}")

        for t in mesh_triangles:
            self.zilla.big_mesh.append(
                Triangle(
                    combined_points[t[0]],
                    combined_points[t[1]],
                    combined_points[t[2]],
                )
            )

    # Test that p is fully within border.
    # It must be at least 0.05 away from any edge.
    # Assumes that border is convex.
    def is_legal_steiner(self, p):
        def position(p1, p2, q):
            num = (
                (p2[0] - p1[0]) * q[1]
                - (p2[1] - p1[1]) * q[0]
                - p2[0] * p1[1]
                + p2[1] * p1[0]
            )
            d_sq = (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2
            return num * num >= d_sq * (0.05) ** 2

        prev = self.border[-1]
        for z in self.border:
            if not (position(prev, z, p)):
                return False
            prev = z
        return True

    def add_raw_steiner(self, p):
        if self.is_legal_steiner(p) and self.not_too_close(p):
            self.points.append(p)

    def force_raw_steiner(self, p):
        if self.is_legal_steiner(p):
            self.points.append(p)

    def add_steiner(self, p):
        a = self.adjust(p)
        self.manual_steiner.append(a)
        self.force_raw_steiner(a)

    def not_too_close(self, p):
        return (
            min((distance(q, p) for q in self.points), default=math.inf) > 0.2
        )

    def add_steiner_points(self, trace=False):
        if trace:
            print("Tracing steiner")
            min_dist = math.inf
        min_x = min(p[0] for p in self.border)
        max_x = max(p[0] for p in self.border)
        min_y = min(p[1] for p in self.border)
        max_y = max(p[1] for p in self.border)
        x_range = max_x - min_x
        y_range = max_y - min_y
        gran = 5
        x_buck = lambda x: min(
            gran - 1, math.floor(gran * (x - min_x) / x_range)
        )
        y_buck = lambda y: min(
            gran - 1, math.floor(gran * (y - min_y) / y_range)
        )

        buckets = [[[] for j in range(gran)] for i in range(gran)]

        def add_to_bucket(s):
            buckets[y_buck(s[1])][x_buck(s[0])].append(s)

        for p in self.points:
            add_to_bucket(p)

        # generates the distances to all the points in adjacent buckets
        def dist_near(p):
            xb = x_buck(p[0])
            yb = y_buck(p[1])
            x_lo = max(0, xb - 1)
            x_hi = min(gran - 1, xb + 1)
            y_lo = max(0, yb - 1)
            y_hi = min(gran - 1, yb + 1)
            for x in range(x_lo, x_hi + 1):
                for y in range(y_lo, y_hi + 1):
                    for s in buckets[y][x]:
                        yield distance(s, p)

        random.seed(12345)
        rand_point = lambda: (
            random.uniform(min_x, max_x),
            random.uniform(min_y, max_y),
        )
        num_points = 0
        miss = 0
        thresh = 100
        while True:
            p = rand_point()
            if not self.is_legal_steiner(p):
                continue

            if self.not_too_close(p):
                if trace:
                    dist = min(
                        (distance(q, p) for q in self.points), default=math.inf
                    )
                    if dist < min_dist:
                        min_dist = dist
                self.points.append(p)
                add_to_bucket(p)
                num_points += 1
                if num_points > thresh:
                    thresh += 100
                miss = 0
            else:
                miss += 1
                if miss > 100:
                    break

        if trace:
            print(f"Minimum Steiner dist {min_dist}")


def main():
    c = Codezilla()
    for i in range(18):
        c.print_digit(i)
    c.check_mesh()
    c.make_model()


def internal_points(r0, r1):
    n = math.ceil(distance(r0, r1) / (10 * STEP))
    dx = r1[0] - r0[0]
    dy = r1[1] - r0[1]
    dz = r1[2] - r0[2]
    for i in range(1, n):
        mx = r0[0] + (i * dx) / n
        my = r0[1] + (i * dy) / n
        mz = r0[2] + (i * dz) / n
        yield (mx, my, mz)


def internal_points_2d(r0, r1):
    n = math.ceil(distance(r0, r1) / (10 * STEP))
    dx = r1[0] - r0[0]
    dy = r1[1] - r0[1]
    for i in range(1, n):
        mx = r0[0] + (i * dx) / n
        my = r0[1] + (i * dy) / n
        yield (mx, my)


class Codezilla:
    def __init__(self):
        self.poly = poly18()
        self.flattened = [None] * len(self.poly)
        self.flattened_center = [None] * len(self.poly)
        self.flattened_top = [None] * len(self.poly)

        # x, y, z normal vectors that will allow transforming
        # the flattened mesh back to its original place on the polyhedron.
        self.axes = [None] * len(self.poly)

        self.translate = [None] * len(self.poly)

        self.store_join_points()

        # This is the model we are producing.
        # A list of 3-tuples of 2-tuples of (x, y, z) coordinates.
        self.big_mesh = []

        self.poles([0, 2, 4, 6, 8])
        self.poles([1, 3, 5, 7, 9])
        for i in (10, 12, 14, 16, 11, 13, 15, 17):
            self.barrel(i)

    # Join points are on the intersection of two faces.
    # When we map 3d points to 2d points and back to 3d,
    # we don't necessarily get back the same points we put in.
    # If the point we get back is close to join point, we move it
    # back to the join point.
    def store_join_points(self):
        edges = dict()
        for x, face in enumerate(self.poly):
            prev = face[0][-1]
            for p in face[0]:
                key = normalize(prev.value(), p.value())
                edges[key] = edges.get(key, 0) + 1
                prev = p
        for v in edges.values():
            assert v == 2

        vertices = set()
        for k in edges.keys():
            vertices.add(k[0])
            vertices.add(k[1])

        self.join_points = [Vertex(*p) for p in vertices]
        for k in edges.keys():
            for p in internal_points(k[0], k[1]):
                self.join_points.append(Vertex(*p))

    def find_join_point(self, v):
        result = None
        for p in self.join_points:
            if p.isclose(v):
                if result is not None:
                    print("ambiguous join")
                    print("approx", v)
                    print(result)
                    print(p)
                    raise RuntimeError("bad")
                result = p
        return result

    def is_join_point(self, v):
        assert isinstance(v, Vertex)
        return v in self.join_points

    def correct(self, v):
        v = Vertex(*v)
        c = self.find_join_point(v)
        return v if c is None else c

    #   def mesh_perimeter(self, mesh, points):
    #       print("Mesh Perimeter")
    #       edges = dict()

    #       def add_edge(p, q, t):
    #           key = normalize(p, q)
    #           s = edges.get(key, None)
    #           if s is None:
    #               s = []
    #               edges[key] = s
    #           s.append(t)

    #       for i, t in enumerate(mesh):
    #           a, b, c = t
    #           add_edge(a, b, i)
    #           add_edge(b, c, i)
    #           add_edge(c, a, i)

    #       path = dict()

    #       def add_path(i, j):
    #           s = path.get(i, None)
    #           if s is None:
    #               s = []
    #               path[i] = s
    #           s.append(j)

    #       def erase_path(i, j):
    #           s = path[i]
    #           s.remove(j)
    #           if len(s) == 0:
    #               del path[i]

    #       start = None
    #       for k, v in edges.items():
    #           if len(v) == 1:
    #               # border edge
    #               add_path(k[0], k[1])
    #               add_path(k[1], k[0])
    #           elif len(v) == 2:
    #               pass
    #           else:
    #               raise RuntimeError(f"{len(v)} triangles with edge")

    #       for p in path.values():
    #           assert len(p) == 2

    #       start = next(iter(path.keys()))
    #       j = start
    #       k = path[j][0]
    #       trail = [j, k]
    #       while True:
    #           n = path[k]
    #           n = n[0] if n[1] == j else n[1]
    #           if n == start:
    #               break
    #           trail.append(n)
    #           j, k = k, n
    #       assert len(path) == len(trail)
    #       if True:
    #           for t in trail:
    #               p = points[t]
    #               # flag = "" if self.is_join_point(Vertex() else " !!!"
    #               # print(f"{p.x:.14f}, {p.y:.14f}, {p.z:.14f}")
    #               print(p)

    #       with svg_writer.SVGWriter("perimeter", 50, 0.01) as ctx:
    #           points = [points[t] for t in trail]
    #           ctx.move_to(p[0], p[1])
    #           for p in points[1:]:
    #               ctx.line_to(p[0], p[1])
    #           ctx.close_path()
    #           ctx.stroke()
    #           for p in points:
    #               ctx.arc(p[0], p[1], 0.02, 0, 2 * math.pi)
    #               ctx.fill()

    def make_model(self):
        model = mesh.Mesh(np.zeros(len(self.big_mesh), dtype=mesh.Mesh.dtype))
        for i, t in enumerate(self.big_mesh):
            model.vectors[i][0] = t.p0.value()
            model.vectors[i][1] = t.p1.value()
            model.vectors[i][2] = t.p2.value()
        model.check(exact=True)
        model.save("c:/users/sigma/documents/model18.stl")
        print("Wrote out model")

    def check_mesh(self):
        counter = 0
        edges = dict()
        for i, t in enumerate(self.big_mesh):

            def add_triangle(key):
                s = edges.get(key, None)
                if s is None:
                    s = []
                    edges[key] = s
                s.append(t.id)

            add_triangle(normalize(t.p0, t.p1))
            add_triangle(normalize(t.p1, t.p2))
            add_triangle(normalize(t.p2, t.p0))

            counter += 1
        print(f"{counter} triangles")
        print(f"{len(edges)} edges")
        bad_triangles = 0
        for k, v in edges.items():
            if len(v) != 2:
                bad_triangles += 1
        print(bad_triangles, "Bad triangles")

        min_area = min(t.area() for t in self.big_mesh)
        print(f"smallest triangle is {min_area}")

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
        self.axes[i] = (x_axis, y_axis, self.poly[i][2])
        self.translate[i] = (0, 0)

        flat = []
        for vx in vertices:
            x = dot_product(x_axis.value(), vx.value())
            y = dot_product(y_axis.value(), vx.value())
            flat.append([x, y])

        self.flattened[i] = flat

        for j, v in enumerate(vertices):
            back = Vertex(*self.rotate_back((flat[j][0], flat[j][1], 0), i))
            if not back.isclose(v):
                self.check_axes(i)
                print("orig =", v)
                print("back =", back)
                raise RuntimeError("round trip failure")

        center = self.poly[i][3]
        x = dot_product(x_axis.value(), center)
        y = dot_product(y_axis.value(), center)
        self.flattened_center[i] = (x, y)
        x = dot_product(x_axis.value(), top.value())
        y = dot_product(y_axis.value(), top.value())
        self.flattened_top[i] = (x, y)

    def check_axes(self, i):
        axes = self.axes[i]

        # Check that the axes are all unit vectors.
        for a in axes:
            assert math.isclose(a.dx * a.dx + a.dy * a.dy + a.dz * a.dz, 1.0)

        # Check that they follow the right-hand rule
        assert axes[0].isclose(cross_product(axes[1], axes[2]))
        assert axes[1].isclose(cross_product(axes[2], axes[0]))
        assert axes[2].isclose(cross_product(axes[0], axes[1]))

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

        self.axes[i] = (x_axis, y_axis, self.poly[i][2])
        flat = []
        max_y = -math.inf
        center = Vertex(*self.poly[i][3])
        for v in vertices:
            x = dot_product(x_axis.value(), v.value())
            y = dot_product(y_axis.value(), v.value())
            if y > max_y:
                max_y = y
            flat.append([x, y])

        # Translate so the coordinate of the apex is(0, 0)
        align = flat[apex][0]
        self.translate[i] = (-align, -max_y)

        for f in flat:
            f[0] -= align
            f[1] -= max_y

        self.flattened[i] = flat

        for j, v in enumerate(vertices):
            back = Vertex(*self.rotate_back((flat[j][0], flat[j][1], 0), i))
            if not back.isclose(v):
                self.check_axes(i)
                print("orig =", v)
                print("back =", back)
                raise RuntimeError("round trip failure")

        center = self.poly[i][3]
        x = dot_product(x_axis.value(), center) - align
        y = dot_product(y_axis.value(), center) - max_y
        self.flattened_center[i] = (x, y)
        self.flattened_top[i] = (0, 0)

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

    # Move the flattened point back to its original 3d position
    def rotate_back(self, p, i, trace=False):
        x_flat = p[0] - self.translate[i][0]
        y_flat = p[1] - self.translate[i][1]
        z_flat = p[2]
        if trace:
            print("x_flat", x_flat)
            print("y_flat", y_flat)
            print("z_flat", z_flat)
        inflated = (
            self.axes[i][0] * x_flat
            + self.axes[i][1] * y_flat
            + self.axes[i][2] * z_flat
        )
        return inflated.move(Vertex(*self.poly[i][3]))

    def print_digit(self, i):
        label = LABELS[i]
        dimension = Font(DummyPen()).draw(label)
        text_height = 0.9  # Warning: dependent on size of die
        scale_factor = text_height / dimension[1]

        pen = DigitPen(
            faceno=i,
            border=self.flattened[i],
            scale=scale_factor,
            x_offset=-0.5 * dimension[0],
            y_offset=self.flattened_center[i][1] / scale_factor
            - 0.5 * dimension[1],
            zilla=self,
        )

        Font(pen).draw(label)
        pen.add_steiner_points(trace=i == 4)
        # pen.dump_data(i)
        pen.dump(i)
        pen.make_mesh(i)


main()
