from font import Font
from poly18 import Vertex
from poly18 import cross_product
from poly18 import dot_product
from poly18 import poly18
from scipy.interpolate import BSpline
import copy
import inside
import math
import random
import svg_writer
import triangle
import numpy as np

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


# The Euclidean distance between two 2D points.
def distance(p, q):
    return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)


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


# The length of segments into which we break down curves.
STEP = 0.005


class DigitPen:
    def __init__(self, border, scale, x_offset, y_offset):
        super().__init__()

        self.upper = []
        self.lower = []

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
                # 3 * STEP, extending from the midpoint of the segment
                # perpendicular to the segment.
                cos = (p[0] - prev[0]) / dist
                sin = (p[1] - prev[1]) / dist
                self.add_raw_steiner(
                    (mid[0] - sin * 3 * STEP, mid[1] + cos * 3 * STEP)
                )
                self.add_raw_steiner(
                    (mid[0] + sin * 3 * STEP, mid[1] - cos * 3 * STEP)
                )
            counter += 1
            if counter == 5:
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

    def triangulate(self, hole):
        result = triangle.triangulate(
            {
                "vertices": self.points,
                "segments": self.segments,
                "holes": self.lower if hole else self.upper,
            },
            opts="p",
        )
        if "triangles" not in result:
            raise RuntimeError("triangle.triangulate failed")
        return (result["vertices"], result["triangles"])

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
            points, triangles = self.triangulate(hole=False)

            ctx.set_source_rgba(0, 0, 1, 0.3)
            for t in triangles:
                ctx.move_to(*points[t[0]])
                ctx.line_to(*points[t[1]])
                ctx.line_to(*points[t[2]])
                ctx.close_path()
                ctx.fill()

            ctx.set_source_rgb(0, 0, 0)
            for t in triangles:
                ctx.move_to(*points[t[0]])
                ctx.line_to(*points[t[1]])
                ctx.line_to(*points[t[2]])
                ctx.close_path()
                ctx.stroke()

            points, triangles = self.triangulate(hole=True)

            ctx.set_source_rgb(0, 0, 0)
            for t in triangles:
                ctx.move_to(*points[t[0]])
                ctx.line_to(*points[t[1]])
                ctx.line_to(*points[t[2]])
                ctx.close_path()
                ctx.stroke()

            print(f"Wrote out {filename}")

    def is_legal_steiner(self, p):
        return inside.inside(p, self.border)

    def add_raw_steiner(self, p):
        if self.is_legal_steiner(p):
            self.points.append(p)

    def add_steiner(self, p):
        self.add_raw_steiner(self.adjust(p))

    def add_steiner_points(self):
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

            closest = min(dist_near(p), default=None)
            if closest is None or closest > 0.05:
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


def main():
    c = Codezilla()
    for x in range(18):
        c.print_digit(x)


class Codezilla:
    def __init__(self):
        self.poly = poly18()
        self.flattened = [None] * len(self.poly)
        self.flattened_center = [None] * len(self.poly)
        self.flattened_top = [None] * len(self.poly)

        self.poles([0, 2, 4, 6, 8])
        self.poles([1, 3, 5, 7, 9])
        for i in (10, 12, 14, 16, 11, 13, 15, 17):
            self.barrel(i)

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
        for vx in vertices:
            x = dot_product(x_axis.value(), vx.value())
            y = dot_product(y_axis.value(), vx.value())
            flat.append([x, y])

        self.flattened[i] = flat

        center = self.poly[i][3]
        x = dot_product(x_axis.value(), center)
        y = dot_product(y_axis.value(), center)
        self.flattened_center[i] = (x, y)
        x = dot_product(x_axis.value(), top.value())
        y = dot_product(y_axis.value(), top.value())
        self.flattened_top[i] = (x, y)

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
        max_y = -math.inf
        for vx in vertices:
            x = dot_product(x_axis.value(), vx.value())
            y = dot_product(y_axis.value(), vx.value())
            if y > max_y:
                max_y = y
            flat.append([x, y])

        # Translate so the coordinate of the apex is(0, 0)
        align = flat[apex][0]
        for f in flat:
            f[0] -= align
            f[1] -= max_y

        self.flattened[i] = flat

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

    def print_digit(self, i):
        label = LABELS[i]

        dimension = Font(DummyPen()).draw(label)
        text_height = 0.45
        scale_factor = text_height / dimension[1]

        pen = DigitPen(
            self.flattened[i],
            scale_factor,
            -0.5 * dimension[0],
            self.flattened_center[i][1] / scale_factor - 0.5 * dimension[1],
        )

        Font(pen).draw(label)
        pen.add_steiner_points()
        pen.dump_data(i)
        pen.dump(i)


main()
