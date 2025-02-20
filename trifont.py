from font import Font
from p2t import CDT
from p2t import Point
from poly18 import Vertex
from poly18 import cross_product
from poly18 import dot_product
from poly18 import poly18
from scipy.interpolate import BSpline
import cairo
import copy
import inside
import math
import random
import numpy as np

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


def start_draw():
    rs = cairo.RecordingSurface(cairo.CONTENT_COLOR_ALPHA, None)
    ctx = cairo.Context(rs)
    ctx.scale(1000, -1000)
    ctx.set_line_width(0.005)
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

    def add_steiner(self, p):
        pass

    def move_to(self, p):
        pass

    def curve_to(self, p, q):
        pass

    def line_to(self, p):
        pass

    def advance(self, x):
        pass

    def close_path(self, hole=False):
        pass


def two_thirds(oncurve, ctrl):
    return ((oncurve[0] + 2 * ctrl[0]) / 3.0, (oncurve[1] + 2 * ctrl[1]) / 3.0)


STEP = 0.005


def shorten_triangles(triangles):
    counter = 0
    stack = []
    while counter < 500:
        if len(stack) > 0:
            a, b, c = stack.pop()
        else:
            t = next(triangles, None)
            if t is None:
                break
            a = t.a.coordinates()
            b = t.b.coordinates()
            c = t.c.coordinates()
        dab = distance(a, b)
        dac = distance(a, c)
        dbc = distance(b, c)
        dmax = max(dab, dac, dbc)
        dmin = min(dab, dac, dbc)
        if 8 * dmin >= dmax:
            # The triangle is not too skinny.Output it.
            yield ((a, b, c))
        elif dmax == dab:
            mid = (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))
            stack.append((a, mid, c))
            stack.append((mid, b, c))
            counter += 1
        elif dmax == dac:
            mid = (0.5 * (a[0] + c[0]), 0.5 * (a[1] + c[1]))
            stack.append((a, b, mid))
            stack.append((b, c, mid))
            counter += 1
        else:
            mid = (0.5 * (b[0] + c[0]), 0.5 * (b[1] + c[1]))
            stack.append((a, b, mid))
            stack.append((mid, c, a))
            counter += 1
    for a, b, c in stack:
        yield ((a, b, c))
    for t in triangles:
        yield ((t.a.coordinates(), t.b.coordinates(), t.c.coordinates()))


# Returns the cosine of the smallest angle in the triangle formed a, b, c
def min_angle(a, b, c):
    # Compute the lengths of the three sides.
    dab_sq = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
    dbc_sq = (b[0] - c[0]) ** 2 + (b[1] - c[1]) ** 2
    dac_sq = (a[0] - c[0]) ** 2 + (a[1] - c[1]) ** 2
    dab = math.sqrt(dab_sq)
    dbc = math.sqrt(dbc_sq)
    dac = math.sqrt(dac_sq)

    # Use the law of cosines to get the cosines of the angles
    a0 = (dab_sq + dbc_sq - dac_sq) / (dab * dbc)
    a1 = (dab_sq + dac_sq - dbc_sq) / (dab * dac)
    a2 = (dbc_sq + dac_sq - dab_sq) / (dbc * dac)
    return max(a0, a1, a2) / 2


def new_point(triangles):
    max_cos = -2
    for t in triangles:
        m = min_angle(t.a.coordinates(), t.b.coordinates(), t.c.coordinates())
        if m > max_cos:
            max_cos = m
            save = t
    return save
    return circumcenter(
        save.a.coordinates(), save.b.coordinates(), save.c.coordinates()
    )


def CheckPoints(points):
    prev = (points[-1][0], points[-1][1])
    for i, p in enumerate(points):
        here = (p[0], p[1])
        if here == prev:
            print("Dup", i, here)
        prev = here


class CDTPen:
    def __init__(self, points, scale, x_offset, y_offset):
        self.scale = scale
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.points = []
        prev = points[-1]
        for p in points:
            self.add_points(prev, p)
            prev = p
        self.current_hole = []
        self.holes = []
        self.islands = []
        self.steiner = []
        self.curve_steiner = []

    def add_points(self, r0, r1):
        n = math.ceil(distance(r0, r1) / (10 * STEP))
        dx = r1[0] - r0[0]
        dy = r1[1] - r0[1]
        for i in range(0, n):
            mx = r0[0] + (i * dx) / n
            my = r0[1] + (i * dy) / n
            self.points.append((mx, my))

    def dump_data(self, i):
        self.add_steiner_points()
        rs, ctx = start_draw()
        ctx.move_to(*self.points[0])
        for p in self.points[1:]:
            ctx.line_to(*p)
        ctx.close_path()
        ctx.stroke()

        for h in self.holes:
            ctx.move_to(*h[0])
            for p in h[1:]:
                ctx.line_to(*p)
            ctx.close_path()
            ctx.stroke()

        ctx.set_source_rgba(1, 0, 0, 1)
        for s in self.steiner:
            ctx.arc(*s, 0.005, 0, 2 * math.pi)
            ctx.fill()

        for island in self.islands:
            ctx.move_to(*island[0][0])
            for p in island[0][1:]:
                ctx.line_to(*p)
            ctx.close_path()
            ctx.stroke()

            for s in island[1]:
                ctx.arc(*s, 0.005, 0, 2 * math.pi)
                ctx.fill()

        finish_draw(rs, ctx, f"data{i:02d}")

    def triangulate(self):
        cdt = CDT([Point(*p) for p in self.points])
        for h in self.holes:
            # Sometimes font + CDT creates the same point at the beginning
            # and end of the list of hole points. This causes CDT to fail
            # silently. Hack around this issue.
            cp = h[:]
            if len(cp) >= 2 and cp[0][0] == cp[-1][0] and cp[0][1] == cp[-1][1]:
                cp.pop()
            cdt.add_hole([Point(*p) for p in cp])
        for s in self.steiner:
            cdt.add_point(Point(*s))
        result = cdt.triangulate()
        for h, s in self.islands:
            cp = h[:]
            if len(cp) >= 2 and cp[0][0] == cp[-1][0] and cp[0][1] == cp[-1][1]:
                cp.pop()

            cdt2 = CDT([Point(*p) for p in cp])
            for p in s:
                cdt2.add_point(Point(*p))
            result += cdt2.triangulate()
        return result

    def add_steiner(self, p):
        pp = self.adjust(p)
        self.steiner.append(pp)

    def raw_steiner(self, p):
        self.steiner.append(p)

    def add_curve_steiner(self, p):
        self.curve_steiner.append(p)

    def clean_steiner(self):
        self.steiner = [p for p in self.steiner if self.is_legal_steiner(p)]

    def advance(self, dx):
        self.x_offset += dx

    def adjust(self, p):
        return (
            (p[0] + self.x_offset) * self.scale,
            (p[1] + self.y_offset) * self.scale,
        )

    def move_to(self, p0):
        self.current_point = self.adjust(p0)
        self.start_path = self.current_point
        self.current_hole.append(self.current_point)

    def interpolate(self, r0, r1):
        n = math.ceil(distance(r0, r1) / (10 * STEP))
        dx = r1[0] - r0[0]
        dy = r1[1] - r0[1]
        for i in range(1, n):
            mx = r0[0] + (i * dx) / n
            my = r0[1] + (i * dy) / n
            self.current_hole.append((mx, my))

    def line_to(self, p0):
        v1 = self.adjust(p0)
        self.interpolate(self.current_point, v1)
        self.current_point = v1
        self.current_hole.append(v1)

    def close_path(self, hole=False):
        self.interpolate(self.current_point, self.start_path)
        if hole:
            self.islands.append([self.current_hole, self.curve_steiner])
        elif len(self.current_hole):
            self.holes.append(self.current_hole)
            self.steiner.extend(self.curve_steiner)
        self.current_hole = []
        self.curve_steiner = []

    def curve_to(self, on, off):
        control_points = np.array(
            [self.current_point, self.adjust(off), self.adjust(on)]
        )
        k = 2  # quadratic
        n = len(control_points)
        t = np.concatenate(
            [np.zeros(k), np.linspace(0, 1, n - k + 1), np.ones(k)]
        )
        spl = BSpline(t, control_points, k)
        u = np.linspace(0, 1, 8)
        length = 0
        prev_x = self.current_point
        for x in spl(u)[1:]:
            length += distance(x, prev_x)
            prev_x = x
        z = math.ceil(length / STEP)
        u = np.linspace(0, 1, num=z)

        counter = 0
        prev_x = self.current_point
        for x in spl(u)[1:]:
            self.current_hole.append(x)
            if counter == 2:
                dist = distance(x, prev_x)
                mid = midpoint(x, prev_x)
                c = (x[0] - prev_x[0]) / dist
                s = (x[1] - prev_x[1]) / dist
                sx = mid[0] - s * 3 * STEP
                sy = mid[1] + c * 3 * STEP
                self.add_curve_steiner((sx, sy))
            counter += 1
            if counter == 5:
                counter = 0
            prev_x = x

        self.current_point = self.adjust(on)

    # on_adj = self.adjust(on)
    # off_adj = self.adjust(off)
    # ct1 = two_thirds(self.ctx.get_current_point(), off_adj)
    # ct2 = two_thirds(on_adj, off_adj)
    # self.ctx.curve_to(*ct1, *ct2, *on_adj)

    def dump(self, i):
        rs, ctx = start_draw()

        ttt = self.triangulate()
        ctx.set_source_rgba(0, 0, 0, 1)
        for t in ttt:
            ctx.move_to(*t.a.coordinates())
            ctx.line_to(*t.b.coordinates())
            ctx.line_to(*t.c.coordinates())
            ctx.close_path()
            ctx.stroke()

        filename = f"tile{i:02d}"
        finish_draw(rs, ctx, filename)
        print(f"Wrote out {filename}")

    def is_legal_steiner(self, p):
        if not inside.inside(p, self.points):
            return False
        for h in self.holes:
            if inside.inside(p, h):
                return False
        return True

    def add_steiner_points(self):
        min_x = min(p[0] for p in self.points)
        max_x = max(p[0] for p in self.points)
        min_y = min(p[1] for p in self.points)
        max_y = max(p[1] for p in self.points)
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
        for s in self.steiner:
            add_to_bucket(s)
        for h in self.holes:
            for p in h:
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
                self.raw_steiner(p)
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
        c.print(x)


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

    def print(self, i):
        label = LABELS[i]
        dimension = Font(DummyPen()).draw(label)
        text_height = 0.45
        scale_factor = text_height / dimension[1]

        pen = CDTPen(
            self.flattened[i],
            scale_factor,
            -0.5 * dimension[0],
            self.flattened_center[i][1] / scale_factor - 0.5 * dimension[1],
        )

        Font(pen).draw(label)
        pen.clean_steiner()
        pen.dump_data(i)
        pen.dump(i)


main()
