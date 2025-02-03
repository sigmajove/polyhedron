from p2t import Point
from p2t import CDT
import math
import cairo
from font import Font
from poly18 import poly18
from poly18 import cross_product
from poly18 import dot_product
from poly18 import Vertex

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
    ctx.scale(100, -100)
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


def comment_out():
    # Define the polygon points
    points = [Point(0, 0), Point(100, 0), Point(100, 100), Point(0, 100)]

    for p in points:
        print(p.coordinates())

    # Create a CDT(Constrained Delaunay Triangulation) object
    cdt = CDT(points)

    # Add a hole(optional)
    hole = [Point(25, 25), Point(75, 25), Point(75, 75), Point(25, 75)]
    cdt.add_hole(hole)

    for x in range(10, 91, 10):
        cdt.add_point(Point(x, 10))

    # Triangulate
    triangles = cdt.triangulate()

    # Print the resulting triangles
    for t in triangles:
        print(t.a.coordinates(), t.b.coordinates(), t.c.coordinates())

    rs, ctx = start_draw()
    ctx.move_to(*points[0].coordinates())
    for p in points[1:]:
        ctx.line_to(*p.coordinates())
    ctx.close_path()
    ctx.stroke()

    ctx.move_to(*hole[0].coordinates())
    for p in hole[1:]:
        ctx.line_to(*p.coordinates())
    ctx.close_path()
    ctx.stroke()

    for t in triangles:
        ctx.move_to(*t.a.coordinates())
        ctx.line_to(*t.b.coordinates())
        ctx.line_to(*t.c.coordinates())
        ctx.close_path()
        ctx.stroke()

    finish_draw(rs, ctx, "trangles")


class DummyPen:
    def __init__(self):
        pass

    def move_to(self, p):
        pass

    def curve_to(self, p, q):
        pass

    def line_to(self, p):
        pass

    def advance(self, x):
        pass

    def close_path(self):
        pass


def two_thirds(oncurve, ctrl):
    return ((oncurve[0] + 2 * ctrl[0]) / 3.0, (oncurve[1] + 2 * ctrl[1]) / 3.0)


class CairoPen:
    def __init__(self, scale, x_offset, y_offset, ctx):
        self.scale = scale
        self.ctx = ctx
        self.x_offset = x_offset
        self.y_offset = y_offset

    def advance(self, dx):
        self.x_offset += dx

    def adjust(self, p):
        return (
            (p[0] + self.x_offset) * self.scale,
            (p[1] + self.y_offset) * self.scale,
        )

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


def main():
    c = Codezilla()
    c.print(1)


class DummyPen:
    def __init__(self):
        pass

    def move_to(self, p):
        pass

    def curve_to(self, p, q):
        pass

    def line_to(self, p):
        pass

    def advance(self, x):
        pass

    def close_path(self):
        pass


class Codezilla:
    def __init__(self):
        self.poly = poly18()
        self.flattened = [None] * len(self.poly)
        self.flattened_center = [None] * len(self.poly)
        self.flattened_top = [None] * len(self.poly)

        self.poles([0, 2, 4, 6, 8])
        self.poles([1, 3, 5, 7, 9])

    def poles(self, faces):
        v = self.find_vertex(faces)
        for f in faces:
            self.rotate_pole(f, v)

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
        rs, ctx = start_draw()
        v = self.flattened[i]
        ctx.move_to(*v[0])
        for n in v[1:]:
            ctx.line_to(*n)
        ctx.close_path()
        ctx.stroke()

        label = LABELS[i]
        dimension = Font(DummyPen()).draw(label)
        text_height = 0.45
        scale_factor = text_height / dimension[1]
        Font(
            CairoPen(
                scale_factor,
                -0.5 * dimension[0],
                self.flattened_center[i][1] / scale_factor - 0.5 * dimension[1],
                ctx,
            )
        ).draw(label)
        finish_draw(rs, ctx, "tile")


main()
