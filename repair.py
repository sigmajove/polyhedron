import svg_writer
import math
import three_points


class Triangle:
    def __init__(self, vertices, t):
        self.a = vertices[t[0]]
        self.b = vertices[t[1]]
        self.c = vertices[t[2]]

    def other_point(self, p0, p1):
        result = []
        for p in [self.a, self.b, self.c]:
            if p != p0 and p != p1:
                result.append(p)
        if len(result) != 1:
            raise RuntimeError("other point fails")
        return result[0]

    def plot(self, ctx):
        ctx.move_to(*self.a)
        ctx.line_to(*self.b)
        ctx.line_to(*self.c)
        ctx.close_path()
        ctx.stroke()


class Edge:
    # A (non-boundary) edge has four parameters.
    # e0 and e1 are the endpoints of the edge. Their order does not matter.
    # t0 and t1 are the opposite apex of the adjacent triangles.
    # Their order does not matter.
    def __init__(self, e0, e1, t0, t1):
        self.e0 = e0
        self.e1 = e1
        self.t0 = t0
        self.t1 = t1

    def remap(self, e, f):
        if self.t0 == e:
            self.t0 = f
        elif self.t1 == e:
            self.t1 = f
        else:
            raise RuntimeError("Remap error")

    def can_rotate(self):
        if not is_convex_quad(self.e0, self.t0, self.e1, self.t1):
            return False
        return in_circle(self.e0, self.t0, self.e1, self.t1)

    def plot(self, ctx):
        ctx.move_to(*self.e0)
        ctx.line_to(*self.e1)
        ctx.stroke()


def plot_edges(v, edges, segments, ctx):
    for e0, e1 in segments:
        ctx.move_to(*v[e0])
        ctx.line_to(*v[e1])
        ctx.stroke()

    for e in edges:
        if e.can_rotate():
            if True:
                x, y, r = three_points.circle_through_points(
                    *e.e0, *e.e1, *e.t0
                )
                ctx.set_source_rgb(0, 0, 1)
                ctx.arc(x, y, r, 0, 2 * math.pi)
                ctx.stroke()
            ctx.set_source_rgb(1, 0, 0)
        else:
            ctx.set_source_rgb(0, 0, 0)
        ctx.move_to(*e.e0)
        ctx.line_to(*e.e1)
        ctx.stroke()


def not_plot_edges(edges, forced_set, ctx):
    # Map from vertex to set of vertices connected
    d = dict()

    def add(e0, e1):
        s = d.get(e0, None)
        if s is None:
            s = set()
            d[e0] = s
        s.add(e1)

    for e0, e1 in forced_set:
        add(e0, e1)
        add(e1, e0)
    for e in edges:
        add(e.e0, e.e1)
        add(e.e1, e.e0)

    def del_edge(e0, e1):
        s = d[e0]
        s.remove(e1)
        if not s:
            del d[e0]

    vert = set()
    for e0, s in d.items():
        vert.add(e0)
        for f in s:
            vert.add(f)
    while d:
        e0, s = next(iter(d.items()))
        e1 = next(iter(s))
        ctx.move_to(*e0)
        ctx.line_to(*e1)
        del_edge(e0, e1)
        del_edge(e1, e0)
        while True:
            s = d.get(e1, None)
            if s is None:
                ctx.stroke()
                break
            e2 = next(iter(s))
            ctx.line_to(*e2)
            del_edge(e1, e2)
            del_edge(e2, e1)
            e1 = e2

    for p in vert:
        ctx.arc(*p, 0.025, 0, 2 * math.pi)
        ctx.fill()

    ctx.save()
    ctx.set_source_rgb(1, 0, 0)
    ctx.restore()


# Normalizes the two endpoints of the edge.
def make_edge(p0, p1):
    if p0 == p1:
        raise RuntimeError("Zero-length edge")
    elif p0 < p1:
        return (p0, p1)
    else:
        return (p1, p0)


EPSILON = 1e-6


# Tests whether the line segment (a, b) intersects with (c, d),
# where a, b, c, and d are tuples of the form (x, y).
# If the line segments are (nearly) parallel, returns False, even if
# the four points are colinear.
def if_intersect(a, b, c, d):
    denom = (a[0] - b[0]) * (c[1] - d[1]) + (c[0] - d[0]) * (b[1] - a[1])
    if abs(denom) < EPSILON:
        # The lines segments are (nearly) parallel.
        return False
    s = a[0] * (c[1] - b[1]) + b[0] * (a[1] - c[1]) + c[0] * (b[1] - a[1])
    t = a[0] * (c[1] - d[1]) + c[0] * (d[1] - a[1]) + d[0] * (a[1] - c[1])
    if denom < 0:
        if s >= 0 or s <= denom + EPSILON or t >= 0 or t <= denom + EPSILON:
            return False
    else:
        if s <= 0 or s + EPSILON >= denom or t <= 0 or t + EPSILON >= denom:
            return False
    print("Intersect", s, t, denom)
    return True


# Test if the quadrilateral (a, b, c, d) is convex.
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


# Returns the center point and radius squared of circle through three points.
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
    dist = (p[0]-x0)**2 + (p[1]-y0)**2
    result = dist <= r_squared
    if result:
        print (dist, r_squared, r_squared - dist)
    return result
    a_x = a[0]
    b_x = b[0]
    c_x = c[0]
    p_x = p[0]

    a_y = a[1]
    b_y = b[1]
    c_y = c[1]
    p_y = p[1]

    sa = a_x * a_x + a_y * a_y
    sb = b_x * b_x + b_y * b_y
    sc = c_x * c_x + c_y * c_y
    sp = p_x * p_x + p_y * p_y

    a_x_b_y = a_x * b_y
    a_x_c_y = a_x * c_y
    a_x_p_y = a_x * p_y
    a_y_b_x = a_y * b_x
    a_y_c_x = a_y * c_x
    a_y_p_x = a_y * p_x
    b_x_c_y = b_x * c_y
    b_x_p_y = b_x * p_y
    b_y_c_x = b_y * c_x
    b_y_p_x = b_y * p_x
    c_x_p_y = c_x * p_y
    c_y_p_x = c_y * p_x

    det = (
        sa * (b_x_c_y - b_x_p_y - b_y_c_x + b_y_p_x + c_x_p_y - c_y_p_x)
        + sb * (-a_x_c_y + a_x_p_y + a_y_c_x - a_y_p_x - c_x_p_y + c_y_p_x)
        + sc * (a_x_b_y - a_x_p_y - a_y_b_x + a_y_p_x + b_x_p_y - b_y_p_x)
        + sp * (-a_x_b_y + a_x_c_y + a_y_b_x - a_y_c_x - b_x_c_y + b_y_c_x)
    )

    return det <= 0


def main():
    a = (-5 + 1, 2)
    b = (+5 + 1, 2)
    c = (1, 7)
    print(circle_through_points(*a, *b, *c))


def check_edge(e, t0, t1, ctx):
    t0x = t0.other_point(*e)
    t1x = t1.other_point(*e)

    if not is_convex_quad(t0x, e[0], t1x, e[1]):
        return

    if in_circle(t0x, e[0], e[1], t1x):
        ctx.set_source_rgb(1, 0, 0)
        ctx.move_to(*e[0])
        ctx.line_to(*e[1])
        ctx.stroke()

        ctx.set_source_rgba(0, 1, 1)
        ctx.move_to(*t0x)
        ctx.line_to(*e[0])
        ctx.line_to(*t1x)
        ctx.line_to(*e[1])
        ctx.close_path()
        ctx.stroke()


def repair(vertices, segments, triangles):
    # Copy the data to a structure I can modify.
    my_triangles = [Triangle(vertices, t) for t in triangles]

    # A map from a pair of points (sorted) to the triangles containing
    # those edges.
    edge_map = dict()

    def store_edge(e, t):
        triangles = edge_map.get(e, [])
        triangles.append(t)
        edge_map[e] = triangles

    for t in my_triangles:
        store_edge(make_edge(t.a, t.b), t)
        store_edge(make_edge(t.b, t.c), t)
        store_edge(make_edge(t.c, t.a), t)

    # Create all the edge objects.
    edge_index = dict()
    edge_count = 0
    for e, t in edge_map.items():
        if len(t) > 2:
            raise RuntimeError("Two many triangles")
        if len(t) == 2:
            edge_index[e] = Edge(
                e[0], e[1], t[0].other_point(*e), t[1].other_point(*e)
            )
            edge_count += 1
    assert len(edge_index) == edge_count

    def rotate_edge(e):
        x = edge_index.get(make_edge(e.e0, e.t0))
        if x is not None:
            x.remap(e.e1, e.t1)

        x = edge_index.get(make_edge(e.t0, e.e1))
        if x is not None:
            x.remap(e.e0, e.t1)

        x = edge_index.get(make_edge(e.e1, e.t1))
        if x is not None:
            x.remap(e.e0, e.t0)

        x = edge_index.get(make_edge(e.t1, e.e0))
        if x is not None:
            x.remap(e.e1, e.t0)

        edge_index[make_edge(e.t0, e.t1)] = Edge(e.t0, e.t1, e.e0, e.e1)
        del edge_index[make_edge(e.e0, e.e1)]
        return

    with svg_writer.SVGWriter("repair_debug", 25, 0.1) as ctx:
        found = True
        ctx.set_source_rgb(0, 0, 0)
        ctx.set_line_width(0.05)
        plot_edges(vertices, iter(edge_index.values()), segments, ctx)
        print("Write repair debug")


main()
