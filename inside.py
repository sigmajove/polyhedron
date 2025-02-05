import math
import cairo

# Given polygon defined by a list of (x, y) vertices in order
# (clockwise or counterclockwise), and a point (x, y), determines whether
# that point is in the interior of the polygon.
# Uses the algorithm described in
# https://en.wikipedia.org/wiki/Point_in_polygon#Ray_casting_algorithm
# The basic idea is to extend a ray infinitely to the right point, and
# count how many times it crosses the boundary. If it crosses an odd number
# of times, the point is on the interior, and if it crosses an even number
# of times the point is on the exterior.

# The are, however, some corner cases to be considered. For example, if the
# point is anywhere on the boundary, we return False.
# See also https://www.scribd.com/document/56214673/Ray-Casting-Algorithm


def inside(point, sides):
    if len(sides) <= 2:
        # At best, a degenerate polygon that can contain no points.
        return False
    prev = sides[-1]
    cross_count = 0
    for here in sides:
        # (Almost) horizontal edges are ignored, since there is no computable
        # point that intersects with the ray.
        if math.isclose(here[1], prev[1]):
            if math.isclose(point[1], 0.5 * (here[1] + prev[1])) and (
                here[0] <= point[0] <= prev[0] or here[0] >= point[0] >= prev[0]
            ):
                # The point is close to the (almost) horizontal edge
                return False
            keep = False
        # If the ray intersects with a vertex, we have to worry about the
        # possibility of couting the crossing twice, the two edges that
        # share that vertex. The trick is to only count as a crossing the
        # intersection of the lower vertex. If the ray crosses a peak,
        # that is zero crossings. If the ray crosses a valley, that is two
        # crossings, but if ray crosses an incline or decline, that will be
        # correctly counted as a single crossing.
        elif here[1] < prev[1]:
            keep = here[1] <= point[1] < prev[1]
        else:
            keep = here[1] > point[1] >= prev[1]

        if keep:
            # The x-coordinate where the ray intersects the edge.
            x = here[0] + (point[1] - here[1]) * (here[0] - prev[0]) / (
                here[1] - prev[1]
            )

            if math.isclose(point[0], x):
                # The point is very near the edge.
                return False

            if point[0] < x:
                # The ray begins to the left of the edge, so it intersects.
                cross_count += 1

        prev = here

    return cross_count % 2 == 1


def shape_test():
    shape = [
        (0, 0),
        (0, 5),
        (1, 5),
        (2, 4),
        (2, 2),
        (3, 2),
        (3, 4),
        (5, 5),
        (5, 2),
        (3, 0),
    ]
    results = set()
    for x in range(-3, 10):
        for y in range(-3, 10):
            if inside((x, y), shape):
                results.add((x, y))
    assert results == {
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 1),
        (3, 1),
        (4, 2),
        (4, 3),
        (4, 4),
    }


def another_test():
    shape = [
        (0, 0),
        (0, 25),
        (5, 15),
        (2, 12),
        (1, 1),
        (6, 1),
        (6, 4),
        (9, 4),
        (9, -3),
    ]
    results = set()
    for x in range(-3, 15):
        for y in range(-8, 30):
            if inside((x, y), shape):
                results.add((x, y))
    assert results == {
        (1, 0),
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (1, 6),
        (1, 7),
        (1, 8),
        (1, 9),
        (1, 10),
        (1, 11),
        (1, 12),
        (1, 13),
        (1, 14),
        (1, 15),
        (1, 16),
        (1, 17),
        (1, 18),
        (1, 19),
        (1, 20),
        (1, 21),
        (1, 22),
        (2, 0),
        (2, 13),
        (2, 14),
        (2, 15),
        (2, 16),
        (2, 17),
        (2, 18),
        (2, 19),
        (2, 20),
        (3, 0),
        (3, 14),
        (3, 15),
        (3, 16),
        (3, 17),
        (3, 18),
        (4, -1),
        (4, 0),
        (4, 15),
        (4, 16),
        (5, -1),
        (5, 0),
        (6, -1),
        (6, 0),
        (7, -2),
        (7, -1),
        (7, 0),
        (7, 1),
        (7, 2),
        (7, 3),
        (8, -2),
        (8, -1),
        (8, 0),
        (8, 1),
        (8, 2),
        (8, 3),
    }


def start_draw():
    rs = cairo.RecordingSurface(cairo.CONTENT_COLOR_ALPHA, None)
    ctx = cairo.Context(rs)
    ctx.scale(100, -100)
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


def inside_test(perimeter, result=None):
    rs, ctx = start_draw()

    ctx.move_to(*perimeter[0])
    for x, y in perimeter[1:]:
        ctx.line_to(x, y)
    ctx.close_path()
    ctx.set_source_rgba(135.0 / 255, 206.0 / 255, 235.0 / 255, 0.5)
    ctx.fill()

    min_x = math.floor(min(p[0] for p in perimeter)) - 5
    max_x = math.ceil(max(p[0] for p in perimeter)) + 5
    min_y = math.floor(min(p[1] for p in perimeter)) - 5
    max_y = math.ceil(max(p[1] for p in perimeter)) + 5

    in_points = set()
    ctx.set_source_rgba(0, 0, 0, 1)
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            ctx.arc(x, y, 0.1, 0, 2 * math.pi)
            if inside((x, y), perimeter):
                in_points.add((x, y))
                ctx.fill()
            else:
                ctx.stroke()

    finish_draw(rs, ctx, "test")

    if result is not None:
        if set(result) != in_points:
            print(f"result {result}")
            print(f"in_points {in_points}")


def run_tests():
    inside_test(
        [
            (0, 0),
            (0, 25),
            (5, 15),
            (2, 12),
            (1, 1),
            (6, 1),
            (6, 4),
            (9, 4),
            (9, -3),
        ],
        [
            (1, 0),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (1, 6),
            (1, 7),
            (1, 8),
            (1, 9),
            (1, 10),
            (1, 11),
            (1, 12),
            (1, 13),
            (1, 14),
            (1, 15),
            (1, 16),
            (1, 17),
            (1, 18),
            (1, 19),
            (1, 20),
            (1, 21),
            (1, 22),
            (2, 0),
            (2, 13),
            (2, 14),
            (2, 15),
            (2, 16),
            (2, 17),
            (2, 18),
            (2, 19),
            (2, 20),
            (3, 0),
            (3, 14),
            (3, 15),
            (3, 16),
            (3, 17),
            (3, 18),
            (4, -1),
            (4, 0),
            (4, 15),
            (4, 16),
            (5, -1),
            (5, 0),
            (6, -1),
            (6, 0),
            (7, -2),
            (7, -1),
            (7, 0),
            (7, 1),
            (7, 2),
            (7, 3),
            (8, -2),
            (8, -1),
            (8, 0),
            (8, 1),
            (8, 2),
            (8, 3),
        ],
    )


shape = [
    (-12.0, 0.0),
    (-9.0, 0.0),
    (-9.0, -4.5),
    (-4.5, -7.5),
    (7.5, -4.5),
    (7.5, 0.0),
    (3.0, 6.0),
    (-4.5, 6.0),
    (-4.5, -3.0),
    (-1.5, -3.0),
    (-1.5, -1.5),
    (-3.0, -1.5),
    (-3.0, 1.5),
    (-1.5, 1.5),
    (-1.5, 3.0),
    (0.0, 3.0),
    (0.0, 1.5),
    (3.0, 1.5),
    (4.5, 0.0),
    (4.5, -1.5),
    (6.0, -1.5),
    (6.0, -4.5),
    (3.0, -4.5),
    (3.0, 0.0),
    (0.0, 0.0),
    (1.5, -3.0),
    (1.5, -4.5),
    (-6.0, -4.5),
    (-7.5, -3.0),
    (-7.5, 7.5),
    (4.5, 7.5),
    (10.5, 1.5),
    (10.5, -10.5),
    (-4.5, -10.5),
    (-12.0, -4.5),
]

inside_test(shape)
