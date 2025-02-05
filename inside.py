import math
import cairo


def inside(point, sides):
    if len(sides) <= 2:
        return False
    prev = sides[-1]
    cross_count = 0
    for here in sides:
        # If the edge is nearly horizontal, we can't determine the
        # the intersection between the ray and the edge.
        # We have to special case the check if the point is on the edge.
        if (
            math.isclose(here[1], point[1])
            and math.isclose(prev[1], point[1])
            and (
                here[0] <= point[0] <= prev[0] or here[0] >= point[0] >= prev[0]
            )
        ):
            return False

        # Determine whether the ray from point to the right intersects with
        # the edge.  First check whether the y coordinate is in range.
        if prev[1] < point[1] < here[1] or prev[1] > point[1] > here[1]:
            # x is the coordinate where the ray interesects the edge.
            x = here[0] + (point[1] - here[1]) * (here[0] - prev[0]) / (
                here[1] - prev[1]
            )
            if math.isclose(x, point[0]):
                # point is very close to an edge.
                # Declare it not to be in the polygon
                return False

            if point[0] < x:
                # The ray begins to the left of the edge, so it intersects.
                cross_count += 1

        prev = here

    # Look for crossings that happen at a vertex
    a = sides[-2]
    b = sides[-1]
    for c in sides:
        if math.isclose(point[0], c[0]) and math.isclose(point[1], c[1]):
            # point is very close to a vertex.
            # Declare it not to be in the polygon
            return False
        if point[1] == b[1] and point[0] < b[0]:
            if (a[1] < b[1] and c[1] > b[1]) or (a[1] > b[1] and c[1] < b[1]):
                cross_count += 1
        a = b
        b = c
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


shape_test()
another_test()


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
    (-8, 0),
    (-6, 0),
    (-6, -3),
    (-3, -5),
    (5, -3),
    (5, 0),
    (2, 4),
    (-3, 4),
    (-3, -2),
    (-1, -2),
    (-1, -1),
    (-2, -1),
    (-2, 1),
    (-1, 1),
    (-1, 2),
    (0, 2),
    (0, 1),
    (2, 1),
    (3, 0),
    (3, -1),
    (4, -1),
    (4, -3),
    (2, -3),
    (2, 0),
    (0, 0),
    (1, -2),
    (1, -3),
    (-4, -3),
    (-5, -2),
    (-5, 5),
    (3, 5),
    (7, 1),
    (7, -7),
    (-3, -7),
    (-8, -3),
]

print(inside((0, -1), shape))
