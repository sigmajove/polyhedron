import svg_writer
import math

p0 = (10 * 2.1569762988, 10 * 2.4950668211)
p1 = (10 * 2.3747626292, 10 * 1.9645745015)
p2 = (10 * 4.4083893922, 10 * 6.0676274578)
p3 = (10 * 2.6180339887, 10 * 1.9021130326)


def dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def circle_through_points(x1, y1, x2, y2, x3, y3):
    # Calculate squared distances
    s1 = x1**2 + y1**2
    s2 = x2**2 + y2**2
    s3 = x3**2 + y3**2

    # Calculate determinants
    M11 = x1 * y2 + x2 * y3 + x3 * y1 - (x2 * y1 + x3 * y2 + x1 * y3)
    M12 = s1 * y2 + s2 * y3 + s3 * y1 - (s2 * y1 + s3 * y2 + s1 * y3)
    M13 = s1 * x2 + s2 * x3 + s3 * x1 - (s2 * x1 + s3 * x2 + s1 * x3)

    # Calculate center coordinates
    x0 = 0.5 * M12 / M11
    y0 = -0.5 * M13 / M11

    # Calculate radius
    r0 = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

    return (x0, y0, r0)


t0 = [
    (10*2.3747626292, 10*1.9645745015),
    (10*4.4083893922, 10*6.0676274578),
    (10*2.1569762988, 10*2.4950668211),
]

t1 = [
    (10*4.4083893922, 10*6.0676274578),
    (10*2.3747626292, 10*1.9645745015),
    (10*2.6180339887, 10*1.9021130326),
]


def main():
    with svg_writer.SVGWriter("three_points", 1, 0.01) as ctx:
        ctx.set_source_rgb(1, 0, 0)
        for p in [p0, p1, p2]:
            ctx.arc(*p, 0.3, 0, 2 * math.pi)
            ctx.fill()
        ctx.set_source_rgb(0, 0, 0)
        ctx.arc(*p3, 0.3, 0, 2 * math.pi)
        ctx.fill()

        x, y, r = circle_through_points(*p1, *p2, *p3)
        print("r", r)
        print("dist0", dist((x, y), p0))
        print("dist1", dist((x, y), p1))
        print("dist2", dist((x, y), p2))

        ctx.arc(x, y, r, 0, 2 * math.pi)
        ctx.stroke()

        ctx.move_to(*t0[0])
        for p in t0[1:]:
            ctx.line_to(*p)
        ctx.close_path()
        ctx.stroke()

        ctx.move_to(*t1[0])
        for p in t1[1:]:
            ctx.line_to(*p)
        ctx.close_path()
        ctx.stroke()
