import cairo
import math

def start_draw():
    rs = cairo.RecordingSurface(cairo.CONTENT_COLOR_ALPHA, None)
    ctx = cairo.Context(rs)
    ctx.scale(100, -100)
    ctx.set_line_width(0.015)
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

def distance(p, q):
    return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)

def main():
    a = (0.0, 1.0)
    b = (8.0, 10.0)
    c = (5.0, 0.0)

    x = circumcenter(a, b, c)
    print (x)

    print (distance(x, a))
    print (distance(x, b))
    print (distance(x, c))

    rs, ctx = start_draw()
    ctx.move_to(*a)
    ctx.line_to(*b)
    ctx.line_to(*c)
    ctx.close_path()
    ctx.stroke()
    ctx.arc(*x, distance(x, a), 0, 2*math.pi)
    ctx.stroke()

    finish_draw(rs, ctx, "circum")

main()
