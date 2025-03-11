from p2t import Point
from p2t import CDT
import cairo

def start_draw():
    rs = cairo.RecordingSurface(cairo.CONTENT_COLOR_ALPHA, None)
    ctx = cairo.Context(rs)
    ctx.scale(1, -1)
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

# Define the polygon points
points = [Point(0, 0), Point(100, 0), Point(100, 100), Point(0, 100)]

for p in points:
    print (p.coordinates())

# Create a CDT (Constrained Delaunay Triangulation) object
cdt = CDT(points)

# Add a hole (optional)
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
