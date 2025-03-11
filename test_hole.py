import math
import svg_writer
import triangle
import repair


def main():
    all_points = []
    segments = []
    for i in range(100):
        theta = (2 * math.pi * i) / 100.0
        all_points.append((10 * math.cos(theta), 10 * math.sin(theta)))
    prev = len(all_points) - 1
    for i in range(len(all_points)):
        segments.append((prev, i))
        prev = i

    for i in range(20):
        theta = (2 * math.pi * i) / 20.0
        all_points.append((7.5 * math.cos(theta), 7.5 * math.sin(theta)))

    hole = []
    for i in range(50):
        theta = (2 * math.pi * i) / 50
        hole.append((2 + 2 * math.cos(theta), 2 * math.sin(theta)))
        if i % 4 == 0:
            # Add Steiner point
            all_points.append(
                (2 + 2.5 * math.cos(theta), 2.5 * math.sin(theta))
            )

    def add_hole():
        start = len(all_points)
        all_points.extend(hole)
        finish = len(all_points)
        prev = finish - 1
        for i in range(start, finish):
            segments.append((i, prev))
            prev = i

    add_hole()

    hole = []
    for i in range(50):
        theta = (2 * math.pi * -i) / 50
        hole.append((-4 + 2 * math.cos(theta), 2 * math.sin(theta)))
    add_hole()

    result = triangle.triangulate(
        {
            "vertices": all_points,
            "segments": segments,
            "holes": [(2, 0), (-4, 0)],
        },
        opts="p",
    )

    vertices = result["vertices"]
    triangles = result["triangles"]
    for a, b in zip(all_points, vertices):
        assert a[0] == b[0]
        assert a[1] == b[1]

    repair.repair(all_points, segments, triangles)

    print(f"{len(all_points)} inputs")
    print(f"{len(vertices)} verticies and {len(triangles)} triangles")

    with svg_writer.SVGWriter("cdt_test", 1, 1) as ctx:
        ctx.set_line_width(0.1)

        if False:
            ctx.move_to(*all_points[s[0]])
            ctx.line_to(*all_points[s[1]])
            ctx.stroke()
        else:
            for t in triangles:
                ctx.move_to(*vertices[t[0]])
                ctx.line_to(*vertices[t[1]])
                ctx.line_to(*vertices[t[2]])
                ctx.close_path()
                ctx.stroke()


main()
