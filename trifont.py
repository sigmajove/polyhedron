from font import Font
from poly18 import Vertex
from poly18 import Vector
from poly18 import cross_product
from poly18 import dot_product
from poly18 import poly18
from matplotlib.colors import to_rgb
from scipy.interpolate import BSpline
from stl import mesh
import argparse
import copy
import fixer
import lib3mf
import math
import numpy
import svg_writer
import sys
import triangle


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


# The Euclidean distance between two points
def distance(p, q):
    assert len(p) == len(q)
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p, q)))


# The midpoint of the line segment pq
def midpoint(p, q):
    return (0.5 * (p[0] + q[0]), 0.5 * (p[1] + q[1]))


class Triangle:
    def __init__(self, p, q, r, color=None):
        self.p0 = p
        self.p1 = q
        self.p2 = r
        self.color = color

    def area(self):
        v1 = self.p1 - self.p0
        v2 = self.p2 - self.p0
        c = cross_product(v1, v2)
        return c.magnitude()


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

    def advance(self, x):
        pass

    def close_path(self, lower, inner):
        pass


STEP = 0.025

# The depth of the indentation numbers
DEPTH = 0.05


# Returns a sorted tuple
def normalize(a, b):
    return (a, b) if a <= b else (b, a)


# Determines if the point p is on the line_segment ab, within floating point rounding error.
def on_segment(p, a, b):
    def between(p, x, y):
        return x <= p <= y or x >= p >= y

    # Test if p is in the bounding box containing (a, b)
    if not all(between(p[i], a[i], b[i]) for i in range(2)):
        return False

    # Test if the line from b to a has the same slope as the line from p to a.
    return math.isclose(
        (b[1] - a[1]) * (p[0] - a[0]), (b[0] - a[0]) * (p[1] - a[1])
    )


class DigitPen:
    def __init__(self, faceno, border, scale, x_offset, y_offset, builder):
        self.faceno = faceno
        self.builder = builder
        self.upper = []
        self.lower = []

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

        bad = 0
        for p in self.current:
            back = Vertex(*self.builder.rotate_back((p[0], p[1], 0), faceno))
            if self.builder.find_join_point(back) is None:
                bad += 1
        if bad:
            print(f"border has {bad} of {len(self.current)} bad points")

        # note self.current begins and ends with border[0]
        # Don't use ClosePath to mark the inner point because
        # it will adjust it.
        self.close_path(lower=False, inner=None)

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
    # These come from reverse-engineered TrueType fonts.
    def curve_to(self, on, off):
        self.b_spline(self.current[-1], self.adjust(off), self.adjust(on))

    # Cubic B-spline
    # These come from reverse-engineered Cairo output.
    def cubic(self, a, b, c):
        self.b_spline(
            self.current[-1], self.adjust(a), self.adjust(b), self.adjust(c)
        )

    def b_spline(self, *args):
        k = len(args) - 1
        spline = BSpline(
            numpy.concatenate(
                [
                    numpy.zeros(k),
                    numpy.linspace(0, 1, 2),
                    numpy.ones(k),
                ]
            ),
            numpy.array(args),
            k,
        )

        # Approximate the length of the curve using eight segments.
        length = 0
        prev = self.current[-1]
        for x, y in spline(numpy.linspace(0, 1, 8))[1:]:
            p = (float(x), float(y))
            length += distance(p, prev)
            prev = p

        # Choose a number of segments so that each segment will
        # have length approximately STEP
        num_segments = math.ceil(length / STEP)

        for x, y in spline(numpy.linspace(0, 1, num_segments))[1:]:
            p = (float(x), float(y))
            prev = self.current[-1]
            self.current.append(p)

    def close_path(self, lower, inner):
        if inner is not None:
            (self.lower if lower else self.upper).append(self.adjust(inner))

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

    def triangulate(self, fixer, upper):
        result = triangle.triangulate(
            {
                "vertices": self.points,
                "segments": self.segments,
                # Note: holes specify the regions that get omitted.
                "holes": self.lower if upper else self.upper,
            },
            opts="pnq10",
        )

        # Find all the Steiner points added by the q option.
        in_points = set(tuple(p) for p in self.points)
        out_points = set((float(p[0]), float(p[1])) for p in result["vertices"])

        # Remove all the Steiner points that are on segments.
        # They cause trouble when building the mesh.
        keep_steiner = []
        for p in out_points - in_points:
            if not any(
                on_segment(p, self.points[s[0]], self.points[s[1]])
                for s in self.segments
            ):
                keep_steiner.append(p)

        # Rebuild the triangulation using just the allowed Steiner points.
        result = triangle.triangulate(
            {
                "vertices": self.points + keep_steiner,
                "segments": self.segments,
                # Note: holes specify the region that don't get included
                "holes": self.lower if upper else self.upper,
            },
            opts="pn",
        )

        # The results returned by triangle.triangulate use weird numpy
        # types for the coordinates. To avoid surprises, convert
        # everything to native Python types.
        points = [(float(p[0]), float(p[1])) for p in result["vertices"]]
        triangles = [
            [int(t[0]), int(t[1]), int(t[2])] for t in result["triangles"]
        ]
        neighbors = [
            [int(n[0]), int(n[1]), int(n[2])] for n in result["neighbors"]
        ]

        # Jonathan Richard Shewchuk's Triangle API works pretty well.
        # However, Blender's 3D print toolkit sometimes flags triangles
        # that have an angle approaching 180 degrees. So I make a pass to
        # try to eliminate triangles that have an angle greater or equal to
        # 135 degrees. This pass is probably addressing issues that are
        # no longer present, but I wrote the code, and will be leaving
        # it in for the nonce.
        fixer.remove_very_obtuse_triangles(points, triangles, neighbors)

        # Find all the boundary edges.
        # These are found in triangles with missing neighbors.
        boundaries = []
        for tid, n in enumerate(neighbors):
            for i, neigh in enumerate(n):
                if neigh < 0:
                    tri = triangles[tid]
                    # List the vertices of the edge in counterclockwise order.
                    match i:
                        case 0:
                            edge = (tri[1], tri[2])
                        case 1:
                            edge = (tri[2], tri[0])
                        case 2:
                            edge = (tri[0], tri[1])
                        case _:
                            raise RuntimeError(
                                "Triangle has more than three neighbors"
                            )
                    if upper:
                        # Use clockwise order
                        edge = (edge[1], edge[0])
                    boundaries.append(edge)
        return (points, triangles, boundaries)

    def advance(self, dx):
        self.x_offset += dx

    # Maps the coordinate system used internally by Font to the
    # one we are using to construct the mesh.
    def adjust(self, p):
        return (
            (p[0] + self.x_offset) * self.scale,
            (p[1] + self.y_offset) * self.scale,
        )

    def write_tile(self, i, fixer):
        filename = f"tile{i:02d}"
        with svg_writer.SVGWriter(filename, 25, 1) as ctx:
            ctx.set_line_width(0.001)
            points, triangles, _ = self.triangulate(fixer, upper=False)

            ctx.set_source_rgb(0, 0, 0)
            for t in triangles:
                # Shade the region that is at the lower level.
                ctx.move_to(*points[t[0]])
                ctx.line_to(*points[t[1]])
                ctx.line_to(*points[t[2]])
                ctx.close_path()
                ctx.set_source_rgba(0, 0, 1, 0.4)
                ctx.fill()

                ctx.set_source_rgb(0, 0, 0)
                ctx.move_to(*points[t[0]])
                ctx.line_to(*points[t[1]])
                ctx.line_to(*points[t[2]])
                ctx.close_path()
                ctx.stroke()

            points, triangles, _ = self.triangulate(fixer, upper=True)

            ctx.set_source_rgb(0, 0, 0)
            for t in triangles:
                ctx.move_to(*points[t[0]])
                ctx.line_to(*points[t[1]])
                ctx.line_to(*points[t[2]])
                ctx.close_path()
                ctx.stroke()
        print(f"Wrote out {filename}")

    def make_mesh(self, fixer, i):
        upper_points, upper_triangles, upper_boundary = self.triangulate(
            fixer, upper=True
        )
        lower_points, lower_triangles, lower_boundary = self.triangulate(
            fixer, upper=False
        )
        if False:
            with svg_writer.SVGWriter("debug", 50, 0.01) as ctx:
                bpoints = set()
                for e, f in lower_boundary:
                    bpoints.add(e)
                    bpoints.add(f)
                min_y = math.inf
                for p in bpoints:
                    xx = lower_points[p]
                    ctx.arc(*xx, 0.01, 0, 2 * math.pi)
                    min_y = min(min_y, xx[1])
                    ctx.set_source_rgb(1, 0, 0)
                    ctx.fill()

                min_y -= 0.5
                bpoints = set()
                for e, f in upper_boundary:
                    bpoints.add(e)
                    bpoints.add(f)
                for p in bpoints:
                    xx = upper_points[p]
                    ctx.arc(xx[0], xx[1] - min_y, 0.01, 0, 2 * math.pi)
                    ctx.set_source_rgb(0, 0, 1)
                    ctx.fill()

        num_triangles = len(lower_triangles) + len(upper_triangles)

        # The combined points will be lower + upper
        def to_upper(p):
            return p + len(lower_points)

        combined_points = []
        for p in lower_points:
            combined_points.append((p[0], p[1], -DEPTH))
        for p in upper_points:
            combined_points.append((p[0], p[1], 0))

        mesh_triangles = [t for t in lower_triangles]
        for t in upper_triangles:
            mesh_triangles.append(tuple([to_upper(p) for p in t]))

        for e in set(lower_boundary) & set(upper_boundary):
            #  u0 -- u1
            #  | \    |
            #  |  \   |
            #  |   \  |
            #  |    \ |
            #  l0 --- l1
            l0 = e[0]
            l1 = e[1]
            u0 = to_upper(e[0])
            u1 = to_upper(e[1])

            # This is a very delicate operation. Order matters.
            # I wish I had a better explanation for why it needs
            # to be this way. The model builders don't complain.
            mesh_triangles.append((l0, u0, l1))
            mesh_triangles.append((u0, u1, l1))
            num_triangles += 2

        # Translate the 2d triangulation back to its
        # original 3d coordinates.
        for j, p in enumerate(combined_points):
            combined_points[j] = self.builder.correct(
                self.builder.rotate_back(p, i)
            )

        for p in combined_points:
            if not isinstance(p, Vertex):
                print(f"Bad point {type(p)}")

        for t in mesh_triangles:
            self.builder.big_mesh.append(
                Triangle(
                    combined_points[t[0]],
                    combined_points[t[1]],
                    combined_points[t[2]],
                )
            )

    def make_colored_mesh(self, fixer, faceno):
        for color in (True, False):
            points, triangles, _ = self.triangulate(fixer, upper=not color)

            # Translate the 2d triangulation back to its original
            # 3d coordinates.
            three_d_points = [
                self.builder.correct(
                    self.builder.rotate_back((p[0], p[1], 0), faceno)
                )
                for p in points
            ]
            for t in triangles:
                self.builder.big_mesh.append(
                    Triangle(
                        three_d_points[t[0]],
                        three_d_points[t[1]],
                        three_d_points[t[2]],
                        color=color,
                    )
                )


def main():
    parser = argparse.ArgumentParser(
        prog="die18", description="A 3D Model of an 18-sided die"
    )
    parser.add_argument(
        "-c", "--color", action="store_true", help="Create color model"
    )
    parser.add_argument(
        "-f", "--foreground", default="black", help="Color of the numbers"
    )
    parser.add_argument(
        "-b", "--background", default="white", help="Color of the die"
    )
    args = parser.parse_args()

    args_error = None
    try:
        rgb = to_rgb(args.foreground)
        args.foreground_rgb = rgb
    except:
        print(f'Unknown foreground color "{args.foreground}"')
        args_error = 1

    try:
        rgb = to_rgb(args.background)
        args.background_rgb = rgb
    except:
        print(f'Unknown background color "{args.foreground}"')
        args_error = 1

    if args_error:
        return args_error

    c = BuildModel(args)
    f = fixer.Fixer()
    for i in range(18):
        c.print_digit(i, f)
    f.print_statistics()
    c.check_mesh()
    if args.color:
        c.make_3mf()
    else:
        c.make_stl()
    return 0


def internal_points(r0, r1):
    n = math.ceil(distance(r0, r1) / (10 * STEP))
    dx = r1[0] - r0[0]
    dy = r1[1] - r0[1]
    dz = r1[2] - r0[2]
    for i in range(1, n):
        mx = r0[0] + (i * dx) / n
        my = r0[1] + (i * dy) / n
        mz = r0[2] + (i * dz) / n
        yield (mx, my, mz)


class BuildModel:
    def __init__(self, args):
        self.args = args
        self.poly = poly18()
        self.flattened = [None] * len(self.poly)
        self.flattened_center = [None] * len(self.poly)
        self.flattened_top = [None] * len(self.poly)

        # x, y, z normal vectors that will allow transforming
        # the flattened mesh back to its original place on the polyhedron.
        self.axes = [None] * len(self.poly)

        self.translate = [None] * len(self.poly)

        self.store_join_points()

        # This is the model we are producing. A list of Triangles.
        self.big_mesh = []

        self.poles([0, 2, 4, 6, 8])
        self.poles([1, 3, 5, 7, 9])
        for i in (10, 12, 14, 16, 11, 13, 15, 17):
            self.barrel(i)

    # Join points are on the intersection of two faces.
    # When we map 3d points to 2d points and back to 3d,
    # we don't necessarily get back the same points we put in.
    # If the point we get back is close to join point, we move it
    # back to the join point.
    def store_join_points(self):
        edges = dict()
        for x, face in enumerate(self.poly):
            prev = face[0][-1]
            for p in face[0]:
                key = normalize(prev.value(), p.value())
                edges[key] = edges.get(key, 0) + 1
                prev = p
        for v in edges.values():
            assert v == 2

        vertices = set()
        for k in edges.keys():
            vertices.add(k[0])
            vertices.add(k[1])

        self.join_points = [Vertex(*p) for p in vertices]
        for k in edges.keys():
            for p in internal_points(k[0], k[1]):
                self.join_points.append(Vertex(*p))

    def find_join_point(self, v):
        result = None
        for p in self.join_points:
            if p.isclose(v):
                if result is not None:
                    print("ambiguous join")
                    print("approx", v)
                    print(result)
                    print(p)
                    raise RuntimeError("bad")
                result = p
        return result

    def is_join_point(self, v):
        assert isinstance(v, Vertex)
        return v in self.join_points

    def correct(self, v):
        v = Vertex(*v)
        c = self.find_join_point(v)
        return v if c is None else c

    def make_stl(self):
        model = mesh.Mesh(
            numpy.zeros(len(self.big_mesh), dtype=mesh.Mesh.dtype)
        )
        for i, t in enumerate(self.big_mesh):
            model.vectors[i][0] = t.p0.value()
            model.vectors[i][1] = t.p1.value()
            model.vectors[i][2] = t.p2.value()
        model.check(exact=True)
        filename = "c:/users/sigma/documents/model18.stl"
        model.save(filename)
        print(f"Wrote out {filename}")

    def make_3mf(self):
        # Use the lib3mf library to create a 3mf file containing
        # a colored model.

        # I couldn't find any useful documentation for the Python
        # bindings for lib3mf. The following is a combination of
        # cut-and-paste from AIs, experimenation, guesswork, and
        # peeking at the C++ source code on GitHub.

        wrapper = lib3mf.get_wrapper()
        model = wrapper.CreateModel()

        model.SetUnit(lib3mf.ModelUnit.MilliMeter)

        # Create a color group
        color_group = model.AddColorGroup()

        # Define the property of the triangles that make up the die.
        background_property_id = color_group.AddColor(
            wrapper.FloatRGBAToColor(*self.args.background_rgb, 1.0)
        )
        background_props = lib3mf.TriangleProperties()
        background_props.ResourceID = color_group.GetResourceID()
        background_props.PropertyIDs[0] = background_property_id
        background_props.PropertyIDs[1] = background_property_id
        background_props.PropertyIDs[2] = background_property_id

        # Define the property of the triangles that make up the digits.
        foreground_property_id = color_group.AddColor(
            wrapper.FloatRGBAToColor(*self.args.foreground_rgb, 1.0)
        )
        foreground_props = lib3mf.TriangleProperties()
        foreground_props.ResourceID = color_group.GetResourceID()
        foreground_props.PropertyIDs[0] = foreground_property_id
        foreground_props.PropertyIDs[1] = foreground_property_id
        foreground_props.PropertyIDs[2] = foreground_property_id

        mesh_object = model.AddMeshObject()

        # Create an index of all the 3D coordinates in the input mesh.
        points = dict()
        next = 0

        def add_point(p):
            nonlocal next
            if points.setdefault(p, next) == next:
                next += 1

        for t in self.big_mesh:
            add_point(t.p0.value())
            add_point(t.p1.value())
            add_point(t.p2.value())

        # Create a list of vertices indexed by the integers assigned
        # and written into dict.
        vertices = len(points) * [None]
        for p, i in points.items():
            position = lib3mf.Position()
            for j, c in enumerate(p):
                position.Coordinates[j] = c
            mesh_object.AddVertex(position)
            vertices[i] = position

        triangles = []
        colors = []
        for t in self.big_mesh:
            triangle = lib3mf.Triangle()
            triangle.Indices[0] = points[t.p0.value()]
            triangle.Indices[1] = points[t.p1.value()]
            triangle.Indices[2] = points[t.p2.value()]

            mesh_object.AddTriangle(triangle)
            triangles.append(triangle)

            # Keep track of the color of each triangle.
            # We are assuming that the indices returned by
            # mesh_object.AddTriangle are consecutive integers
            # beginning with zero. I don't like that assumption
            # but I couldn't get anything else to work.
            colors.append(foreground_props if t.color else background_props)

        mesh_object.SetGeometry(vertices, triangles)

        # It appears by experimentation that I cannot set the colors
        # of the triangle in the model until after SetGeometry is called.
        # I would prefer to set the colors one at a time, using
        # mesh_object.SetTriangleProperties, using the indices
        # returned by mesh_object.AddTriangle above. But that doesn't work.
        # SetAllTriangleProperties sets an object-level property (whatever
        # that is) and SetTriangleProperty doesn't. And the object-level
        # property seems to be needed to write out a colored model.
        mesh_object.SetAllTriangleProperties(colors)

        model.AddBuildItem(mesh_object, wrapper.GetIdentityTransform())

        # Save the model to a 3MF file
        filename = "c:/users/sigma/documents/model18.3mf"
        model.QueryWriter("3mf").WriteToFile(filename)
        print(f"Wrote out {filename}")

    def check_mesh(self):
        counter = 0
        edges = dict()
        for t in self.big_mesh:

            def add_triangle(key):
                s = edges.get(key, None)
                if s is None:
                    s = []
                    edges[key] = s
                s.append(t)

            add_triangle(normalize(t.p0, t.p1))
            add_triangle(normalize(t.p1, t.p2))
            add_triangle(normalize(t.p2, t.p0))

            counter += 1
        print(f"{counter} triangles")
        print(f"{len(edges)} edges")
        bad_triangles = 0
        for k, v in edges.items():
            if len(v) != 2:
                bad_triangles += 1
        print(bad_triangles, "Bad triangles")

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
        self.axes[i] = (x_axis, y_axis, self.poly[i][2])
        self.translate[i] = (0, 0)

        flat = []
        for vx in vertices:
            x = dot_product(x_axis.value(), vx.value())
            y = dot_product(y_axis.value(), vx.value())
            flat.append([x, y])

        self.flattened[i] = flat

        for j, v in enumerate(vertices):
            back = Vertex(*self.rotate_back((flat[j][0], flat[j][1], 0), i))
            if not back.isclose(v):
                self.check_axes(i)
                print("orig =", v)
                print("back =", back)
                raise RuntimeError("round trip failure")

        center = self.poly[i][3]
        x = dot_product(x_axis.value(), center)
        y = dot_product(y_axis.value(), center)
        self.flattened_center[i] = (x, y)
        x = dot_product(x_axis.value(), top.value())
        y = dot_product(y_axis.value(), top.value())
        self.flattened_top[i] = (x, y)

    def check_axes(self, i):
        axes = self.axes[i]

        # Check that the axes are all unit vectors.
        for a in axes:
            assert math.isclose(a.dx * a.dx + a.dy * a.dy + a.dz * a.dz, 1.0)

        # Check that they follow the right-hand rule
        assert axes[0].isclose(cross_product(axes[1], axes[2]))
        assert axes[1].isclose(cross_product(axes[2], axes[0]))
        assert axes[2].isclose(cross_product(axes[0], axes[1]))

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

        self.axes[i] = (x_axis, y_axis, self.poly[i][2])
        flat = []
        max_y = -math.inf
        center = Vertex(*self.poly[i][3])
        for v in vertices:
            x = dot_product(x_axis.value(), v.value())
            y = dot_product(y_axis.value(), v.value())
            if y > max_y:
                max_y = y
            flat.append([x, y])

        # Translate so the coordinate of the apex is(0, 0)
        align = flat[apex][0]
        self.translate[i] = (-align, -max_y)

        for f in flat:
            f[0] -= align
            f[1] -= max_y

        self.flattened[i] = flat

        for j, v in enumerate(vertices):
            back = Vertex(*self.rotate_back((flat[j][0], flat[j][1], 0), i))
            if not back.isclose(v):
                self.check_axes(i)
                print("orig =", v)
                print("back =", back)
                raise RuntimeError("round trip failure")

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

    # Move the flattened point back to its original 3d position
    def rotate_back(self, p, i, trace=False):
        x_flat = p[0] - self.translate[i][0]
        y_flat = p[1] - self.translate[i][1]
        z_flat = p[2]
        if trace:
            print("x_flat", x_flat)
            print("y_flat", y_flat)
            print("z_flat", z_flat)
        inflated = (
            self.axes[i][0] * x_flat
            + self.axes[i][1] * y_flat
            + self.axes[i][2] * z_flat
        )
        return inflated.move(Vertex(*self.poly[i][3]))

    def print_digit(self, faceno, fixer):
        label = LABELS[faceno]
        dimension = Font(DummyPen()).draw(label)
        text_height = 0.9  # Warning: dependent on size of die
        scale_factor = text_height / dimension[1]

        pen = DigitPen(
            faceno=faceno,
            border=self.flattened[faceno],
            scale=scale_factor,
            x_offset=-0.5 * dimension[0],
            y_offset=self.flattened_center[faceno][1] / scale_factor
            - 0.5 * dimension[1],
            builder=self,
        )

        Font(pen).draw(label)
        pen.write_tile(faceno, fixer)
        if self.args.color:
            pen.make_colored_mesh(fixer, faceno)
        else:
            pen.make_mesh(fixer, faceno)


if __name__ == "__main__":
    sys.exit(main())
