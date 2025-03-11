import math
import font
import svg_writer


def two_thirds(oncurve, ctrl):
    return ((oncurve[0] + 2 * ctrl[0]) / 3.0, (oncurve[1] + 2 * ctrl[1]) / 3.0)


class Pen:
    def __init__(self, ctx):
        self.ctx = ctx
        self.x_offset = 0

    def scale_point(self, p):
        return (
            p[0] + self.x_offset,
            p[1],
        )

    def close_path(self, hole=False, debug=False):
        self.ctx.close_path()
        if hole:
            self.ctx.set_source_rgba(1, 0, 0, 1)
        else:
            self.ctx.set_source_rgba(0, 0, 0, 1)
        if debug:
            path = self.ctx.copy_path()
            print ("debug path")
            for p in path:
                print (p)
        self.ctx.stroke()

    def advance(self, dx):
        self.x_offset += dx

    def move_to(self, point):
        self.ctx.move_to(*self.scale_point(point))

    def line_to(self, point):
        self.ctx.line_to(*self.scale_point(point))

    def curve_to(self, on, off):
        on = self.scale_point(on)
        off = self.scale_point(off)
        ct1 = two_thirds(self.ctx.get_current_point(), off)
        ct2 = two_thirds(on, off)
        self.ctx.curve_to(*ct1, *ct2, *on)

    def dot(self):
        here = self.ctx.get_current_point()
        self.ctx.new_sub_path()
        self.ctx.arc(*here, 50, 0, 2 * math.pi)
        self.ctx.move_to(*here)

    def debug(self):
        print ("Current point is", self.ctx.get_current_point())
        print(f"Path elements: {[str(p) for p in path]}")

def main():
    with svg_writer.SVGWriter("test", 1, 1) as ctx:
        pen = Pen(ctx)
        f = font.Font(pen)
        f.draw(9)


main()
