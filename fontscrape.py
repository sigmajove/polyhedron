import cairo
import math
from fontTools.ttLib import TTFont
from fontTools.pens.basePen import BasePen


def two_thirds(oncurve, ctrl):
    x = (oncurve[0] + 2 * ctrl[0]) / 3.0
    y = (oncurve[1] + 2 * ctrl[1]) / 3.0
    return (x, y)


class MyGuy:
    def __init__(self):
        self.rs = cairo.RecordingSurface(cairo.CONTENT_COLOR_ALPHA, None)
        self.ctx = cairo.Context(self.rs)
        self.ctx.scale(0.1, -0.1)
        self.ctx.set_line_width(5)
        self.ctx.set_source_rgba(0, 0, 0, 1)
        self.next_r = 25.0
        self.first = True

    def moveTo(self, p0):
        print("move_to", *p0)
        self.ctx.move_to(*p0)
        self.next_r = 50.0

    def lineTo(self, p1):
        print("line_to", *p1)
        self.ctx.line_to(*p1)

    def oneCurve(self, on, off):
        ct1 = two_thirds(self.ctx.get_current_point(), off)
        ct2 = two_thirds(on, off)
        self.ctx.curve_to(*ct1, *ct2, *on)

    def qCurveTo(self, *args):
        assert len(args) == 2 or len(args) == 3
        if len(args) == 2:
            self.oneCurve(args[1], args[0])
        else:
            implicit = tuple(0.5 * (args[0][i] + args[1][i]) for i in range(2))
            self.oneCurve(implicit, args[0])
            self.oneCurve(args[2], args[1])

        return
        # self.ctx.arc(*(args[-1]), 25.0, 0, math.pi * 2)
        # self.ctx.fill()
        return
        if len(args) == 3:
            p0, p1, p2 = args
        else:
            p1, p2 = args
            p0 = tuple((p1[i] + p2[i]) / 2 for i in range(2))
        n = 20
        for i in range(n):
            t = i / float(n - 1)
            omt = 1.0 - t
            p = tuple(
                omt * (omt * p0[i] + t * p1[i]) + t * (omt * p1[i] + t * p2[i])
                for i in range(2)
            )
            self.ctx.line_to(*p)

    def closePath(self):
        print("close_path")
        self.ctx.close_path()
        self.ctx.stroke()

    def plotCurve(self, on, off):
        self.ctx.curve(
            two_thirds(self.ctx.get_current_point(), off.point()),
            two_thirds(on.point(), off.point()),
            *on,
        )

    def plotContour(self, c):
        expanded = []
        # Add the implied points.
        for i in range(len(c)):
            prev_i = i - 1 if i > 0 else len(c) - 1
            if not c[prev_i].on and not c[i].on:
                # Add the implied on point that it midway between the two offs.
                expanded.append(
                    TTPoint(
                        x=(c[prev_i].x + c[i].x) * 0.5,
                        y=(c[prev_i].y + c[i].y) * 0.5,
                        on=1,
                    )
                )
            expanded.append(c[i])

        for i, x in enumerate(expanded):
            print(f"({i}: {x.point()} on={x.on}")

        for i in range(len(expanded) - 1, -1, -1):
            if expanded[i].on:
                print(f"Move to {i}")
                self.ctx.move_to(*expanded[i].point())
                break

        for i in range(len(expanded)):
            if not expanded[i].on:
                continue
            prev_i = i - 1 if i > 0 else len(expanded) - 1
            if expanded[prev_i].on:
                print(f"Line to {i}")
                self.ctx.line_to(*expanded[i].point())
            else:
                print(f"Curve to {i}")
                self.plotCurve(expanded[i], expanded[prev_i])
        self.ctx.close_path()
        self.ctx.stroke()

    def finish(self):
        x, y, width, height = self.rs.ink_extents()
        print("dim", width, height)
        surface = cairo.SVGSurface(
            "C:/Users/sigma/Documents/truetype.svg", width, height
        )
        ccc = cairo.Context(surface)
        ccc.set_source_surface(self.rs, -x, -y)
        ccc.paint()
        surface.finish()
        surface.flush()
        self.rs.finish()


def main():
    font = TTFont(
        "c:/Users/sigma/AppData/Local/Microsoft/Windows/Fonts/Helvetica-Bold.ttf"
        # "c:/Windows/Fonts/verdana.ttf"
    )
    glyphSet = font.getGlyphSet()
    glyph = glyphSet["nine"]

    pen = MyGuy()
    glyph.draw(pen)
    pen.finish()


class TTPoint:
    def __init__(self, x, y, on):
        self.x = x
        self.y = y
        self.on = on

    def point(self):
        return (self.x, self.y)


points = [
    [
        TTPoint(x=61, y=1173, on=0),
        TTPoint(x=330, y=1462, on=0),
        TTPoint(x=545, y=1462, on=1),
        TTPoint(x=876, y=1462, on=0),
        TTPoint(x=999, y=1169, on=1),
        TTPoint(x=1069, y=1003, on=0),
        TTPoint(x=1069, y=732, on=1),
        TTPoint(x=1069, y=469, on=0),
        TTPoint(x=1002, y=293, on=1),
        TTPoint(x=874, y=-42, on=0),
        TTPoint(x=532, y=-42, on=1),
        TTPoint(x=369, y=-42, on=0),
        TTPoint(x=109, y=151, on=0),
        TTPoint(x=90, y=336, on=1),
        TTPoint(x=374, y=336, on=1),
        TTPoint(x=384, y=272, on=0),
        TTPoint(x=472, y=192, on=0),
        TTPoint(x=545, y=192, on=1),
        TTPoint(x=686, y=192, on=0),
        TTPoint(x=743, y=348, on=1),
        TTPoint(x=774, y=434, on=0),
        TTPoint(x=782, y=599, on=1),
        TTPoint(x=743, y=550, on=0),
        TTPoint(x=699, y=524, on=1),
        TTPoint(x=619, y=476, on=0),
        TTPoint(x=502, y=476, on=1),
        TTPoint(x=329, y=476, on=0),
        TTPoint(x=61, y=715, on=0),
        TTPoint(x=61, y=940, on=1),
    ],
    [
        TTPoint(x=623, y=708, on=0),
        TTPoint(x=675, y=742, on=1),
        TTPoint(x=772, y=804, on=0),
        TTPoint(x=772, y=957, on=1),
        TTPoint(x=772, y=1080, on=0),
        TTPoint(x=657, y=1224, on=0),
        TTPoint(x=557, y=1224, on=1),
        TTPoint(x=484, y=1224, on=0),
        TTPoint(x=432, y=1183, on=1),
        TTPoint(x=350, y=1119, on=0),
        TTPoint(x=350, y=966, on=1),
        TTPoint(x=350, y=837, on=0),
        TTPoint(x=455, y=708, on=0),
        TTPoint(x=564, y=708, on=1),
    ],
]


def moon():
    my_guy = MyGuy()
    for contour in points:
        my_guy.plotContour(contour)
    my_guy.ctx.stroke()
    my_guy.finish()


main()
