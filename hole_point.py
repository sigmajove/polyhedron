import font
import math
import numpy as np
import svg_writer
from scipy.interpolate import BSpline

STEP = 30


def distance(p, q):
    return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)


def get_angle(v0, v1, v2):
    ax = v1[0] - v0[0]
    ay = v1[1] - v0[1]
    bx = v2[0] - v1[0]
    by = v2[1] - v1[1]
    return (ax * by - ay * bx) / (
        math.sqrt(ax * ax + ay * ay) * math.sqrt(bx * bx + by * by)
    )


def vec_div(a, d):
    return (a[0] / d, a[1] / d)


def vec_diff(a, b):
    return (a[0] - b[0], a[1] - b[1])


def vec_add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def magnitude(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1])


class PointPen:
    def __init__(self, ctx):
        self.ctx = ctx
        self.current_hole_ = []
        self.holes = []

    def adjust(self, p):
        return p

    def move_to(self, p0):
        self.current_point = self.adjust(p0)
        self.start_path = self.current_point
        self.current_hole_.append(self.current_point)

    def line_to(self, p0):
        v1 = self.adjust(p0)
        self.current_point = v1
        self.current_hole_.append(v1)

    def curve_to(self, on, off):
        control_points = np.array(
            [self.current_point, self.adjust(off), self.adjust(on)]
        )
        k = 2  # quadratic
        n = len(control_points)
        t = np.concatenate(
            [np.zeros(k), np.linspace(0, 1, n - k + 1), np.ones(k)]
        )
        spl = BSpline(t, control_points, k)
        u = np.linspace(0, 1, 8)
        length = 0
        prev_x = self.current_point
        for x in spl(u)[1:]:
            length += distance(x, prev_x)
            prev_x = x
        z = math.ceil(length / STEP)
        u = np.linspace(0, 1, num=z)
        for x in spl(u)[1:]:
            self.current_hole_.append(x)

        self.current_point = self.adjust(on)

    def cubic(self, a, b, c):
        control_points = np.array(
            [self.current_point, self.adjust(a), self.adjust(b), self.adjust(c)]
        )
        k = 3  # quadratic
        n = len(control_points)
        t = np.concatenate(
            [np.zeros(k), np.linspace(0, 1, n - k + 1), np.ones(k)]
        )
        spl = BSpline(t, control_points, k)
        u = np.linspace(0, 1, 8)
        length = 0
        prev_x = self.current_point
        for x in spl(u)[1:]:
            length += distance(x, prev_x)
            prev_x = x
        z = math.ceil(length / STEP)
        u = np.linspace(0, 1, num=z)
        for x in spl(u)[1:]:
            self.current_hole_.append(x)

        self.current_point = self.adjust(c)

    def close_path(self, hole=False, inner=None):
        if self.current_hole_:
            tail = self.current_hole_[-1]
            if tail[0] != self.start_path[0] or tail[1] != self.start_path[1]:
                self.current_hole_.append(self.start_path)
            self.ctx.move_to(*self.current_hole_[0])
            for h in self.current_hole_[1:]:
                self.ctx.line_to(*h)
            self.ctx.stroke()
            counter = 0
            x = None
            y = None
            if hole:
                print(f"There are {len(self.current_hole_)} segments")
                for i in range(len(self.current_hole_) - 1, 1, -1):
                    if (
                        get_angle(
                            self.current_hole_[i],
                            self.current_hole_[i - 1],
                            self.current_hole_[i - 2],
                        )
                        < 0
                    ):
                        a = vec_diff(
                            self.current_hole_[i], self.current_hole_[i - 1]
                        )
                        b = vec_diff(
                            self.current_hole_[i - 2],
                            self.current_hole_[i - 1],
                        )
                        bisect = vec_add(
                            vec_div(a, magnitude(a)),
                            vec_div(b, magnitude(b)),
                        )
                        bisect = vec_div(bisect, magnitude(bisect))

                        x = self.current_hole_[i - 1][0] + 10 * bisect[0]
                        y = self.current_hole_[i - 1][1] + 10 * bisect[1]
                        break
            else:
                for i in range(2, len(self.current_hole_)):
                    if (
                        get_angle(
                            self.current_hole_[i - 2],
                            self.current_hole_[i - 1],
                            self.current_hole_[i],
                        )
                        < 0
                    ):
                        a = vec_diff(
                            self.current_hole_[i - 2], self.current_hole_[i - 1]
                        )
                        b = vec_diff(
                            self.current_hole_[i], self.current_hole_[i - 1]
                        )
                        bisect = vec_add(
                            vec_div(a, magnitude(a)),
                            vec_div(b, magnitude(b)),
                        )
                        bisect = vec_div(bisect, magnitude(bisect))

                        x = self.current_hole_[i - 1][0] + 10 * bisect[0]
                        y = self.current_hole_[i - 1][1] + 10 * bisect[1]
                        break
            if x is not None and y is not None:
                x = round(x, 2)
                y = round(y, 2)
                print(f"inner = ({x}, {y})")
                self.ctx.set_source_rgb(1, 0, 0)
                self.ctx.arc(x, y, 5, 0, 2 * math.pi)
                self.ctx.fill()

        self.current_hole_ = []


def main():
    with svg_writer.SVGWriter("hole_point", 25, 0.5) as ctx:
        f = font.Font(PointPen(ctx))
        f.dot()


main()
