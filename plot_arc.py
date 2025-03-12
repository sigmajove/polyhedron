import svg_writer
import math
def main():
    with svg_writer.SVGWriter("an_arc", 25, 0.1) as ctx:
        ctx.arc(150, 150, 150, 0, 2*math.pi)
        ctx.stroke()
main()

