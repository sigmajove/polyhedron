import cairo


# Context manager for creating an svg file.
# Parameters are filename relative to User/Sigma/Documents, without suffix,
# scale factor to set the size of the drawing, and the width of lines
# in the drawing. Line color initialized to opaque black.
# Returns a Cairo context on which to draw.
class SVGWriter:
    def __init__(self, filename, scale, line_width):
        self.filename = filename
        self.rs = cairo.RecordingSurface(cairo.CONTENT_COLOR_ALPHA, None)
        self.ctx = cairo.Context(self.rs)
        self.ctx.scale(scale, -scale)
        self.ctx.set_line_width(line_width)
        self.ctx.set_source_rgba(0, 0, 0, 1)

    def __enter__(self):
        return self.ctx

    def __exit__(self, exc_type, exc_value, traceback):
        x, y, width, height = self.rs.ink_extents()
        surface = cairo.SVGSurface(
            f"C:/Users/sigma/Documents/{self.filename}.svg", width, height
        )
        context = cairo.Context(surface)
        context.set_source_surface(self.rs, -x, -y)
        context.paint()
        surface.flush()
        surface.finish()
        del context
        del self.ctx
        self.rs.finish()


# Example usage
def example():
    with SVGWriter("test", 10, 0.5) as ctx:
        ctx.move_to(0, 0)
        ctx.line_to(0, 10)
        ctx.line_to(10, 10)
        ctx.line_to(10, 0)
        ctx.close_path()
        ctx.stroke()
