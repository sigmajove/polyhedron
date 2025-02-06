import math

# Given polygon defined by a list of (x, y) vertices in order
# (clockwise or counterclockwise), and a point (x, y), determines whether
# that point is in the interior of the polygon.
# Uses the algorithm described in
# https://en.wikipedia.org/wiki/Point_in_polygon#Ray_casting_algorithm
# The basic idea is to extend a ray infinitely to the right point, and
# count how many times it crosses the boundary. If it crosses an odd number
# of times, the point is on the interior, and if it crosses an even number
# of times the point is on the exterior.

# The are, however, some corner cases to be considered. For example, if the
# point is anywhere on the boundary, we abort the count and return False.
# See also https://www.scribd.com/document/56214673/Ray-Casting-Algorithm


def inside(point, sides):
    if len(sides) <= 2:
        # At best, a degenerate polygon that can contain no points.
        return False
    prev = sides[-1]
    cross_count = 0
    for here in sides:
        # (Almost) horizontal edges are ignored, since there is no computable
        # point that intersects with the horizontal ray. We declare the ray
        # cannot cross these edges.
        if math.isclose(here[1], prev[1]):
            if math.isclose(point[1], 0.5 * (here[1] + prev[1])) and (
                here[0] <= point[0] <= prev[0] or here[0] >= point[0] >= prev[0]
            ):
                # The point is close to the (almost) horizontal edge
                return False
            keep = False
        # If the ray intersects a vertex, we have to worry about the
        # possibility of counting the crossing twice as we process the
        # two edges that share that vertex.
        # The trick is to only count as a crossing the intersection of the
        # lower vertex. If the ray crosses a peak that is zero crossings.
        # If the ray crosses a valley, that is two crossings, but if ray
        # crosses an incline or decline, that will be correctly counted
        # as a single crossing.
        elif here[1] < prev[1]:
            keep = here[1] <= point[1] < prev[1]
        else:
            keep = here[1] > point[1] >= prev[1]

        if keep:
            # The x-coordinate where the ray intersects the edge.
            x = here[0] + (point[1] - here[1]) * (here[0] - prev[0]) / (
                here[1] - prev[1]
            )

            if math.isclose(point[0], x):
                # The point is very near the edge.
                return False

            if point[0] < x:
                # The ray begins to the left of the edge, so it intersects.
                cross_count += 1

        prev = here

    return cross_count % 2 == 1
