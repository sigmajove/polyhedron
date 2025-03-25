import math


def distance(p, q):
    assert len(p) == len(q)
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p, q)))


# Tests if a, b, c, d are the consecutive vertices of a convex quadrilateral
def is_convex_quad(a, b, c, d):
    prev2 = c
    prev = d
    sign = None
    for p in [a, b, c, d]:
        dx0 = prev[0] - prev2[0]
        dy0 = prev[1] - prev2[1]
        dx1 = p[0] - prev[0]
        dy1 = p[1] - prev[1]

        # Compute the cross product of two adjacent sides.
        cross = dx0 * dy1 - dy0 * dx1

        if abs(cross) < 1e-4:
            # if the angle is very small, don't count the quadrilateral
            # as convex.
            return False

        # The quadrilateal is convex if all the cross products
        # have the same sign. The sign determines whether a, b, c, d
        # are clockwise or counterclockwise.
        s = -1 if cross < 0 else 1
        if sign is None:
            sign = s
        else:
            if sign != s:
                return False

        prev2 = prev
        prev = p
    return True


# Tests if the triangle with vertices (p0, p1, p2) is skinny.
# If so, returns the vertex (0, 1, 2) opposite the long side.
# If not, returns -1.
# If the triangle is isosceles there may be one long side. We
# return one of them.

# THRESHOLD defines a skinny triangle as one with angle 135 degrees or more.
# Note cos(135 deg) = -1/sqrt(2)
THRESHOLD = 2 / math.sqrt(2.0)


def is_skinny(p0, p1, p2):
    d01 = sum((a - b) ** 2 for a, b in zip(p0, p1))
    d12 = sum((a - b) ** 2 for a, b in zip(p1, p2))
    d20 = sum((a - b) ** 2 for a, b in zip(p2, p0))

    if d01 >= d12 and d01 >= d20:
        result = 2
        skinny = d01 - d12 - d20 >= THRESHOLD * math.sqrt(d12 * d20)
    elif d12 >= d01 and d12 >= d20:
        result = 0
        skinny = d12 - d01 - d20 >= THRESHOLD * math.sqrt(d01 * d20)
    else:
        result = 1
        skinny = d20 - d01 - d12 >= THRESHOLD * math.sqrt(d01 * d12)

    return result if skinny else -1


class Fixer:
    def __init__(self):
        self.border_triangle = 0
        self.concave_quad = 0
        self.skinny_quad = 0
        self.repaired = 0

    def print_statistics(self):
        # Hack. Counters are doubled because triangulate is called twice
        # for each mesh
        if self.border_triangle > 0:
            print(f"{self.border_triangle//2} not repaired on border")
        if self.concave_quad > 0:
            print(
                f"{self.concave_quad//2} not repaired in concave quadrilaterals"
            )
        if self.skinny_quad > 0:
            print(
                f"{self.skinny_quad//2} not repaired in skinny quadrilaterals"
            )
        if self.repaired > 0:
            print(f"{self.repaired//2} triangles repaired")

    # Attempts to eliminate skinny triangles.
    # If necesary and possible, updates triangles and neighbors so that
    # triangles[t] is not skinny.
    def repair_if_skinny(self, points, triangles, neighbors, t):
        tri_t = triangles[t]

        i0, i1, i2 = tri_t

        p0 = points[i0]
        p1 = points[i1]
        p2 = points[i2]

        t0 = is_skinny(p0, p1, p2)
        if t0 < 0:
            # Not skinny
            return

        # t0, t1, t2 are a cyclic permutation of 0, 1, 2.
        t1 = (t0 + 1) % 3
        t2 = (t1 + 1) % 3

        # We attempt to flip the longest edge of the skinny triangle.
        # This will often eliminate a skinny triangle. If flipping isn't
        # possible for some reason, I suppose we could try to other two,
        # but that feels like too much work.

        # s is the adjacent triangle along the longest edge of t
        nbr_t = neighbors[t]
        s = nbr_t[t0]
        if s < 0:
            # There is no adjacent triangle. No repair possible.
            self.border_triangle += 1
            return

        tri_s = triangles[s]
        nbr_s = neighbors[s]

        # i0, i1, and i2 are the point indices of triangle t.
        i0 = tri_t[t0]
        i1 = tri_t[t1]
        i2 = tri_t[t2]

        # Find vertex of s that is not shared by t.
        i3 = None
        for i, p in enumerate(tri_s):
            if p != i1 and p != i2:
                if i3 is not None:
                    raise RuntimeError("too many i3s")
                i3 = p
                s0 = i
        if i3 is None:
            raise RuntimeError("too many i3s")

        # s0, s1, s2 is a cyclic permuation of 0, 1, 2.
        s1 = (s0 + 1) % 3
        s2 = (s1 + 1) % 3

        # i0, i1, i3, i2 are the indices of the quatrilateral. See below.
        #
        #         s2
        #         i1
        #        /|\
        #    w  / | \  y
        #      /  |  \
        # i0  / t | s \ i3
        # s2  \   |   / s0
        #      \  |  /
        #    x  \ | /  z
        #        \|/
        #         i2
        #         s1

        p0 = points[i0]
        p1 = points[i1]
        p2 = points[i2]
        p3 = points[i3]

        if not is_convex_quad(p0, p1, p3, p2):  # p3, p2 is not a typo
            # Cannot convert non-convex quadrilateral
            self.concave_quad += 1
            return

        # We want to flip the vertical diagnonal to the horizontal
        #         s2
        #         i1
        #        / \
        #    w  /   \  y
        #      /  t  \
        # i0  /_______\ i3
        # s2  \       / s0
        #      \  s  /
        #    x  \   /  z
        #        \ /
        #         i2
        #         s1

        # If either of the two new triangles are skinny,
        # then give up.
        if is_skinny(p0, p1, p3) >= 0 or is_skinny(p2, p0, p3) >= 0:
            self.skinny_quad += 1
            return

        # Perform the diagonal flip.
        self.repaired += 1

        # Get all the adjacent triangles.
        # Refer to the first of the two diagrams.
        w = nbr_t[t2]
        x = nbr_t[t1]
        y = nbr_s[s1]
        z = nbr_s[s2]

        # Update all the neighbors
        nbr_t[t0] = y
        nbr_t[t1] = s

        nbr_s[s0] = x
        nbr_s[s1] = t

        if x >= 0:
            neighbor_x = neighbors[x]
            neighbor_x[neighbor_x.index(t)] = s

        if y >= 0:
            neighbor_y = neighbors[y]
            neighbor_y[neighbor_y.index(s)] = t

        # Update the vertices
        tri_t[t2] = i3
        tri_s[s2] = i0

    def remove_very_obtuse_triangles(self, points, triangles, neighbors):
        for i in range(len(triangles)):
            self.repair_if_skinny(points, triangles, neighbors, i)

        # Make sure we didn't corrupt the neighbors.
        check_neighbors(triangles, neighbors)


# Returns a sorted tuple
def normalize(a, b):
    return (a, b) if a <= b else (b, a)


# Rebuilds the neighbor matrix from triangles.
# This function is pretty fast. Perhaps we didn't really
# need to incrementally update the neighbors above.
def compute_neighbors(triangles):
    edges = dict()

    def add_edge(t, e0, e1):
        key = normalize(e0, e1)
        s = edges.get(key, None)
        if s is None:
            s = []
            edges[key] = s
        s.append(t)

    for i, t in enumerate(triangles):
        add_edge(i, t[0], t[1])
        add_edge(i, t[1], t[2])
        add_edge(i, t[2], t[0])

    neighbors = [[-1, -1, -1] for _ in range(len(triangles))]

    def set_neighbor(t0, v, other):
        if neighbors[t0][v] >= 0:
            raise RuntimeError("neighbors already set")
        neighbors[t0][v] = other

    def other_vertex(t, e0, e1):
        result = None
        for i, v in enumerate(triangles[t]):
            if v != e0 and v != e1:
                if result is not None:
                    raise RuntimeError("too many vertices")
                result = i
        if result is None:
            raise RuntimeError("vertex not found")
        return result

    for key, ts in edges.items():
        if len(ts) >= 3:
            raise RuntimeError("too many triangles")
        if len(ts) == 2:
            t1, t2 = ts
            set_neighbor(t1, other_vertex(t1, *key), t2)
            set_neighbor(t2, other_vertex(t2, *key), t1)

    return neighbors


# Checks the consistency of triangles and neighbors.
def check_neighbors(triangles, neighbors):
    adj = compute_neighbors(triangles)
    assert len(adj) == len(neighbors)
    for i in range(len(adj)):
        n = neighbors[i]
        a = adj[i]
        assert len(a) == len(n) == 3
        if not all(a[i] == n[i] for i in range(3)):
            print("Fail", i, "good", a, "bad", n)
            # raise RuntimeError("bad neighbors")
