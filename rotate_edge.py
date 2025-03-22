import calc
import math


def distance(p, q):
    assert len(p) == len(q)
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p, q)))


def is_convex_quad(a, b, c, d):
    prev2 = c
    prev = d
    sign = None
    for p in [a, b, c, d]:
        dx0 = prev[0] - prev2[0]
        dy0 = prev[1] - prev2[1]
        dx1 = p[0] - prev[0]
        dy1 = p[1] - prev[1]

        cross = dx0 * dy1 - dy0 * dx1
        if abs(cross) < 1e-4:
            return False
        s = -1 if cross < 0 else 1
        if sign is None:
            sign = s
        else:
            if sign != s:
                return False
        prev2 = prev
        prev = p

    return True


# Test if the triangle with vertices (p0, p1, p2) is skinny.
# If so, returns the vertex (0, 1, 2) opposite the long side.
# If not, returns -1.
def is_skinny(p0, p1, p2):
    d01 = distance(p0, p1)
    d12 = distance(p1, p2)
    d20 = distance(p2, p0)

    if d01 >= d12 and d01 >= d20:
        mx = d01
        result = 2
    elif d12 >= d01 and d12 >= d20:
        mx = d12
        result = 0
    else:
        mx = d20
        result = 1

    sum = d01 + d12 + d20 - mx
    return result if sum < mx * 1.07 else -1


def repair_if_skinny(float_v, triangles, neighbors, t):
    def get_neighbor(t, opposite):
        for i, v in enumerate(triangles[t]):
            if v == opposite:
                return neighbors[t][i]
        raise RuntimeError("Can't find vertex")

    def replace_neighbor(t, find, replace, trace=False):
        if trace:
            print(f"In triangle {t} find {find} replace with {replace}")
        if t < 0:
            return
        n = neighbors[t]
        if trace:
            print(f"Before {n}")
        found = False
        for i, x in enumerate(n):
            if x == find:
                if found:
                    raise RuntimeError("Ambiguous replacemenet")
                n[i] = replace
                if trace:
                    print(f"AfterAfter {n}")
                found = True
                return
        raise RuntimeError("Can't find neighbor")

    def replace_vertex(t, find, replace):
        if t < 0:
            return
        tri = triangles[t]
        for i, x in enumerate(tri):
            if x == find:
                tri[i] = replace
                return
        raise RuntimeError("Can't find vertex")

    i0, i1, i2 = triangles[t]

    p0 = float_v[i0]
    p1 = float_v[i1]
    p2 = float_v[i2]

    t0 = is_skinny(p0, p1, p2)
    if t0 < 0:
        # Not skinny
        return

    # t0, t1, t2 are a permutation of 0, 1, 2.
    t1, t2 = (1, 2) if t0 == 0 else (0, 2) if t0 == 1 else (0, 1)

    # The point indices of the vertices of t

    ins = 0.3

    # s is the adjacent triangle
    s = neighbors[t][t0]
    if s < 0:
        # Can't find adjacent triangle
        return

    i3 = None

    i0 = triangles[t][t0]
    i1 = triangles[t][t1]
    i2 = triangles[t][t2]
    # Find vertex of s that is not shared by t.
    for p in triangles[s]:
        if p != i1 and p != i2:
            if i3 is not None:
                raise RuntimeError("too many i3s")
            i3 = p
    if i3 is None:
        raise RuntimeError("too many i3s")
    # i0, i1, i3, i2 are the indices of the quatrilateral.
    # i3, i2 is not a typo
    p0 = float_v[i0]
    p1 = float_v[i1]
    p2 = float_v[i2]
    p3 = float_v[i3]

    if not is_convex_quad(p0, p1, p3, p2):
        # Cannot convert non-convex quadrilateral
        return

    if is_skinny(p0, p1, p3) >= 0:
        return

    if is_skinny(p2, p0, p3) >= 0:
        return

    # Perform the conversion

    # Get all the adjacent triangles.
    w = get_neighbor(t, i2)
    x = get_neighbor(t, i1)
    y = get_neighbor(s, i2)
    z = get_neighbor(s, i1)

    # Update all the neighbors
    trace = False

    xxx = neighbors[t].tolist()
    print (s, xxx)
    s_slot = xxx.index(s)
    x_slot = xxx.index(x)
    neighbors[t][x_slot] = s
    neighbors[t][s_slot] = y

    yyy = neighbors[s].tolist()
    t_slot = yyy.index(t)
    y_slot = yyy.index(y)
    neighbors[s][t_slot] = x
    neighbors[s][y_slot] = t

    replace_neighbor(x, t, s, trace)
    replace_neighbor(y, s, t, trace)

    # Update the vertices
    replace_vertex(t, i2, i3)
    replace_vertex(s, i1, i0)

    calc.check_neighbors(triangles, neighbors)


def check_edge(float_v, triangles, neighbors, t):
    i0, i1, i2 = triangles[t]

    p0 = float_v[i0]
    p1 = float_v[i1]
    p2 = float_v[i2]

    t0 = is_skinny(p0, p1, p2)
    if t0 < 0:
        return None

    # t0, t1, t2 are a permutation of 0, 1, 2.
    t1, t2 = (1, 2) if t0 == 0 else (0, 2) if t0 == 1 else (0, 1)

    # The point indices of the vertices of t

    ins = 0.3

    # s is the adjacent triangle
    s = neighbors[t][t0]
    if s < 0:
        return (255, 128, 0, ins)

    i3 = None

    i0 = triangles[t][t0]
    i1 = triangles[t][t1]
    i2 = triangles[t][t2]
    # Find vertex of s that is not shared by t.
    for p in triangles[s]:
        if p != i1 and p != i2:
            if i3 is not None:
                raise RuntimeError("too many i3s")
            i3 = p
    if i3 is None:
        raise RuntimeError("too many i3s")
    # i0, i1, i3, i2 are the indices of the quatrilateral.
    # i3, i2 is not a typo
    p0 = float_v[i0]
    p1 = float_v[i1]
    p2 = float_v[i2]
    p3 = float_v[i3]

    if not is_convex_quad(p0, p1, p3, p2):
        # Cannot convert non-convex quadrilateral
        return None
        return (255, 0, 255, ins)

    if is_skinny(p0, p1, p3) >= 0:
        return None
        return (0, 255, 255, ins)

    if is_skinny(p2, p0, p3) >= 0:
        return None
        return (0, 255, 255, ins)

    # Can convert
    print("Can convert")
    return (255, 0, 0, ins)
