import json
import math
import random


# Return a random number in the range (0, 1)
def rand01():
    while True:
        result = random.random()
        if result > 0 and result < 1:
            return result


# Returns spherical coordinates (theta, phi) of a random point
# on the unit sphere.
# See https://mathworld.wolfram.com/SpherePointPicking.html
def rand_point():
    return (2 * math.pi * rand01(), math.acos(2 * rand01() - 1))


def xyz(theta, phi):
    sin_phi = math.sin(phi)
    return (math.cos(theta) * sin_phi, math.sin(theta) * sin_phi, math.cos(phi))


def rand_poly():
    result = []
    for _ in range(9):
        theta, phi = rand_point()
        cart = xyz(theta, phi)
        result.append(cart)
        result.append(tuple(-p for p in cart))
    return result


# dot product of two vectors
def dot_product(v0, v1):
    length = len(v0)
    assert length == len(v1)
    return sum(v0[i] * v1[i] for i in range(length))


# Returns the smallest spherical distance between two points
def min_sep(vectors):
    # See https://mathworld.wolfram.com/SphericalDistance.html
    result = None
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            d = dot_product(vectors[i], vectors[j])
            if d < -1:
                d = -1
            elif d > 1:
                d = 1
            ac = math.acos(d)
            if result is None or ac < result:
                result = ac
    return result


def trials():
    result = rand_poly()
    sep = min_sep(result)
    for _ in range(100000000):
        x = rand_poly()
        s = min_sep(x)
        if s > sep:
            result = x
            sep = s
    return result


if __name__ == "__main__":
    p = trials()
    print("sep", min_sep(p))
    with open('points18.json', 'w', encoding='utf-8') as file:
        file.write(json.dumps(p))
        file.write("\n")


