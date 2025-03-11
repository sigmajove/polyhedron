def intersect2(a, b, c, d):
    # standard form line eq Line_AB
    a1 = b[1] - a[1]
    b1 = a[0] - b[0]
    c1 = a1 * a[0] + b1 * a[1]

    # standard form line eq Line_CD
    a2 = d[1] - c[1]
    b2 = c[0] - d[0]
    c2 = a2 * c[0] + b2 * c[1]

    determinant = a1 * b2 - a2 * b1

    if determinant == 0:
        return math.inf, math, inf
    else:
        x = (b2 * c1 - b1 * c2) / determinant
        y = (a1 * c2 - a2 * c1) / determinant
        return x, y

def main():
    a = (500.0, 232.0)
    b = (627.0, 232.0)
    c = 
    pass

if __name__ == "__main__":
    main()

