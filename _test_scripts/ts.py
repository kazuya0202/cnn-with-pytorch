def xx(a=False, b=False):
    print(a, b)


if __name__ == "__main__":
    # xx(a=True, b=False)
    x = {
        1: "b",
        2: "c",
    }

    cl = dict([(v, k) for (k, v) in x.items()])
    print(cl)
