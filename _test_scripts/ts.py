
def xx(a=False, b=False):
    print(a, b)


if __name__ == "__main__":
    # xx(a=True, b=False)
    s = True if False else False
    xx(*(s * 2))
