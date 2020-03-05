def xx(**options):
    a = options.pop('a', False)
    b = options.pop('b', False)

    save_config = {
        'x': 1,
        'y': 2,
    }
    if a:
        save_config['optimizer'] = 3

    if b:
        save_config['epoch'] = 4

    print(save_config)


if __name__ == "__main__":
    xx(a=True, b=False)
