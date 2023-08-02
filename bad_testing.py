def load_ini(fp):
    settings = {}
    for line in open(fp, 'r').readlines():
        if not line.strip() or line.strip()[0] == '#':
            continue

        try:
            k, v = map(str.strip, line.split('='))
        except:
            continue

        if not v:
            continue
        for char in v:
            if char not in '1234567890.,':
                settings[k] = v
                break
        if settings.get(k) is None:
            settings[k] = eval(v)
    return settings