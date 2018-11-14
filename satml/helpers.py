

def fixpoint(arg, func, comp):
    """Applies `func` to `arg` repeatedly until `comp` returns YES."""
    prev = arg
    cur = func(arg)

    while not comp(cur, prev):
        prev = cur
        cur = func(cur)

    return cur
