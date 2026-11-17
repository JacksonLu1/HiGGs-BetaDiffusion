def require(condition, message):
    if not condition:
        raise RuntimeError(message)
