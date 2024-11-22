import random


def random_float_in_range(range):
    """Generates a random float in the given range."""
    number = random.random()
    number *= range[1] - range[0]
    number += range[0]

    return number
