from enum import Enum


class BackgroundType(Enum):
    BRECHADO = 0
    PERLIN = 1
    FLAT = 2


class PatternType(Enum):
    SIMPLE = 0
    IMPULSE = 1
    SPOTS = 2
    GRANULE = 3
    BRECHADO = 4
    INVERSEBRECHADO = 5
