from enum import Enum


class Type(str, Enum):
    NORMAL = "normal"
    FIGHTING = "fighting"
    FLYING = "flying"
    POISON = "poison"
    GROUND = "ground"
    ROCK = "rock"
    BUG = "bug"
    GHOST = "ghost"
    STEEL = "steel"
    FIRE = "fire"
    WATER = "water"
    GRASS = "grass"
    ELECTRIC = "electric"
    PSYCHIC = "psychic"
    ICE = "ice"
    DRAGON = "dragon"
    DARK = "dark"


TYPES = list(Type)

_MATRIX = [
    # Nor  Fig  Fly  Poi  Gro  Roc  Bug  Gho  Ste  Fir  Wat  Gra  Ele  Psy  Ice  Dra  Dar
    [1,   1,   1,   1,   1,   .5,  1,   0,   .5,  1,   1,   1,   1,   1,   1,   1,   1],   # Normal
    [2,   1,   .5,  .5,  1,   2,   .5,  0,   2,   1,   1,   1,   1,   .5,  2,   1,   2],   # Fighting
    [1,   2,   1,   1,   1,   .5,  2,   1,   .5,  1,   1,   2,   .5,  1,   1,   1,   1],   # Flying
    [1,   1,   1,   .5,  .5,  .5,  1,   .5,  0,   1,   1,   2,   1,   1,   1,   1,   1],   # Poison
    [1,   1,   0,   2,   1,   2,   .5,  1,   2,   2,   1,   .5,  2,   1,   1,   1,   1],   # Ground
    [1,   .5,  2,   1,   .5,  1,   2,   1,   .5,  2,   1,   1,   1,   1,   2,   1,   1],   # Rock
    [1,   .5,  .5,  .5,  1,   1,   1,   .5,  .5,  .5,  1,   2,   1,   2,   1,   1,   2],   # Bug
    [0,   1,   1,   1,   1,   1,   1,   2,   .5,  1,   1,   1,   1,   2,   1,   1,   .5],  # Ghost
    [1,   1,   1,   1,   1,   2,   1,   1,   .5,  .5,  .5,  1,   .5,  1,   2,   1,   1],   # Steel
    [1,   1,   1,   1,   1,   .5,  2,   1,   2,   .5,  .5,  2,   1,   1,   2,   .5,  1],   # Fire
    [1,   1,   1,   1,   2,   2,   1,   1,   1,   2,   .5,  .5,  1,   1,   1,   .5,  1],   # Water
    [1,   1,   .5,  .5,  2,   2,   .5,  1,   .5,  .5,  2,   .5,  1,   1,   1,   .5,  1],   # Grass
    [1,   1,   2,   1,   0,   1,   1,   1,   1,   1,   2,   .5,  .5,  1,   1,   .5,  1],   # Electric
    [1,   2,   1,   2,   1,   1,   1,   1,   .5,  1,   1,   1,   1,   .5,  1,   1,   0],   # Psychic
    [1,   1,   2,   1,   2,   1,   1,   1,   .5,  .5,  .5,  2,   1,   1,   .5,  2,   1],   # Ice
    [1,   1,   1,   1,   1,   1,   1,   1,   .5,  1,   1,   1,   1,   1,   1,   2,   1],   # Dragon
    [1,   .5,  1,   1,   1,   1,   1,   2,   .5,  1,   1,   1,   1,   2,   1,   1,   .5],  # Dark
]

# TYPE_CHART[attaccante][difensore] -> moltiplicatore (0, 0.5, 1, 2)
TYPE_CHART: dict[Type, dict[Type, float]] = {
    atk: {dfn: _MATRIX[i][j] for j, dfn in enumerate(TYPES)}
    for i, atk in enumerate(TYPES)
}


def effectiveness(attacker: Type, *defenders: Type) -> float:
    mult = 1.0
    for d in defenders:
        mult *= TYPE_CHART[attacker][d]
    return mult





if __name__ == "__main__":
    print(effectiveness(Type.ground, Type.electric))            # 2.0

