"""Versioned own-player growth counters from the public strategy accessor.

Only this fixed allowlist enters the actor. Missing counters are unknown, not zero;
a live adapter must reconstruct them from player-visible history or displays.
"""

import numpy as np

VERSION = 1
# Keep order stable: it is part of the checkpoint observation contract.
FEATURE_NAMES = (
    "eternal_knight_deaths", "knight_bonus", "undead_atk", "undead_hp",
    "beetle_atk", "beetle_hp", "volumizer_atk", "volumizer_hp",
    "elem_give_atk", "elem_give_hp", "tavern_elemental_atk", "tavern_elemental_hp",
    "baller_bonus", "lobster_bonus", "magnetized", "spellcraft_cast",
)
N_FEATURES = len(FEATURE_NAMES)


def from_views(views):
    rows = []
    for view in views:
        if view.get("schema_version") != 1:
            raise ValueError("Growth observations require the public strategy schema")
        stats = view.get("me", {}).get("stats", {})
        row = []
        for name in FEATURE_NAMES:
            value = stats.get(name)
            if type(value) is not int or not -(2**31) <= value < 2**31:
                raise ValueError(f"Missing or invalid public growth counter: {name}")
            row.append(value)
        rows.append(row)
    return np.asarray(rows, dtype=np.float32).reshape(-1, N_FEATURES)
