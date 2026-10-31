"""Named groups of frozen policies shared by sampling and evaluation drivers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class OpponentGroup:
    name: str
    policy: object
    count: int


@dataclass(frozen=True)
class FrozenOpponentPool:
    groups: tuple[OpponentGroup, ...]

    def __post_init__(self):
        names = [group.name for group in self.groups]
        if not names or len(set(names)) != len(names) or "learner" in names:
            raise ValueError("Frozen opponent names must be unique and exclude learner")
        if any(not name or not name.replace("_", "").isalnum() for name in names):
            raise ValueError("Opponent names may contain only letters, digits and underscores")
        if any(type(group.count) is not int or group.count < 1 for group in self.groups):
            raise ValueError("Opponent group counts must be positive integers")
        if not 1 <= self.seats <= 7:
            raise ValueError("A frozen opponent pool must occupy 1..7 seats")

    @property
    def seats(self):
        return sum(group.count for group in self.groups)

    @property
    def forward_precision(self):
        modes = {group.policy.forward_precision for group in self.groups}
        if len(modes) != 1:
            raise ValueError("Frozen opponent groups must use one configured precision")
        return next(iter(modes))

    def set_forward_precision(self, precision):
        for group in self.groups:
            group.policy.set_forward_precision(precision)


def opponent_groups(policy, count):
    if not 1 <= count <= 7:
        raise ValueError("frozen_opponents must be in 1..7")
    if isinstance(policy, FrozenOpponentPool):
        if policy.seats != count:
            raise ValueError("frozen_opponents differs from the named pool's seat count")
        return policy.groups
    return (OpponentGroup("bot_v1", policy, count),)
