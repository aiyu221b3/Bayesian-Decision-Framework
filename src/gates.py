from enum import Enum
from operator import index


def _id(y):
    if isinstance(y, (set, frozenset)):
        return frozenset(y)
    return int(y)


def _sup(y):
    if isinstance(y, frozenset):
        return y
    return frozenset([int(y)])


class GateState(Enum):
    ARMED = "ARMED"
    DISARMED = "DISARMED"


ARMED = GateState.ARMED
DISARMED = GateState.DISARMED


class Cross:
    """upward threshold cross detector"""

    def __init__(self, up, lo):
        self.up = float(up)
        self.lo = float(lo)
        if not 0 <= self.lo < self.up <= 1:
            raise ValueError("need 0 <= lower < upper <= 1")
        self.reset()

    def reset(self):
        self.arm = True

    def step(self, score):
        score = float(score)
        if not 0 <= score <= 1:
            raise ValueError("score must lie in [0, 1]")
        if self.arm:
            if score >= self.up:
                self.arm = False
                return True
        elif score <= self.lo:
            self.arm = True
        return False


class EmitterGate:
    """hysteretic emitter. refr for debounce"""

    def __init__(self, up=0.8, lo=0.7, ref=8):
        upper = float(up)
        lower = float(lo)
        if not 0 <= lower < upper <= 1:
            raise ValueError("need 0 <= lower < upper <= 1")
        try:
            validated_ref = index(ref)
        except TypeError as exc:
            raise ValueError("refr must be a nonnegative integer") from exc
        if isinstance(ref, bool) or validated_ref < 0:
            raise ValueError("refr must be a nonnegative integer")
        self.upper = upper
        self.lower = lower
        self.refr = validated_ref
        self.reset()

    def reset(self):
        self.state = ARMED
        self.last_output = None
        self.last_support = None
        self.last_emit_time = None

    @property
    def up(self):
        return self.upper

    @property
    def lo(self):
        return self.lower

    @property
    def ref(self):
        return self.refr

    @property
    def arm(self):
        return self.state is ARMED

    @property
    def last(self):
        return self.last_support

    @property
    def t0(self):
        return self.last_emit_time

    def step(self, t, y, s, ls=None, bd=False):
        s = float(s)
        if not 0 <= s <= 1 + 1e-9:
            raise ValueError("score must lie in [0, 1]")
        if ls is not None:
            ls = float(ls)
        if y is None:
            if self.state is DISARMED:
                if bd or (ls is not None and ls <= self.lower):
                    self.state = ARMED
            return None
        ident = _id(y)
        support = _sup(ident)
        if self.state is DISARMED:
            release = (
                bd
                or (ls is not None and ls <= self.lower)
                or (
                    self.last_support is not None
                    and support.isdisjoint(self.last_support)
                    and s >= self.upper
                )
            )
            if release:
                self.state = ARMED
        if self.state is DISARMED or s < self.upper:
            return None
        # debounce suppress without re-arm
        if self.last_emit_time is not None and t - self.last_emit_time <= self.refr:
            return None
        self.state = DISARMED
        self.last_output = ident
        self.last_support = support
        self.last_emit_time = int(t)
        return ident


# compat
Gate = EmitterGate
