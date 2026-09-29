"""Reference models ``x_ref`` for a 32-value block. Every model keeps O(1) state.

A reference turns a block ``x`` into the residual ``x - x_ref`` that is scored. All models are
causal at block granularity: ``x_ref`` for a block depends only on earlier valid blocks and, for
``detrend`` and ``diff``, on the block itself (the block is scored as a whole).

``median``   median of the last ``ref_blocks * 32`` valid samples (ring buffer). Constant
             across the block. This is the oes512q-latch benchmark reference and was the
             default in 0.1.0.dev0 (as ``"trailing_median"``, still accepted as an alias).
``ewma``     exponentially weighted level of all earlier valid samples,
             ``level <- level + alpha * (x_j - level)`` per sample. Constant across the block.
``detrend``  least-squares line fitted to the block itself; ``x_ref_j`` is the fitted value.
             Needs no history. A whole-block offset or a purely linear block scores 0.
``diff``     first difference: ``x_ref = [last sample of the previous block, x_0, ..., x_30]``,
             so ``r_j = x_j - x_{j-1}``. Reset by an invalid block (see ``on_invalid``).
fixed        a scalar or 32-vector given by the caller.

``ref(x)`` returns the reference or ``None`` if the model has no history yet (the guard then
latches with ``NO_REFERENCE``). ``push(x)`` updates the state with a valid block.
``on_invalid()`` is called when a block fails validation; only ``diff`` reacts (it forgets its
last sample, so the block after a gap latches with ``NO_REFERENCE`` rather than scoring a
difference across the gap).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from streamguard.score import BLOCK

MODES = ("median", "ewma", "detrend", "diff")
DEFAULT_REFERENCE = "diff"  # chosen from the SYNTHETIC evaluation, see README
ALIASES = {"trailing_median": "median"}

_T = np.arange(BLOCK, dtype=np.float64) - (BLOCK - 1) / 2.0  # centred time index
_T_SS = float(_T @ _T)


class MedianRef:
    __slots__ = ("buf", "count", "pos")
    name = "median"

    def __init__(self, ref_blocks: int) -> None:
        self.buf = np.empty(ref_blocks * BLOCK, dtype=np.float64)
        self.count = 0
        self.pos = 0

    def ref(self, x: np.ndarray) -> np.ndarray | None:
        if self.count == 0:
            return None
        return np.asarray(np.median(self.buf[: self.count]))

    def push(self, x: np.ndarray) -> None:
        self.buf[self.pos : self.pos + BLOCK] = x
        self.pos = (self.pos + BLOCK) % self.buf.size
        self.count = min(self.count + BLOCK, self.buf.size)

    def on_invalid(self) -> None:
        pass

    def config(self) -> dict:
        return {"ref_window": int(self.buf.size)}


class EwmaRef:
    __slots__ = ("alpha", "decay", "level", "weights")
    name = "ewma"

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        # level after 32 samples = decay * level + weights @ x  (closed form of the recursion)
        self.decay = (1.0 - alpha) ** BLOCK
        self.weights = alpha * (1.0 - alpha) ** np.arange(BLOCK - 1, -1, -1, dtype=np.float64)
        self.level: float | None = None

    def ref(self, x: np.ndarray) -> np.ndarray | None:
        return None if self.level is None else np.asarray(self.level)

    def push(self, x: np.ndarray) -> None:
        if self.level is None:
            self.level = float(np.median(x))
        else:
            self.level = float(self.decay * self.level + self.weights @ x)

    def on_invalid(self) -> None:
        pass

    def config(self) -> dict:
        return {"ewma_alpha": self.alpha}


class DetrendRef:
    __slots__ = ()
    name = "detrend"

    def ref(self, x: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore", invalid="ignore"):
            slope = float(_T @ x) / _T_SS
            return x.mean() + slope * _T

    def push(self, x: np.ndarray) -> None:
        pass

    def on_invalid(self) -> None:
        pass

    def config(self) -> dict:
        return {}


class DiffRef:
    __slots__ = ("last",)
    name = "diff"

    def __init__(self) -> None:
        self.last: float | None = None

    def ref(self, x: np.ndarray) -> np.ndarray | None:
        if self.last is None:
            return None
        out = np.empty(BLOCK, dtype=np.float64)
        out[0] = self.last
        out[1:] = x[:-1]
        return out

    def push(self, x: np.ndarray) -> None:
        self.last = float(x[-1])

    def on_invalid(self) -> None:
        # The previous sample is no longer adjacent to the next block: forget it, so the next
        # valid block reports NO_REFERENCE instead of a difference that spans the gap.
        self.last = None

    def config(self) -> dict:
        return {}


class FixedRef:
    __slots__ = ("value",)
    name = "fixed"

    def __init__(self, value: ArrayLike) -> None:
        ref = np.asarray(value, dtype=np.float64)
        if ref.shape not in ((), (BLOCK,)) or not np.all(np.isfinite(ref)):
            raise ValueError(f"reference must be finite, scalar or shape ({BLOCK},)")
        self.value = ref.copy()

    def ref(self, x: np.ndarray) -> np.ndarray:
        return self.value

    def push(self, x: np.ndarray) -> None:
        pass

    def on_invalid(self) -> None:
        pass

    def config(self) -> dict:
        return {"reference_value": self.value.tolist()}


def make_reference(
    reference: str | float | ArrayLike, ref_blocks: int = 4, ewma_alpha: float = 0.05
):
    """Build a reference model from a mode name, a scalar or a 32-vector."""
    if isinstance(reference, str):
        mode = ALIASES.get(reference, reference)
        if mode == "median":
            if isinstance(ref_blocks, bool) or not isinstance(ref_blocks, int) or ref_blocks < 1:
                raise ValueError("ref_blocks must be an int >= 1")
            return MedianRef(ref_blocks)
        if mode == "ewma":
            a = float(ewma_alpha)
            if not (0.0 < a <= 1.0):
                raise ValueError(f"ewma_alpha must be in (0, 1], got {ewma_alpha!r}")
            return EwmaRef(a)
        if mode == "detrend":
            return DetrendRef()
        if mode == "diff":
            return DiffRef()
        raise ValueError(f"unknown reference {reference!r}; expected one of {MODES}")
    return FixedRef(reference)
