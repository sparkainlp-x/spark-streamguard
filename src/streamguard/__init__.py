"""Spark StreamGuard: a streaming, fail-closed guardrail for numeric data in 32-value blocks.

Research prototype (milestone 1). No benchmarks have been run yet; every performance figure
in this project is TARGET or UNRUN.
"""

from streamguard.calibrate import (
    DEFAULT_QUANTILE,
    DEFAULT_TAU,
    CalibrationError,
    DriftConfig,
    TauCalibration,
    calibrate_tau,
    drift_update,
)
from streamguard.latch import Reason, check_block, decide, latch_bit
from streamguard.score import (
    BLOCK,
    W_MEAN_ABS,
    W_PEAK,
    W_RMS,
    BlockScore,
    block_score,
    residual,
    score_components,
)
from streamguard.stream import AuditLogWriter, Decision, StreamGuard

__version__ = "0.1.0.dev0"

__all__ = [
    "BLOCK",
    "DEFAULT_QUANTILE",
    "DEFAULT_TAU",
    "W_MEAN_ABS",
    "W_PEAK",
    "W_RMS",
    "AuditLogWriter",
    "BlockScore",
    "CalibrationError",
    "Decision",
    "DriftConfig",
    "Reason",
    "StreamGuard",
    "TauCalibration",
    "__version__",
    "block_score",
    "calibrate_tau",
    "check_block",
    "decide",
    "drift_update",
    "latch_bit",
    "residual",
    "score_components",
]
