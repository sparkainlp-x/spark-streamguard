"""Spark StreamGuard: a fail-closed, auditable gate for any streaming detector.

Blocks of 32 values are checked, scored by a pluggable scorer (``ewma`` by default, ``spark``,
optional ``isoforest``, or your own), admitted or latched with a reason code, and logged.
Research prototype; see the README for SYNTHETIC and REPORTED evidence.
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
from streamguard.scorers import (
    SCORERS,
    EwmaScorer,
    IsoForestScorer,
    Scorer,
    SparkScorer,
    make_scorer,
)
from streamguard.stream import (
    DEFAULT_SCORER,
    STREAM_QUANTILE,
    STREAM_WARMUP_BLOCKS,
    AuditLogWriter,
    Decision,
    StreamGuard,
)

__version__ = "0.1.0"

__all__ = [
    "BLOCK",
    "DEFAULT_QUANTILE",
    "DEFAULT_SCORER",
    "DEFAULT_TAU",
    "SCORERS",
    "W_MEAN_ABS",
    "W_PEAK",
    "W_RMS",
    "STREAM_QUANTILE",
    "STREAM_WARMUP_BLOCKS",
    "AuditLogWriter",
    "BlockScore",
    "CalibrationError",
    "Decision",
    "DriftConfig",
    "EwmaScorer",
    "IsoForestScorer",
    "Reason",
    "Scorer",
    "SparkScorer",
    "StreamGuard",
    "TauCalibration",
    "__version__",
    "block_score",
    "calibrate_tau",
    "check_block",
    "decide",
    "drift_update",
    "latch_bit",
    "make_scorer",
    "residual",
    "score_components",
]
