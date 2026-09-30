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

__version__ = "0.1.0.dev0"

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
