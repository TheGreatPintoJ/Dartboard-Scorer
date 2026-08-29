"""Camera-based dartboard scoring."""

from .calibration import Calibration
from .detector import Dart, DartDetector, State
from .geometry import Score, score_at
from .session import Session

__version__ = "1.0.0"
__all__ = ["Calibration", "Dart", "DartDetector", "State", "Score", "score_at", "Session"]
