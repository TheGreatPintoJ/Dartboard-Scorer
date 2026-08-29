"""Camera-based dartboard scoring."""

from .calibration import Calibration
from .config import AppConfig
from .detector import Dart, DartDetector, State
from .geometry import Score, score_at
from .session import Session

__version__ = "2.0.0"
__all__ = ["AppConfig", "Calibration", "Dart", "DartDetector", "State",
           "Score", "score_at", "Session"]
