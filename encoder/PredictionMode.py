from enum import Enum


class PredictionMode(Enum):
    INTER_FRAME = 0  # P-frame
    INTRA_FRAME = 1  # I-frame
