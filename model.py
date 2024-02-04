import queue
from enum import Enum

signal_queue = queue.Queue()


class Prediction(Enum):
    LONG = 1
    SHORT = 2
    NEUTRAL = 3
    STOP = 4


class Signal:
    def __init__(self):
        self.model_id = None
        self.signal_id = None
        self.instrument = None
        self.prediction_time = None
        self.prediction = None
        self.quantity = None

    def __repr__(self):
        return str(self.__dict__)


class SignalProcessingException(Exception):
    pass
