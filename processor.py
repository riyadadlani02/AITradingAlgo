from enum import Enum
from tpqoa import tpqoa
from threading import RLock
from collections import defaultdict
from model import signal_queue, Prediction, SignalProcessingException


class StateAttrDef:
    QUANTITY = 'quantity'
    PRICE = 'price'
    REAL_PNL = 'real_pnl'


class SignalProcessingType(Enum):
    GO_LONG = 1
    GO_SHORT = 2
    GO_LONG_FROM_SHORT = 3
    GO_SHORT_FROM_LONG = 4
    STOP = 5
    NOOP = 6


class SignalProcessor:
    def __init__(self, conf_file='../oanda.cfg'):
        self.state = defaultdict(self._get_empty_state)
        self.oanda = tpqoa(conf_file)
        self.lock = RLock()

    @staticmethod
    def _get_empty_state():
        empty_state = dict()
        empty_state[StateAttrDef.QUANTITY] = 0.0
        empty_state[StateAttrDef.PRICE] = 0.0
        empty_state[StateAttrDef.REAL_PNL] = 0.0
        return empty_state

    def listen_to_signal(self):
        while True:
            try:
                signal = signal_queue.get()
                print(f'Processing signal: {signal.signal_id}')
                self.process_signal(signal)
            except Exception as e:
                import traceback
                print(f'{traceback.format_exc()}')

    def process_signal(self, signal):
        with self.lock:
            current_state = self.state[signal.instrument]
            print(f'Start processing signal {signal}.\nState={current_state}')
            signum_value = self._get_signum(current_state[StateAttrDef.QUANTITY])
            processing_type = self._get_processing_type(
                    signum_value, signal.prediction)
            print(f'Processing type: {processing_type}')
            state = self.place_trades_for_signal(
                    signal, processing_type, current_state)
            print(f'Processed signal: {signal}.\nState={state}')

    def place_trades_for_signal(self, signal, processing_type, current_state):
        if SignalProcessingType.NOOP == processing_type:
            return current_state

        if processing_type in [SignalProcessingType.GO_LONG,
                SignalProcessingType.GO_LONG_FROM_SHORT]:
            net_qty = signal.quantity + abs(current_state[StateAttrDef.QUANTITY])
        elif processing_type in [SignalProcessingType.GO_SHORT,
                SignalProcessingType.GO_SHORT_FROM_LONG]:
            net_qty = ((-1 * signal.quantity) -
                    abs(current_state[StateAttrDef.QUANTITY]))
        elif processing_type == SignalProcessingType.STOP:
            net_qty = -1 * current_state[StateAttrDef.QUANTITY]
        else:
            msg = f'Invalid processing type {processing_type} encountered.'
            raise SignalProcessingException(msg)

        order_response = self._place_trade_for_ins(signal.instrument, net_qty)
        updated_state = self._update_state(order_response, current_state)

        return updated_state

    @staticmethod
    def _update_state(order_response, state):
        if 'tradesClosed' in order_response:
            for closed_trd in order_response['tradesClosed']:
                state[StateAttrDef.QUANTITY] += float(closed_trd['units'])
                state[StateAttrDef.REAL_PNL] += float(closed_trd['realizedPL'])

                if abs(state[StateAttrDef.QUANTITY]) <= 0.0001:
                    state[StateAttrDef.PRICE] = 0.0

        if 'tradeOpened' in order_response:
            open_trade = order_response['tradeOpened']
            state[StateAttrDef.QUANTITY] += float(open_trade['units'])
            state[StateAttrDef.PRICE] += float(open_trade['price'])

        return state

    def _place_trade_for_ins(self, instrument, qty):
        response = self.oanda.create_order(
                instrument, qty, ret=True, suppress=True)

        # if type is not ORDER_FILL, there is some problem wih the order placement.
        if response['type'] != 'ORDER_FILL':
            raise SignalProcessingException(f'Error creating order: {response}.')

        return response

    @staticmethod
    def _get_processing_type(signum_value, prediction):
        if Prediction.STOP == prediction:
            return SignalProcessingType.STOP

        if signum_value == 0:
            return SignalProcessingType.GO_LONG if prediction == Prediction.LONG \
                    else SignalProcessingType.GO_SHORT

        if signum_value == 1:
            return SignalProcessingType.GO_SHORT_FROM_LONG \
                    if prediction == Prediction.SHORT \
                    else SignalProcessingType.NOOP

        if signum_value == -1:
            return SignalProcessingType.GO_LONG_FROM_SHORT \
                    if prediction == Prediction.LONG \
                    else SignalProcessingType.NOOP

    @staticmethod
    def _get_signum(x):
        if x == 0:
            return 0
        else:
            return 1 if x > 0 else -1
