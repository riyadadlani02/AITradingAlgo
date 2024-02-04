import re
import math
import uuid
import numpy as np
import pandas as pd
from time import sleep
from tpqoa import tpqoa
from retrying import retry
from dateutil import parser
from abc import abstractmethod, ABCMeta
from datetime import timedelta, datetime
from model import Signal, Prediction, signal_queue


valid_instruments = ['EUR_USD', 'BCO_USD']
valid_frequency = ['M1', 'M5', 'M10', 'M30']


def roundup(x, freq):
    return int(math.ceil(x / freq)) * freq


def round_time(dt=None, date_delta=timedelta(minutes=1), to='average'):
    """
        Round a datetime object to a multiple of a timedelta
        dt: datetime.datetime object, default now.
        dateDelta: timedelta object, we round to a multiple of this,
            default 1 minute.
        from: http://stackoverflow.com/questions/3463930/how-to-round-the-minute-of-a-datetime-object-python
    """
    round_to = date_delta.total_seconds()

    if dt is None:
        dt = datetime.utcnow()
    seconds = (dt - timedelta(dt.minute)).second

    if to == 'up':
        rounding = (seconds + round_to) // round_to * round_to
    elif to == 'down':
        rounding = seconds // round_to * round_to
    else:
        rounding = (seconds + round_to / 2) // round_to * round_to

    dt = dt + timedelta(seconds=(rounding - seconds),
            microseconds=-dt.microsecond)
    return dt


class BaseStrategy(metaclass=ABCMeta):
    def __init__(self, model_parameters, config):
        self.data_source = tpqoa(config)
        self.data = pd.DataFrame()
        self.live_price_data = pd.DataFrame()
        self.model_params = model_parameters
        self.n_bars = 120
        self.first_run = True
        self.stop_model = False
        self.model_id = uuid.uuid4()
        self.signal_count = 0
        self.feature_labels = set()

        # Important model params
        self.trading_quantity = None
        self.instrument = None
        self.frequency = None

        self.initialize()
        self.validate()


    def _initialize_model_params(self):
        model_parameters = self.model_params
        for key, value in model_parameters.items():
            setattr(self, key, value)

    @staticmethod
    def _get_time_unit_and_duration(freq):
        freq = re.findall(r'[A-Za-z]+|\d+', freq)
        min_or_sec = freq[0]
        duration = int(freq[1])
        return duration, min_or_sec

    @retry(stop_max_attempt_number=7, wait_fixed=5000, wrap_exception=True)
    def _get_data(self, instrument, start, end, freq=None, price='M'):
        if freq is None:
            freq = self.frequency
        msg = f"Trying to get data from OANDA for {instrument} {start} {end}"
        msg += f" {freq} {price} at {datetime.utcnow()}"
        print(msg)

        start = parser.parse(start).strftime("%Y-%m-%d %H:%M:%S")
        end = parser.parse(end).strftime("%Y-%m-%d %H:%M:%S")
        raw_data = self.data_source.get_history(
                instrument, start, end, freq, price)
        return raw_data

    @staticmethod
    def _sleep_for_signal_gen(duration, signal_date):
        current_min = parser.parse(signal_date).minute
        current_second = parser.parse(signal_date).second
        next_min_level = roundup(current_min, duration)
        seconds_to_sleep = (((next_min_level - current_min) * 60) + 1 -
                current_second)
        if seconds_to_sleep > 0:
            print(f'signal gen thread: sleeping for {seconds_to_sleep} seconds')
            sleep(seconds_to_sleep)

    @staticmethod
    def _sleep_until_next_signal(duration, min_or_sec, signal_date):
        time_diff = (parser.parse(signal_date) -
                parser.parse(datetime.utcnow().isoformat() + 'Z'))
        seconds_diff = time_diff.seconds
        microseconds_diff = time_diff.microseconds

        # Sleep till the next min
        sleep_duration = duration
        if min_or_sec == 'M':
            sleep_duration = 60 * duration
        if seconds_diff < sleep_duration:
            msg = f'signal gen thread: sleeping for '
            msg += f'{seconds_diff + microseconds_diff / 1000000} seconds'
            print(msg)
            sleep(seconds_diff + microseconds_diff / 1000000)

    def _publish_stop_signal(self):
        signal = Signal()
        signal.signal_id = uuid.uuid4()
        signal.model_id = self.model_id
        signal.instrument = self.instrument
        signal.prediction = Prediction.STOP
        self._publish_signal(signal)

    @staticmethod
    def _publish_signal(signal):
        print(f'Publishing Signal: {signal.signal_id}')
        signal_queue.put(signal)

    def _prepare_predict_data(self, original_signal_date):
        predict_data = pd.DataFrame()
        predict_data[self.instrument + '_close'] = self.live_price_data['c']
        predict_data[self.instrument + '_open'] = self.live_price_data['o']
        predict_data[self.instrument + '_high'] = self.live_price_data['h']
        predict_data[self.instrument + '_low'] = self.live_price_data['l']
        predict_data[self.instrument + '_volume'] = self.live_price_data['volume']
        predict_data[self.instrument + '_date'] = self.live_price_data['time']
        predict_data[self.instrument + '_return'] = \
            np.log(predict_data[self.instrument + '_close'] / \
            predict_data[self.instrument + '_close'].shift(1))
        predict_data.dropna(inplace=True)
        predict_data.set_index(self.instrument + '_date', inplace=True)
        predict_data.loc[parser.parse(original_signal_date)] = 100
        return predict_data

    def _get_signal_for_prediction(self, prediction):
        signal = Signal()
        signal.signal_id = uuid.uuid4()
        signal.model_id = self.model_id
        signal.instrument = self.instrument
        signal.prediction = prediction
        signal.quantity = self.trading_quantity
        return signal


    def set_n_bars(self, n_bars):
        # Override the number of candles to be fetched from data source.
        self.n_bars = n_bars

    def initialize(self):
        self._initialize_model_params()

    def validate(self):
        instrument = self.model_params['instrument']
        if instrument not in valid_instruments:
            exit(f'{instrument} is not a valid/supported instruments')
        self.instrument = instrument

        frequency = self.model_params['frequency']
        if frequency not in valid_frequency:
            exit(f'{frequency} is not a valid/supported frequency')
        self.frequency = frequency

        if 'trading_quantity' not in self.model_params:
            exit(f'trading quantity is mandatory')
        else:
            self.trading_quantity = self.model_params['trading_quantity']

        if ('n_signals_to_gen' not in self.model_params) \
                and ('stop_time' not in self.model_params):
            exit('stop_time or n_signals_to_gen required as exit condition')

    def generate_signal(self):
        signal_date = datetime.utcnow().isoformat()[:-7] + 'Z'
        duration, min_or_sec = self._get_time_unit_and_duration(self.frequency)

        if self.first_run is True and 'trade_immediately' in self.model_params and \
                self.model_params['trade_immediately'] is True:
            self.first_run = False
        else:
            self.first_run = False
            if min_or_sec == 'M':
                self._sleep_for_signal_gen(duration, signal_date)

            signal_date = datetime.utcnow().isoformat()[:-7] + 'Z'

        print(f"generating signal now {datetime.utcnow()}")
        while True:
            try:
                self.check_for_stop_condition(signal_date)
                if self.stop_model is True:
                    self._publish_stop_signal()
                    break

                if min_or_sec == 'M':
                    signal_date = round_time(parser.parse(signal_date),
                            date_delta=timedelta(minutes=duration),
                            to='up').isoformat()[:-6] + 'Z'
                signal = self.predict_for_time(signal_date)
                self.signal_count += 1
                self._publish_signal(signal)

                if min_or_sec == 'M':
                    self._sleep_for_signal_gen(duration, signal_date)
                sleep(2)

                self._sleep_until_next_signal(duration, min_or_sec, signal_date)
                signal_date = datetime.utcnow().isoformat()[:-7] + 'Z'

            except Exception as e:
                import traceback
                print(f'{traceback.format_exc()}')

    def check_for_stop_condition(self, signal_time):
        if 'n_signals_to_gen' in self.model_params:
            if self.signal_count >= self.model_params['n_signals_to_gen']:
                self.stop_model = True
        if 'stop_time' in self.model_params:
            stop_time = parser.parse(
                parser.parse(self.model_params['stop_time']
                    ).strftime("%Y-%m-%dT%H:%M:%SZ"))
            if stop_time <= signal_time:
                self.stop_model = True

    def predict_for_time(self, signal_date=None, is_first_run=False):
        signal_date = signal_date[:-1]
        original_signal_date = signal_date
        signal_date = parser.parse(signal_date)

        # * 3 is to avoid the lags being NaN
        time_periods_to_populate = self.n_bars

        start = self.get_starting_time(signal_date, time_periods_to_populate)

        raw_data = self._get_data(self.instrument,
                start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                freq=self.frequency, price='M')
        raw_data_len = len(raw_data)
        time_diff = (signal_date - start).seconds / 60
        duration, min_or_sec = self._get_time_unit_and_duration(self.frequency)

        retry_count = 0
        while raw_data_len < time_diff / duration:
            sleep(2)
            if retry_count > 6:
                print("Expected candles are {} got {}; stopping model.".format(
                    str(int(time_diff / duration)), str(raw_data_len)))
                self.stop_model = True
                break
            raw_data = self._get_data(self.instrument,
                    start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    freq=self.frequency, price='M')
            raw_data_len = len(raw_data)
            retry_count += 1
        print("Expected candles are {} got {}.".format(
            str(int(time_diff / duration)), str(raw_data_len)))

        if self.stop_model is True:
            return

        raw_data.dropna(inplace=True)
        self.live_price_data = raw_data.reset_index()

        predict_data = self._prepare_predict_data(original_signal_date)
        self.custom_data_preparation(predict_data, False)

        prediction = self.on_signal(predict_data, signal_date)
        signal = self._get_signal_for_prediction(prediction)

        return signal

    def get_starting_time(self, signal_date, delta):
        duration, min_or_sec = self._get_time_unit_and_duration(self.frequency)
        if 'D' in self.frequency:
            return_date = signal_date - timedelta(days=delta * duration)
        elif 'M' in self.frequency:
            return_date = signal_date - timedelta(minutes=delta * duration)
        elif 'S' in self.frequency:
            return_date = signal_date - timedelta(seconds=delta * duration * 2)
        else:
            raise Exception(self.frequency + ' is not supported')

        return return_date


    @abstractmethod
    def custom_data_preparation(self, data, is_train_date):
        pass

    @abstractmethod
    def on_signal(self, predicted_data, signal_date):
        pass
