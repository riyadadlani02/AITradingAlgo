import numpy as np
from model import Prediction
from base import BaseStrategy


class sma(BaseStrategy):
    """
    This is the set of default model parameters.
        Override and add where applicable.
    """

    def __init__(self, model_parameters, config):
        super().__init__(model_parameters, config)
        if model_parameters['sma1'] or model_parameters['sma2'] > self.n_bars:
            if model_parameters['sma1'] > model_parameters['sma2']:
                self.n_bars = model_parameters['sma1'] * 3
            else:
                self.n_bars = model_parameters['sma2'] * 3

    def custom_data_preparation(self, data, is_training_data):
        """
        Add required data preparations here.
        """
        prediction = self.instrument + '_prediction'
        data['sma1'] = (data[self.instrument + '_close'].rolling(
            self.sma1).mean().shift(1))
        data['sma2'] = (data[self.instrument + '_close'].rolling(
            self.sma2).mean().shift(1))
        data.dropna(inplace=True)
        data[prediction] = np.where(data['sma1'] > data['sma2'], 1, -1)

    def on_signal(self, predicted_data, signal_date):
        """
        This method is called every time the strategy generates a signal.
        """
        direction = predicted_data.loc[signal_date][
                self.instrument + '_prediction']
        if direction == -1:
            prediction = Prediction.SHORT
        else:
            prediction = Prediction.LONG
        return prediction
