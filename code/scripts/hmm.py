import numpy as np
from hmmlearn.hmm import GaussianHMM

class HMM:
    def __init__(self, n_states=2):
        self.model = GaussianHMM(n_components=n_states, covariance_type="full")
        # TODO: Add any custom initialization variables you need

    def prepare_data(self, data):
        # TODO: Convert raw financial data to stationary features (e.g., log returns)
        # TODO: Reshape the array to 2D (n_samples, n_features) as required by hmmlearn
        pass

    def fit(self, data):
        # TODO: Run data through self.prepare_data()
        # TODO: Call self.model.fit() on the prepared features
        # TODO: Analyze self.model.means_ and self.model.covars_ to label your regimes
        pass

    def predict(self, data):
        # TODO: Run data through self.prepare_data()
        # TODO: Call self.model.predict() to get the hidden state sequence
        # TODO: Return the sequence (or just the most recent state for live trading)
        pass

