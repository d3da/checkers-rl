import torch
import numpy as np
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from sklearn.base import BaseEstimator, RegressorMixin

from train import QModelTrainRun
from model import CheckersQModel

class CheckersQModelWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, num_hidden_layers=0, hidden_size=0, learning_rate=0.01,
                 self_play_games_per_iter=25, weight_decay=1e-4):
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.self_play_games_per_iter = self_play_games_per_iter
        self.weight_decay = weight_decay

        # Initialize model here
        self.model = None

        # Initialize train_hist to an empty list
        self.train_hist = []

    def fit(self, X, y=None, **kwargs):
        # Update the model's parameters
        self.model = CheckersQModel(
            num_hidden_layers=self.num_hidden_layers,
            hidden_size=self.hidden_size
        )

        # Set other parameters as needed
        self.optimzer = torch.optim.SGD(self.model.parameters(),
                                        lr=self.learning_rate,
                                        weight_decay=self.weight_decay)
        self.train_run = QModelTrainRun(self.model, self.optimzer)

        # Train the model and get the metric to optimize (e.g., average win rate)
        self.train_hist = self.train_run.train(selfplay_games_p_i=self.self_play_games_per_iter)
        return self

    def score(self, X, y=None, **kwargs):
        num_evaluation_games = 100
        wins, draws = self.train_run.evaluate_strength(num_evaluation_games=num_evaluation_games)
        return wins / num_evaluation_games

    def get_params(self, deep=True):
        # Implement get_params method here
        return {'num_hidden_layers': self.num_hidden_layers,
                'hidden_size': self.hidden_size,
                'learning_rate': self.learning_rate,
                'self_play_games_per_iter': self.self_play_games_per_iter,
                'weight_decay': self.weight_decay}

    def set_params(self, **parameters):
        # Implement set_params method here
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


def optimize_hyperparameters():
    # Define the hyperparameter search space
    search_space = {
        'num_hidden_layers': Integer(0, 3),
        'hidden_size': Integer(64, 1024),
        'learning_rate': Real(1e-5, 1e-1, 'log-uniform'),
        'self_play_games_per_iter': Integer(1, 25),
        'weight_decay': Real(1e-5, 1e-1, 'log-uniform')
    }

    wrapper = CheckersQModelWrapper()

    # Use BayesSearchCV for optimization
    opt = BayesSearchCV(wrapper, search_space, n_iter=30, random_state=42, verbose=2)

    # Pass a dummy X (input data) and y (target) for optimization
    X_dummy = np.random.rand(100, 10)  # 100 samples, 10 features
    y_dummy = np.random.randint(0, 2, 100)
    best_optimized_params = opt.fit(X_dummy, y_dummy).best_params_

    return best_optimized_params


if __name__ == '__main__':
    # Perform Bayesian optimization to find the best hyperparameters
    best_hyperparams = optimize_hyperparameters()

    # Use the best hyperparameters to train the model
    best_model = CheckersQModel(
        num_hidden_layers=best_hyperparams['num_hidden_layers'],
        hidden_size=best_hyperparams['hidden_size']
    )

    # Train the model using the best hyperparameters
    best_optimizer = torch.optim.SGD(best_model.parameters(),
                                     lr=best_hyperparams['learning_rate'],
                                     weight_decay=best_hyperparams['weight_decay'])
    best_hparams_train_run = QModelTrainRun(best_model, best_optimizer)
    train_hist = best_hparams_train_run.train(selfplay_games_p_i=best_hyperparams['self_play_games_per_iter'])

    # Evaluate the model against random moves with the best hyperparameters
    num_games = 100
    wins, draws = best_hparams_train_run.evaluate_strength(num_evaluation_games=num_games)
    print(f'Final Win Rate: {wins / num_games}, Draw Rate: {draws / num_games}')
