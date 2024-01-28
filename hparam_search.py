#!/usr/bin/env python3
from abc import ABC, abstractmethod
import os
import pandas as pd
import torch
import numpy as np
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from sklearn.base import BaseEstimator, RegressorMixin
from agent import PieceCountHeuristicsAgent

from train import QModelTrainRun, VModelTrainRun
from model import CheckersQModel, CheckersVModel

# np.int is deprecated since numpy 1.20, but skopt still uses it.
# For some reason they haven't fixed it so this workaround will do for now.
np.int = np.int64


class ModelWrapper(BaseEstimator, RegressorMixin, ABC):
    def __init__(self, num_hidden_layers=0, hidden_size=0, learning_rate=0.01,
                 train_batches_p_i=100, weight_decay=1e-4):
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.train_batches_p_i = train_batches_p_i
        self.weight_decay = weight_decay

        # Initialize model here
        self.model = None
        self.optimizer = None
        self.train_run = None

        # Initialize train_hist to an empty list
        self.train_hist = []

    @abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError

    def fit(self, X, y=None, **kwargs):
        # Update the model's parameters
        self._setup()
        assert self.model
        assert self.optimizer
        assert self.train_run

        # Train the model and get the metric to optimize (e.g., average win rate)
        self.train_hist = self.train_run.train(train_batches_p_i=self.train_batches_p_i,
                                               disable_progress=True)
        return self

    def score(self, X, y=None, **kwargs):
        assert self.train_run
        wr, dr, lr = self.train_run.evaluate_strength(disable_progress=True,
                                                      enemy_agent=PieceCountHeuristicsAgent(),
                                                      enemy_agent_kwargs=dict(depth=2))
        print(f'winrate: {wr}, draws: {dr}, losses: {lr}')
        return wr

    def get_params(self, deep=True):
        # Implement get_params method here
        return {'num_hidden_layers': self.num_hidden_layers,
                'hidden_size': self.hidden_size,
                'learning_rate': self.learning_rate,
                'train_batches_p_i': self.train_batches_p_i,
                'weight_decay': self.weight_decay}

    def set_params(self, **parameters):
        # Implement set_params method here
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


class QModelWrapper(ModelWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup(self) -> None:
        assert self.learning_rate
        assert self.weight_decay
        self.model = CheckersQModel(
            num_hidden_layers=self.num_hidden_layers,
            hidden_size=self.hidden_size
        )
        # Set other parameters as needed
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                        lr=self.learning_rate,
                                        weight_decay=self.weight_decay)
        self.train_run = QModelTrainRun(self.model, self.optimizer)


class VModelWrapper(ModelWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup(self) -> None:
        assert self.learning_rate
        assert self.weight_decay
        self.model = CheckersVModel(
            num_hidden_layers=self.num_hidden_layers,
            hidden_size=self.hidden_size
        )
        # Set other parameters as needed
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=self.learning_rate,
                                         weight_decay=self.weight_decay)
        self.train_run = VModelTrainRun(self.model, self.optimizer)


def optimize_hyperparameters(wrapper_cls):
    # Define the hyperparameter search space
    search_space = {
        'num_hidden_layers': Integer(0, 3),
        'hidden_size': Integer(512, 2048),
        'learning_rate': Real(1e-4, 1e-1, 'log-uniform'),
        'train_batches_p_i': Integer(25, 500),
        'weight_decay': Real(1e-5, 1e-1, 'log-uniform')
    }

    wrapper = wrapper_cls()

    # Use BayesSearchCV for optimization
    n_jobs = int((os.cpu_count() or 1.5) * 0.75)
    opt = BayesSearchCV(wrapper,
                        search_space,
                        n_iter=25,
                        random_state=42,
                        verbose=2,
                        cv=5,
                        n_jobs=n_jobs,
                        n_points=8)

    # Pass a dummy X (input data) and y (target) for optimization
    X_dummy = np.random.rand(100, 10)  # 100 samples, 10 features
    y_dummy = np.random.randint(0, 2, 100)
    opt.fit(X_dummy, y_dummy)
    return opt.best_params_, opt.cv_results_


if __name__ == '__main__':
    # Perform Bayesian optimization to find the best hyperparameters
    best_hyperparams, results = optimize_hyperparameters(VModelWrapper)
    res_df = pd.DataFrame(results)

    print(best_hyperparams)
    print(res_df)

    res_df.to_csv('search_results.csv')

    # Use the best hyperparameters to train the model
    best_model = CheckersVModel(
        num_hidden_layers=best_hyperparams['num_hidden_layers'],
        hidden_size=best_hyperparams['hidden_size']
    )

    # Train the model using the best hyperparameters
    print('Training using best hyperparameters')
    best_optimizer = torch.optim.SGD(best_model.parameters(),
                                     lr=best_hyperparams['learning_rate'],
                                     weight_decay=best_hyperparams['weight_decay'])
    best_hparams_train_run = VModelTrainRun(best_model, best_optimizer)
    train_hist = best_hparams_train_run.train(train_batches_p_i=best_hyperparams['train_batches_p_i'])

    # Evaluate the model against random moves with the best hyperparameters
    wr, dr, lr = best_hparams_train_run.evaluate_strength()
    print(f'Against random moves: final Win Rate: {wr}, Draw Rate: {dr}')

    wr, dr, lr = best_hparams_train_run.evaluate_strength(enemy_agent=PieceCountHeuristicsAgent(), enemy_agent_kwargs=dict(depth=2))
    print(f'Against heuristics (depth=3): Win rate {wr}, Draw rate {dr}')

    exit()

