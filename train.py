#!/bin/env python3

from abc import ABC, abstractmethod
from collections import Counter, deque
from typing import NamedTuple, Literal, Any
import os
import torch 
import numpy as np
import tqdm
import random
import math
import pandas as pd

from model import CheckersQModel, CheckersVModel
from game import Game, GameState, Player, Action, NUM_ACTIONS
from agent import play_agent_game, BaseAgent, QModelAgent, RandomAgent, UserInputAgent, VModelAgent
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from sklearn.base import BaseEstimator, RegressorMixin


class Sample(NamedTuple):
    game_state: GameState
    action: Action
    current_player: Player
    winner: Player


class ReplayBuffer:
    """
    Store tuples of game state, action, current player and the game winner,
    which can be used by the model to (hopefully) learn something from experience
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: deque[Sample] = deque([], capacity)
        self.winners = Counter()
        self.game_lengths = Counter()

    def append(self,
               game_history: list[tuple[GameState, Action, Player]],
               winner: Player) -> None:
        """
        Add a played out game to the replay buffer
        """
        for state, action, current_player in game_history:
            self.buffer.append(Sample(state, action, current_player, winner))

        self.winners.update([winner])
        self.game_lengths.update([len(game_history)])

    def sample_k(self, k: int) -> list[Sample]:
        """Randomly sample batch_size samples from the replay buffer"""
        return random.choices(self.buffer, weights=None, k=k)

    @property
    def is_full(self) -> bool:
        return len(self) == self.buffer.maxlen

    def __len__(self) -> int:
        return len(self.buffer)


class SmoothedAverage:
    """
    Class implementing an exponential moving average (EMA)
    https://en.wikipedia.org/wiki/Exponential_smoothing

    Mainly so the progress meters don't jump all over the place, and they are a bit easier to interpret
    """
    def __init__(self, alpha: float = 0.25, initial: float | None = None) -> None:
        self._ema: float | None = initial
        self._last_value: float | None = initial
        self.alpha = alpha

    def update(self, value: float) -> None:
        self._last_value = value
        if self._ema is None:
            self._ema = value
            return

        self._ema *= 1 - self.alpha
        self._ema += self.alpha * value

    @property
    def value(self) -> float:
        if self._ema is None:
            return math.nan
        return self._ema

    @property
    def unsmoothed_value(self) -> float | None:
        return self._last_value


class TrainRun(ABC):

    def __init__(self, model_agent: BaseAgent, optimizer: torch.optim.Optimizer) -> None:
        self.model_agent = model_agent
        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        self.optimizer = optimizer

        self.model_agent = model_agent
        self.random_agent = RandomAgent()
        self.game = Game()

        self.total_selfplay_games = 0

    def train(self,
              replay_buffer_capacity: int = 100_000,
              initial_experience_samples: int | None = 10_000,
              num_train_iterations: int = 10,
              selfplay_games_p_i: int = 10,
              train_batches_p_i: int = 100,
              batch_size: int = 128,
              epsilon_anneal_iters: int = 60,
              min_epsilon: float = 0.2,
              max_num_moves: int | None = 200,
              disable_progress: bool = False) -> pd.DataFrame:

        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)
        self._fill_replay_buffer_random_moves(initial_experience_samples, disable_progress=disable_progress)

        train_history = self._train_loop(num_train_iterations,
                                         selfplay_games_p_i, train_batches_p_i,
                                         batch_size,
                                         epsilon_anneal_iters, min_epsilon,
                                         max_num_moves=max_num_moves,
                                         disable_progress=disable_progress)
        return pd.DataFrame(train_history)

    def _fill_replay_buffer_random_moves(self,
                                         num_samples: int | None,
                                         disable_progress: bool) -> None:
        """
        Fill the given replay buffer with num_samples samples,
        or fill it completely if num_samples is None

        Games are played by random agents.
        Returns the number of games played.
        """
        def is_done():
            if num_samples is None:
                return self.replay_buffer.is_full
            return len(self.replay_buffer) >= num_samples

        progress_bar = tqdm.tqdm(total=num_samples or self.replay_buffer.capacity,
                                 dynamic_ncols=True,
                                 desc='Filling replay buffer',
                                 disable=disable_progress)
        while not is_done():
            _, winner, game_history = play_agent_game(self.game, self.random_agent, self.random_agent)
            self.replay_buffer.append(game_history, winner)
            self.total_selfplay_games += 1
            progress_bar.update(len(game_history))
            progress_bar.set_postfix(dict(games=self.total_selfplay_games))

    def _make_batch(self, samples: list[Sample]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Turn a batch from the replay buffer into something usable by pytorch modules
        """
        states: list[np.ndarray] = []
        actions: list[Action] = []
        winners: list[Literal[-1, 0, 1]] = []
        for sample in samples:
            states.append(sample.game_state.to_array())
            actions.append(sample.action)
            winners.append(sample.winner.value)

        return torch.tensor(np.array(states, dtype=np.float32)), \
                torch.tensor(np.array(actions, dtype=np.int64)), \
                torch.tensor(np.array(winners, dtype=np.float32))

    @abstractmethod
    def _train_on_experience(self,
                            num_train_batches: int,
                            batch_size: int,
                            disable_progress: bool) -> list[float]:
        raise NotImplementedError

    def _generate_selfplay_experience(self,
                                      num_selfplay_games: int,
                                      selfplay_agent_kwargs: dict[str, Any] | None,
                                      max_num_moves: int | None,
                                      disable_progress: bool) -> None:
        selfplay_agent_kwargs = selfplay_agent_kwargs or {}

        for _ in tqdm.trange(num_selfplay_games, position=1, leave=False,
                             desc='Playing against self', disable=disable_progress):
            _, winner, history = play_agent_game(self.game,
                                                 self.model_agent, self.model_agent,
                                                 selfplay_agent_kwargs, selfplay_agent_kwargs,
                                                 max_num_moves)
            self.replay_buffer.append(history, winner)
            self.total_selfplay_games += 1

    def _train_loop(self,
                    num_iterations: int,
                    selfplay_games_p_i: int,
                    train_batches_p_i: int,
                    batch_size: int,
                    epsilon_anneal_iters: int,
                    min_epsilon: float,
                    max_num_moves: int | None,
                    disable_progress: bool):
        train_history: list[dict[str, Any]] = []

        progress_bar = tqdm.trange(num_iterations, dynamic_ncols=True, colour='#22ee22', disable=disable_progress)
        loss_ema = SmoothedAverage(initial=1.0)  # Used only in the progress bar

        for iteration in progress_bar:
            epsilon = max(min_epsilon, (epsilon_anneal_iters - iteration) / epsilon_anneal_iters)
            if iteration > 0:
                self._generate_selfplay_experience(selfplay_games_p_i,
                                                   dict(epsilon=epsilon),
                                                   max_num_moves=max_num_moves,
                                                   disable_progress=disable_progress)

            batch_losses = self._train_on_experience(train_batches_p_i, batch_size, disable_progress=disable_progress)
            iteration_loss = sum(batch_losses) / train_batches_p_i
            if not disable_progress:
                loss_ema.update(iteration_loss)
                progress_bar.set_postfix_str(f'loss={loss_ema.value:.4f} ' +
                                             f'eps={epsilon:.4f} ' +
                                             f'games={self.total_selfplay_games}')

            train_history.append(dict(iteration=iteration,
                                      loss=iteration_loss,
                                      epsilon=epsilon,
                                      selfplay_games=self.total_selfplay_games))

        return train_history

    def evaluate_strength(self,
                          num_evaluation_games: int = 100,
                          evaluation_epsilon: float = 0.05,
                          enemy_agent: BaseAgent | None = None,
                          enemy_agent_kwargs: dict[str, Any] | None = None,
                          disable_progress: bool = False) -> tuple[int, int]:
        """
        Play multiple games using this model vs a given enemy agent.
        Returns tuple(the number of times this model won, the number of draws)
        """
        enemy_agent = enemy_agent or RandomAgent()
        player_wins = draws = 0
        for _ in tqdm.trange(num_evaluation_games, position=1, leave=False,
                             desc='Playing against random moves', disable=disable_progress):
            rl_player, winner, _ = play_agent_game(self.game,
                                                   self.model_agent, enemy_agent,
                                                   dict(epsilon=evaluation_epsilon), enemy_agent_kwargs)
            if winner == rl_player:
                player_wins += 1
            elif winner == Player.NEUTRAL:
                draws += 1

        return player_wins, draws


class QModelTrainRun(TrainRun):
    def __init__(self, model: CheckersQModel, optimizer: torch.optim.Optimizer) -> None:
        super().__init__(QModelAgent(model), optimizer)
        self.model = model

    def _train_on_experience(self, num_train_batches: int, batch_size: int, disable_progress: bool) -> list[float]:
        losses = []
        # The model learn to predict the game winner given a state and action
        for _ in tqdm.trange(num_train_batches, position=1, leave=False,
                             desc='Learning from past experience',
                             disable=disable_progress):
            self.optimizer.zero_grad()

            states, actions, winners = self._make_batch(self.replay_buffer.sample_k(batch_size))
            predictions = self.model(states)  # (batch, num_actions)
            action_mask = torch.nn.functional.one_hot(actions, num_classes=NUM_ACTIONS)

            action_mask = action_mask.type(torch.float32)  # (batch, num_actions)
            predictions = predictions * action_mask
            winners = winners.unsqueeze(-1) * action_mask  # (batch, num_actions)

            loss = self.loss_fn(predictions, winners) / batch_size
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

        return losses



class VModelTrainRun(TrainRun):
    def __init__(self, model: CheckersVModel, optimizer: torch.optim.Optimizer) -> None:
        super().__init__(VModelAgent(model), optimizer)
        self.model = model

    def _train_on_experience(self, num_train_batches: int, batch_size: int, disable_progress: bool) -> list[float]:
        losses = []
        # The model learn to predict the game winner given a state and action
        for _ in tqdm.trange(num_train_batches, position=1, leave=False,
                             desc='Learning from past experience',
                             disable=disable_progress):
            self.optimizer.zero_grad()

            states, _, winners = self._make_batch(self.replay_buffer.sample_k(batch_size))
            predictions = self.model(states)  # (batch, )
            winners = winners.unsqueeze(-1)  # (batch, )

            loss = self.loss_fn(predictions, winners) / batch_size
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

        return losses


@torch.no_grad()
def evaluate_model_vs_random(rl_agent: QModelAgent, random_agent: RandomAgent, epsilon: float, num_games: int):
    game = Game()
    player_wins = 0
    draws = 0

    for _ in tqdm.trange(num_games, position=1, leave=False, desc='Playing against random moves'):
        rl_player, winner, _ = play_agent_game(game, rl_agent, random_agent, dict(epsilon=epsilon))
        if winner == rl_player:
            player_wins += 1
        elif winner == Player.NEUTRAL:
            draws += 1

    win_rate = player_wins / num_games
    draw_rate = draws / num_games

    return win_rate, draw_rate


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
    # model = CheckersQModel(num_hidden_layers=1, hidden_size=256)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=3.0)
    # trainrun = QModelTrainRun(model, optimizer)
    # train_hist = trainrun.train()
    # print(train_hist)

    # Perform Bayesian optimization to find the best hyperparameters
    best_hyperparams = optimize_hyperparameters()

    # Use the best hyperparameters to train the model
    best_model = CheckersQModel(
        num_hidden_layers=best_hyperparams['num_hidden_layers'],
        hidden_size=best_hyperparams['hidden_size']
    )
    import pdb; pdb.set_trace()

    # Train the model using the best hyperparameters
    # train_hist = train_loop(
        # best_model,
        # learning_rate=best_hyperparams['learning_rate'],
        # self_play_games_per_iter=best_hyperparams['self_play_games_per_iter'],
        # weight_decay=best_hyperparams['weight_decay']
    # )

    # Evaluate the model against random moves with the best hyperparameters
    # win_rate, draw_rate = evaluate_model_vs_random(best_model, epsilon=0.05, num_games=10)
    # print(f'Final Win Rate: {win_rate}, Draw Rate: {draw_rate}')
