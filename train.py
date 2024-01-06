#!/bin/env python3

from collections import Counter, deque
from typing import NamedTuple, Literal
# import os
import torch 
import numpy as np
import tqdm
import random
import math
# import pandas as pd
# import matplotlib.pyplot as plt

from model import CheckersQModel
from game import Game, GameState, Player, Action
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from sklearn.base import BaseEstimator, RegressorMixin


@torch.no_grad()
def self_play_game(model: torch.nn.Module,
                   game: Game,
                   epsilon: float) \
                           -> tuple[list[tuple[GameState, Action, Player]], Player]:
    """
    Play through a game using self-play
    """
    game.reset()
    game_history: list[tuple[GameState, Action, Player]] = []

    while True:

        # Get the current game status from the Game object
        state = game.get_state()
        current_player = game.get_current_player()
        legal_actions = game.get_legal_actions()

        # Sample an action and pass it to the Game
        action = select_action(model, state, current_player, legal_actions, epsilon)
        game.play(action)

        # Record history
        game_history.append((state, action, current_player))

        if game.has_ended():
            return game_history, game.get_winner()


def select_action(model: torch.nn.Module,
                  state: GameState,
                  current_player: Player,
                  legal_actions: list[Action],
                  epsilon: float) -> Action:
    """
    Make the model select an action
    using an epsilon-greedy strategy w.r.t. the action-value function

    In other words, compute the estimated win probability for every action
    and select the legal action with the highest value.
    Or, with a chance of epsilon, select any random legal action.
    """
    if random.random() < epsilon:
        return select_random_action(legal_actions)

    action_values = model(torch.tensor(state, dtype=torch.float32))

    # Mask unavailable moves (by setting the values to NaN)
    action_mask = np.full_like(action_values, fill_value=np.nan)
    action_mask[np.array(legal_actions)] = 1.
    action_values *= action_mask

    # Greedily return the best-value move for the current player
    if current_player == Player.WHITE:
        best_action = int(np.nanargmax(action_values))
    else:
        best_action = int(np.nanargmin(action_values))

    return best_action


def select_random_action(legal_actions: list[Action]) -> Action:
    return random.choice(legal_actions)


def play_random_moves_opponent(model: torch.nn.Module,
                               game: Game,
                               epsilon: float) -> tuple[bool, bool]:
    """
    Play a game using the model against an opponent playing random moves only.

    Returns two (mutually exclusive) booleans indicating whether the model player won,
    and whether the game was a draw.
    """
    # Randomly select a side
    model_player = random.choice([Player.WHITE, Player.BLACK])
    game.reset()

    while True:
        state = game.get_state()
        current_player = game.get_current_player()
        legal_actions = game.get_legal_actions()

        if current_player == model_player:
            action = select_action(model, state, current_player, legal_actions, epsilon)
        else:
            action = select_random_action(legal_actions)
        game.play(action)

        if game.has_ended():
            return game.get_winner() == model_player, game.get_winner() == Player.NEUTRAL


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


def make_batch(samples: list[Sample]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Turn a batch from the replay buffer into something usable by pytorch modules
    """
    states: list[GameState] = []
    actions: list[Action] = []
    winners: list[Literal[-1, 0, 1]] = []
    for sample in samples:
        states.append(sample.game_state)
        actions.append(sample.action)
        winners.append(sample.winner.value)

    return torch.tensor(np.array(states, dtype=np.float32)), \
        torch.tensor(np.array(actions, dtype=np.int64)), \
        torch.tensor(np.array(winners))


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


def fill_replay_buffer(replay_buffer: ReplayBuffer,
                       model: torch.nn.Module,
                       game: Game,
                       epsilon: float,
                       num_samples: int | None) -> int:
    """
    Fill the given replay buffer with num_samples samples,
    or fill it completely if num_samples is None
    """
    def is_done():
        if num_samples is None:
            return replay_buffer.is_full
        return len(replay_buffer) >= num_samples

    progress_bar = tqdm.tqdm(total=num_samples or replay_buffer.capacity,
                             dynamic_ncols=True,
                             desc='Filling replay buffer')
    games_played = 0
    while not is_done():
        games_played += 1
        game_history, winner = self_play_game(model, game, epsilon=epsilon)
        replay_buffer.append(game_history, winner)
        progress_bar.update(len(game_history))
        progress_bar.set_postfix(dict(games=games_played))

    return games_played


def train_loop(model: torch.nn.Module,
               learning_rate: float,  # Add the learning rate as an argument
               self_play_games_per_iter: int,  # Add the self-play games per iteration as an argument
               weight_decay: float) -> list[dict]:
    # Generate games into replay buffer
    buffer_size = 100_000
    buffer_initial_samples: int | None = 10_000

    game = Game()
    replay_buffer = ReplayBuffer(buffer_size)
    selfplay_games = fill_replay_buffer(replay_buffer,
                                        model,
                                        game,
                                        epsilon=1.0,
                                        num_samples=buffer_initial_samples)

    loss_fn = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    loss_ema = SmoothedAverage(initial=1.)
    winrate_ema = SmoothedAverage(alpha=0.1)
    loserate_ema = SmoothedAverage(alpha=0.1)
    drawrate_ema = SmoothedAverage(alpha=0.1)

    num_iterations = 100
    epsilon_anneal_range = 100
    min_epsilon = 0.2

    self_play_games_per_iter = 25
    batches_per_iter = 100
    batch_size = 128

    validation_epsilon = 0.05
    random_validation_games = 10

    train_history = []
    progress_bar = tqdm.trange(num_iterations, dynamic_ncols=True, colour='#22ee22')
    for iteration in progress_bar:

        losses = []

        # The model learn to predict the game winner given a state and action
        for _ in tqdm.trange(batches_per_iter, position=1, leave=False,
                             desc='Learning from past experience'):
            optimizer.zero_grad()

            states, actions, winners = make_batch(replay_buffer.sample_k(batch_size))
            predictions = model(states)  # (batch, num_actions)
            action_mask = torch.nn.functional.one_hot(actions, num_classes=Game.NUM_ACTIONS)
            
            action_mask = action_mask.type(torch.float32)  # (batch, num_actions)
            predictions = predictions * action_mask
            winners = winners.unsqueeze(-1) * action_mask  # (batch, num_actions)

            loss = loss_fn(predictions, winners) / batch_size
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
        
        loss_ema.update(sum(losses) / len(losses))

        # Play vs random opponent to validate performance
        winrate, drawrate = evaluate_model_vs_random(model,
                                                     epsilon=validation_epsilon,
                                                     num_games=random_validation_games)
        winrate_ema.update(winrate)
        drawrate_ema.update(drawrate)
        loserate_ema.update(1 - winrate - drawrate)

        # Play vs self to generate training data
        epsilon = max(min_epsilon, (epsilon_anneal_range - iteration) / epsilon_anneal_range)
        for _ in tqdm.trange(self_play_games_per_iter, position=1, leave=False,
                             desc='Playing against self'):
            replay_buffer.append(*self_play_game(model, game, epsilon=epsilon))
            selfplay_games += 1

        iteration_hist = dict(iteration=iteration,
                              loss=loss_ema.unsmoothed_value,
                              win=winrate_ema.unsmoothed_value,
                              lose=loserate_ema.unsmoothed_value,
                              draw=drawrate_ema.unsmoothed_value,
                              epsilon=epsilon,
                              selfplay_games=selfplay_games)

        progress_bar.set_postfix_str(f'loss={loss_ema.value:.2f} ' + 
                                     f'W={winrate_ema.value:.2f} ' +
                                     f'L={loserate_ema.value:.2f} ' +
                                     f'D={drawrate_ema.value:.2f} ' +
                                     f'eps={epsilon:.2f} ' +
                                     f'games={selfplay_games}')
        train_history.append(iteration_hist)

    return train_history

            
@torch.no_grad()
def evaluate_model_vs_random(model: torch.nn.Module, epsilon: float, num_games: int):
    game = Game()
    player_wins = 0
    draws = 0

    for i in tqdm.trange(num_games, position=1, leave=False, desc='Playing against random moves'):
        player_won, draw = play_random_moves_opponent(model, game, epsilon)
        print(f"Game {i + 1}: Player Won = {player_won}, Draw = {draw}")
        if player_won:
            player_wins += 1
        if draw:
            draws += 1

    win_rate = player_wins / num_games
    draw_rate = draws / num_games

    return -win_rate, draw_rate


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
            hidden_size=self.hidden_size,
            output_size=Game.NUM_ACTIONS
        )

        # Set other parameters as needed

        # Train the model and get the metric to optimize (e.g., average win rate)
        self.train_hist = train_loop(self.model, learning_rate=self.learning_rate,
                                     self_play_games_per_iter=self.self_play_games_per_iter,
                                     weight_decay=self.weight_decay)
        # Return the metric to optimize (e.g., average win rate)
        return self

    def score(self, X, y=None, train_hist=None):  # Pass train_hist as an argument
        if train_hist is None:
            train_hist = self.train_hist  # Use self.train_hist if not provided

        # Extract relevant information from the training history
        num_wins = sum(iteration.get('num_wins', 0) for iteration in train_hist)
        num_losses = sum(iteration.get('num_losses', 0) for iteration in train_hist)

        # Calculate the average win rate
        total_games = num_wins + num_losses
        if total_games > 0:
            average_win_rate = num_wins / total_games
        else:
            average_win_rate = 0.0  # Default if no games have been played

        return average_win_rate

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
        'self_play_games_per_iter': Integer(10, 100),
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
        hidden_size=best_hyperparams['hidden_size'],
        output_size=Game.NUM_ACTIONS
    )

    # Train the model using the best hyperparameters
    train_hist = train_loop(
        best_model,
        learning_rate=best_hyperparams['learning_rate'],
        self_play_games_per_iter=best_hyperparams['self_play_games_per_iter'],
        weight_decay=best_hyperparams['weight_decay']
    )

    # Print or save the training history
    print(train_hist)

    # Evaluate the model against random moves with the best hyperparameters
    win_rate, draw_rate = evaluate_model_vs_random(best_model, epsilon=0.05, num_games=10)
    print(f'Final Win Rate: {win_rate}, Draw Rate: {draw_rate}')
