#!/bin/env python3
import os
from abc import ABC, abstractmethod
from collections import Counter, deque
from typing import NamedTuple, Literal, Any
import torch 
import numpy as np
import tqdm
import random
import math
import pandas as pd

from model import CheckersQModel, CheckersVModel
from game import Game, GameState, Player, Action, NUM_ACTIONS
from agent import play_agent_game, BaseAgent, QModelAgent, RandomAgent, VModelAgent


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
              num_train_iterations: int = 50,
              selfplay_games_p_i: int = 10,
              train_batches_p_i: int = 100,
              batch_size: int = 128,
              epsilon_anneal_iters: int = 40,
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

    def _build_agent_kwargs_train(self, **kwargs):
        return kwargs

    def _build_agent_kwargs_eval(self, **kwargs):
        return kwargs

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
            train_kwargs = self._build_agent_kwargs_train(epsilon=epsilon)

            if iteration > 0:
                self._generate_selfplay_experience(selfplay_games_p_i,
                                                   train_kwargs,
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
                          disable_progress: bool = False) -> tuple[float, float, float]:
        """
        Play multiple games using this model vs a given enemy agent.
        Returns (winrate, drawrate, lossrate) from the model agent's point of view
        """
        enemy_agent = enemy_agent or RandomAgent()
        player_wins = draws = losses = 0
        progress_bar = tqdm.trange(num_evaluation_games, position=0, leave=False,
                                   desc='Evaluating playing strength', disable=disable_progress)
        for _ in progress_bar:
            eval_kwargs = self._build_agent_kwargs_eval(epsilon=evaluation_epsilon)
            rl_player, winner, _ = play_agent_game(self.game,
                                                   self.model_agent, enemy_agent,
                                                   eval_kwargs,
                                                   enemy_agent_kwargs)
            if winner == rl_player:
                player_wins += 1
            elif winner == Player.NEUTRAL:
                draws += 1
            else:
                losses += 1
            progress_bar.set_postfix_str(f'wins={player_wins}, draws={draws}, losses={losses}')

        winrate = player_wins / num_evaluation_games
        drawrate = draws / num_evaluation_games
        lossrate = losses / num_evaluation_games
        return winrate, drawrate, lossrate

    def save_model(self):
        model_path = 'model/model.pth'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Save model
        torch.save(self.model, model_path)
        print(f"Model saved at: {model_path}")
    def load_model(self):
        model_path = 'model/model.pth'
        try:
            # Load model
            self.model.load_state_dict(torch.load(model_path))
            print(f"Model loaded from: {model_path}")
        except FileNotFoundError:
            print(f"No model found at: {model_path}. Training model from scratch.")

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
    def __init__(self,
                 model: CheckersVModel,
                 optimizer: torch.optim.Optimizer,
                 train_search_depth: int = 2,
                 eval_search_depth: int = 3) -> None:
        super().__init__(VModelAgent(model), optimizer)
        self.model = model
        self.train_search_depth = train_search_depth
        self.eval_search_depth = eval_search_depth

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

    def _build_agent_kwargs_train(self, **kwargs):
        return dict(depth=self.train_search_depth) | super()._build_agent_kwargs_train(**kwargs)

    def _build_agent_kwargs_eval(self, **kwargs):
        return dict(depth=self.eval_search_depth) | super()._build_agent_kwargs_eval(**kwargs)



if __name__ == '__main__':
    model = CheckersVModel(num_hidden_layers=1, hidden_size=1024)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-5)
    trainrun = VModelTrainRun(model, optimizer)
    # Shortened training for testing
    train_hist = trainrun.train()
    trainrun.save_model()
    print(train_hist)

    wr, dr, lr = trainrun.evaluate_strength(100, 0.05)
    print(f'win: {wr}, draw: {dr}, loss: {lr}')

