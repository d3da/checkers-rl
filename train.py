#!/bin/env python3

from collections import Counter, deque
from typing import NamedTuple, Literal, Any
import os
import torch 
import numpy as np
import tqdm
import random
import math
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


from model import CheckersQModel
from game import Game, GameState, Player, Action, NUM_ACTIONS
from agent import BaseAgent, QModelAgent, RandomAgent, UserInputAgent


class Sample(NamedTuple):
    game_state: GameState
    action: Action
    current_player: Player
    winner: Player


class ReplayBuffer:
    """
    Store tuples of gamestate, action, current player and the game winner,
    which can be used by the model to (hopefully) learn something from past experience
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
    states: list[np.ndarray] = []
    actions: list[Action] = []
    winners: list[Literal[-1, 0, 1]] = []
    for sample in samples:
        states.append(sample.game_state.to_array())
        actions.append(sample.action)
        winners.append(sample.winner.value)

    return torch.tensor(np.array(states, dtype=np.float32)), \
            torch.tensor(np.array(actions, dtype=np.int64)), \
            torch.tensor(np.array(winners))


class SmoothedAverage:
    """
    Class implementing an exponential moving average (EMA)
    https://en.wikipedia.org/wiki/Exponential_smoothing

    Mainly so the progress meters don't jump all over the place and they are a bit easier to interpret
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


def fill_replay_buffer_random_moves(replay_buffer: ReplayBuffer,
                                    game: Game,
                                    agent: BaseAgent,
                                    agent_kwargs: dict[str, Any] | None = None,
                                    num_samples: int | None = None) -> int:
    """
    Fill the given replay buffer with num_samples samples,
    or fill it completely if num_samples is None
    """
    agent_kwargs = agent_kwargs or {}
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
        _, winner, game_history = play_agent_game(game, agent, agent, agent_kwargs, agent_kwargs)
        replay_buffer.append(game_history, winner)
        progress_bar.update(len(game_history))
        progress_bar.set_postfix(dict(games=games_played))

    return games_played


def train_loop(model: CheckersQModel):
    # Generate games into replay buffer
    buffer_size = 1000
    buffer_initial_samples: int | None = 100

    game = Game()
    rl_agent = QModelAgent(model)
    random_agent = RandomAgent()

    replay_buffer = ReplayBuffer(buffer_size)
    selfplay_games = fill_replay_buffer_random_moves(replay_buffer,
                                                     game,
                                                     random_agent,
                                                     num_samples=buffer_initial_samples)

    loss_fn = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=5.0)

    loss_ema = SmoothedAverage(initial=1.)
    winrate_ema = SmoothedAverage(alpha=0.1)
    loserate_ema = SmoothedAverage(alpha=0.1)
    drawrate_ema = SmoothedAverage(alpha=0.1)

    num_iterations = 3
    epsilon_anneal_range = 100
    min_epsilon = 0.2

    self_play_games_per_iter = 5
    batches_per_iter = 10
    batch_size = 128

    validation_epsilon = 0.05
    random_validation_games = 5

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
            action_mask = torch.nn.functional.one_hot(actions, num_classes=NUM_ACTIONS)
            
            action_mask = action_mask.type(torch.float32)  # (batch, num_actions)
            predictions = predictions * action_mask
            winners = winners.unsqueeze(-1) * action_mask  # (batch, num_actions)

            loss = loss_fn(predictions, winners) / batch_size
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
        
        loss_ema.update(sum(losses) / len(losses))

        # Play vs random opponent to validate performance
        winrate, drawrate = evaluate_model_vs_random(rl_agent,
                                                     random_agent,
                                                     epsilon=validation_epsilon,
                                                     num_games=random_validation_games)
        winrate_ema.update(winrate)
        drawrate_ema.update(drawrate)
        loserate_ema.update(1 - winrate - drawrate)

        # Play vs self to generate training data
        epsilon = max(min_epsilon, (epsilon_anneal_range - iteration) / epsilon_anneal_range)
        for _ in tqdm.trange(self_play_games_per_iter, position=1, leave=False,
                             desc='Playing against self'):
            agent_kwargs = dict(epsilon=epsilon)
            _, winner, history = play_agent_game(game, rl_agent, rl_agent, agent_kwargs, agent_kwargs)
            replay_buffer.append(history, winner)
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


def play_agent_game(game: Game,
                    agent_a: BaseAgent,
                    agent_b: BaseAgent,
                    agent_a_kwargs: dict[str, Any] | None = None,
                    agent_b_kwargs: dict[str, Any] | None = None) \
                            -> tuple[Player, Player, list[tuple[GameState, Action, Player]]]:
    agent_a_kwargs = agent_a_kwargs or {}
    agent_b_kwargs = agent_b_kwargs or {}

    agent_a_player = random.choice([Player.BLACK, Player.WHITE])
    game_history: list[tuple[GameState, Action, Player]] = []
    game.reset()

    while True:
        state = game.get_state()
        current_player = game.get_current_player()

        agent = agent_a if current_player == agent_a_player else agent_b
        kwargs = agent_a_kwargs if current_player == agent_a_player else agent_b_kwargs

        action = agent.select_action(game=game, **kwargs)
        game.play(action)
        game_history.append((state, action, current_player))

        if game.has_ended():
            return agent_a_player, game.get_winner(), game_history

def save_model(model):
    # Save model
    model_path = 'model/model.pth'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at: {model_path}")

def load_model(model):
    model_path = 'model/model.pth'
    try:
        model.load_state_dict(torch.load(model_path))
        print(f"Model loaded from: {model_path}")
    except FileNotFoundError:
        print(f"No model found at: {model_path}. Training from scratch.")

if __name__ == '__main__':

    # Setup
    model = CheckersQModel(num_hidden_layers=0, hidden_size=512)
    load_model(model)  # load previously saved model, or from scratch

    # training loop
    while True:
        train_hist = train_loop(model)
        save_model(model)

        model_agent = QModelAgent(model)
        user_agent = UserInputAgent()

        #play_agent_game(Game(), model_agent, user_agent, dict(epsilon=0.05))

        df = pd.DataFrame(train_hist)

        # Save training results
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        results_dir = f'results/{timestamp}/'
        os.makedirs(results_dir, exist_ok=True)
        csv_filename = f'train_results_{timestamp}.csv'
        df.to_csv(os.path.join(results_dir, csv_filename))

        # PLOTTING

        # Convert win and loss rates to percentages
        df['win_percent'] = df['win'] * 100
        df['lose_percent'] = df['lose'] * 100

        # Figure formatting
        plt.figure(figsize=(10, 6))  # Set the figure size
        plt.plot(df['iteration'], df['win_percent'], label='Win Rate', color='green', marker='o')
        plt.plot(df['iteration'], df['lose_percent'], label='Loss Rate', color='red', marker='o')
        plt.axhline(0, color='black', linestyle='--', linewidth=1)
        plt.axhline(100, color='black', linestyle='--', linewidth=1)
        plt.xlabel('Iteration')
        plt.ylabel('Win/Loss Rate (%)')
        plt.title('Win-Loss Ratio Over Iterations')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Ask for another training loop
        user_input = input("Do you want to start another training loop? (y/n): ").lower()
        if user_input != 'y':
            print("Finished training.")
            break  # Exit the loop if the user enters anything other than 'y'
