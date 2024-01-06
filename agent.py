from abc import ABC, abstractmethod
import torch
import math
import numpy as np
import random
import heuristics
from typing import Any
from game import Action, Player, Game, GameState
from model import CheckersQModel, CheckersVModel

class BaseAgent(ABC):
    """
    Base class that defines an agent. All different agents inherit from this one.
    An agent selects an Action given a Game, by calling the select_action function.
    How it selects the action is defined in the agent class, by overriding _select_action.
    """

    def __init__(self, *required_kwarg_names: str) -> None:
        """
        Subclasses may define strings here that are required as kwargs in the call to _select_action.
        For example, an epsilon-greedy agent may define the kwarg 'epsilon' to be required here.
        The required kwargs are checked every call to select_action.
        """
        self.required_kwarg_names = required_kwarg_names

    def _check_kwargs(self, **kwargs):
        """
        Check if the required_kwarg_names are passed as keys to the select_action call.
        """
        if __debug__:
            for key in self.required_kwarg_names:
                assert key in kwargs.keys(), f'Agent {type(self).__name__} requires kwarg {key}'

    def select_action(self,
                      game: Game,
                      **kwargs) -> Action:
        """
        Method to be called by other code.
        Returns an action selected by the agent.
        """
        self._check_kwargs(**kwargs)
        return self._select_action(game, **kwargs)

    @abstractmethod
    def _select_action(self,
                       game: Game,
                       **kwargs) -> Action:
        """
        This method should be overridden by subclasses to define how to select an action.
        """
        raise NotImplementedError


class BaseEpsilonAgent(BaseAgent, ABC):
    """
    Base class for defining an epsilon-greedy strategy.
    That is, a random action is chosen with chance epsilon,
    and a different strategy is used otherwise.
    """

    def __init__(self, *required_kwarg_names: str) -> None:
        # Ensure that a value for epsilon is passed or an error is thrown.
        super().__init__('epsilon', *required_kwarg_names)

    def _select_action(self,
                       game: Game,
                       **kwargs) -> Action:
        self._check_kwargs(**kwargs)
        if random.random() < kwargs['epsilon']:
            return random.choice(game.get_legal_actions())
        return self._select_nonepsilon_action(game, **kwargs)

    @abstractmethod
    def _select_nonepsilon_action(self,
                                  game: Game,
                                  **kwargs) -> Action:
        raise NotImplementedError


class RandomAgent(BaseAgent):
    """
    Simple agent that selects purely random moves.
    """

    def _select_action(self,
                       game: Game,
                       **kwargs) -> Action:
        return random.choice(game.get_legal_actions())


class QModelAgent(BaseEpsilonAgent):
    """
    Make the model select an action
    using an epsilon-greedy strategy w.r.t. the learned Q-function

    In other words, compute the estimated win probability for every action
    and select the legal action with highest value.
    Or, with a chance of epsilon, select any random legal action.
    """

    def __init__(self, model: CheckersQModel):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def _select_nonepsilon_action(self,
                                  game: Game,
                                  **kwargs) -> Action:
        action_values = self.model(torch.tensor(game.get_state().to_array(), dtype=torch.float32))

        # Mask unavailable moves (by setting the values to NaN)
        action_mask = np.full_like(action_values, fill_value=np.nan)
        action_mask[np.array(game.get_legal_actions())] = 1.
        action_values *= action_mask

        # Greedily return the best-value move for the current player
        if game.get_current_player() == Player.WHITE:
            best_action = int(np.nanargmax(action_values))
        else:
            best_action = int(np.nanargmin(action_values))

        return best_action


class BaseTreeSearchAgent(BaseAgent, ABC):
    """
    Agent that performs tree search (a single step for now)
    and picks a move that maximizes/minimizes a certain heuristic function.

    TODO: Remove this (superseded by minimax tree search)
    """

    def _select_action(self,
                       game: Game,
                       **kwargs) -> Action:
        scores = []
        legal_actions = game.get_legal_actions()
        for action in legal_actions:
            game.play(action)
            score = self._heuristic(game)
            scores.append(score)
            game.undo()

        best_score = max(scores) if game.get_current_player() == Player.WHITE else min(scores)
        action_idx = random.choice([index for index, value in enumerate(scores) if value == best_score])
        return legal_actions[action_idx]

    @abstractmethod
    def _heuristic(self, game: Game) -> float:
        """
        To be overridden by subclasses.
        The heuristic should return higher values for better posisions for the white player,
        and should return lower values for positions that favor black more.
        """
        raise NotImplementedError


class BaseEpsilonTreeSearchAgent(BaseEpsilonAgent, BaseTreeSearchAgent, ABC):
    """
    An agent that picks a random move with chance epsilon (see BaseEpsilonAgent)
    and otherwise performs tree search (see BaseTreeSearchAgent) using the
    _heuristic function implemented by a subclass.
    """

    def __init__(self, *required_kwarg_names: str) -> None:
        super().__init__(*required_kwarg_names)
        assert 'epsilon' in self.required_kwarg_names

    def _select_action(self, game: Game, **kwargs) -> Action:
        return super()._select_action(game, **kwargs)

    def _select_nonepsilon_action(self, game: Game, **kwargs) -> Action:
        # This calls the BaseTreeSearchAgent's _select_action
        # (since it comes after BaseEpsilonAgent in self.__class__.__mro__)
        return super(BaseEpsilonAgent, self)._select_action(game, **kwargs)


class VModelAgent(BaseEpsilonTreeSearchAgent):
    """
    Use a neural network that estimates the state-value function as a heuristic,
    and perform tree search using this heuristic.

    TODO: calculate all moves in one batch?
    """

    def __init__(self, model: CheckersVModel):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def _heuristic(self, game: Game) -> float:
        return self.model(torch.tensor(game.get_state().to_array(), dtype=torch.float32))


class UserInputAgent(BaseAgent):
    """
    Use sys.stdin to select an action.
    """

    def _select_action(self,
                       game: Game,
                       **kwargs) -> Action:
        self._check_kwargs(**kwargs)

        legal_actions = game.get_legal_actions()
        legal_actions_humanreadable = map(game._action_to_steps_move, legal_actions)
        while True:
            print(game)
            print(f'Legal actions: {list(enumerate(legal_actions_humanreadable))}')
            print('type the action index >', end='')
            next_move = input()
            try:
                action = legal_actions[int(next_move)]
            except Exception:
                continue
            break
        return action


class BaseMinMaxSearchAgent(BaseTreeSearchAgent):
    def __init__(self, *required_kwarg_names: str) -> None:
        super().__init__('depth', *required_kwarg_names)

    def _select_action(self, game: Game, **kwargs) -> Action:
        _, best_action = self.minimax(game, kwargs['depth'])
        assert best_action is not None
        return best_action

    def minimax(self, game: Game, depth: int, alpha: float = -math.inf, beta: float = math.inf) \
            -> tuple[float, Action | None]:
        """
        Minimax algorithm
        Returns the score and the best action to play according to the _heuristic function

        TO DO:
        - Problem with moves that are evaluated the same: the model will have a tendency to play a little too much on the left (or right) 
          --> implement some randomness
        """
        max_player = game.get_current_player() == Player.WHITE

        if depth == 0 or game.has_ended():
            return self._heuristic(game), None

        best_eval = -math.inf if max_player else math.inf
        best_action = None
        for action in game.get_legal_actions():
            game.play(action)
            eval, _ = self.minimax(game, depth - 1, alpha=alpha, beta=beta)
            game.undo()

            if max_player:
                best_eval = max(best_eval, eval)
                alpha = max(alpha, eval)
            else:
                best_eval = min(best_eval, eval)
                beta = min(beta, eval)

            if best_eval == eval:
                best_action = action
            if beta <= alpha:
                break

        return best_eval, best_action


class PieceCountHeuristicsAgent(BaseMinMaxSearchAgent):
    """
    Select an action using tree search with the piece_count function
    as a heuristic.
    """

    def _heuristic(self, game: Game) -> float:
        return heuristics.piece_count(game)


def play_agent_game(game: Game,
                    agent_a: BaseAgent,
                    agent_b: BaseAgent,
                    agent_a_kwargs: dict[str, Any] | None = None,
                    agent_b_kwargs: dict[str, Any] | None = None,
                    max_num_moves: int | None = None) \
                            -> tuple[Player, Player, list[tuple[GameState, Action, Player]]]:
    """
    Play a single game between two agents.
    Some agents require additional options (kwargs), and these can be supplied
    with the agent_*_kwargs parameter.
    Optionally, the length of the game can be limited using the max_num_moves parameter.
    The black and white pieces are given randomly to either player.

    Returns:
        - The player color of agent_a
        - The winning player (Player.NEUTRAL in case of a draw)
        - The game history as list of (gamestate, action, current_player) tuples.
    """
    agent_a_kwargs = agent_a_kwargs or {}
    agent_b_kwargs = agent_b_kwargs or {}

    agent_a_player = random.choice([Player.BLACK, Player.WHITE])
    game_history: list[tuple[GameState, Action, Player]] = []
    game.reset()

    num_moves: int = 0
    while True:
        num_moves += 1

        state = game.get_state()
        current_player = game.get_current_player()

        agent = agent_a if current_player == agent_a_player else agent_b
        kwargs = agent_a_kwargs if current_player == agent_a_player else agent_b_kwargs

        action = agent.select_action(game=game, **kwargs)
        game.play(action)
        game_history.append((state, action, current_player))

        # Check if game has ended
        if game.has_ended():
            return agent_a_player, game.get_winner(), game_history

        # Stop playing if max_num_moves is reached
        if max_num_moves and num_moves > max_num_moves:
            print('Reached max num moves')
            return agent_a_player, Player.NEUTRAL, game_history

