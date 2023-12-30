from abc import ABC, abstractmethod
import torch
import numpy as np
import random
import heuristics
from game import Action, Player, Game
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
        return super(BaseEpsilonAgent)._select_action(game, **kwargs)


class VModelAgent(BaseEpsilonTreeSearchAgent):
    """
    Use a neural network that estimates the state-value function as a heuristic,
    and perform tree search using this heuristic.
    """

    def __init__(self, model: CheckersVModel):
        super().__init__()
        self.model = model

    def _heuristic(self, game: Game) -> float:
        return self.model(game.get_state())


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


class PieceCountHeuristicsAgent(BaseTreeSearchAgent):
    """
    Select an action using tree search with the piece_count function
    as a heuristic.
    """

    def _heuristic(self, game: Game) -> float:
        return heuristics.piece_count(game)

