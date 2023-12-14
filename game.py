import numpy as np
from enum import Enum
from draughts import Board, Move
import draughts

GameState = np.ndarray
"""
A GameState should be an 8x8 array filled with Player.WHITE and Player.BLACK for the pieces
and Player.NEUTRAL for empty spaces.
"""

Action = int
"""
An Action is an integer that corresponds to a single move, ranging from 0 to Game.NUM_ACTIONS
"""

class Player(Enum):
    WHITE = 1
    NEUTRAL = 0
    BLACK = -1

class Game:
    """
    Class to handle the logic for playing a game of checkers,
    and returning info about the game state and available moves
    to be used during self-play.
    """

    VARIANT = 'standard'

    BOARD_SIZE = 50
    """
    TODO: Include information about jump-moves in the state
    when the player is currently in a jump chain
    """

    NUM_ACTIONS = BOARD_SIZE ** 2
    """
    The first 32 selects the "from" square, the last 32 selects the "to" square
    So the most commong opening move 11-15 is represented as the number 11*32+15
    """

    def __init__(self) -> None:
        """
        Initialize a game board
        """
        self.reset()

    def reset(self) -> None:
        """
        Reset the board to its initial position.
        """
        self.board = Board(variant=self.VARIANT)
        self.move_cache: list[list[int]] = []
        self.current_jump: list[int] = []

    def get_state(self) -> GameState:
        """
        Return the game's state in a format that is usable by the NN as input.
        """
        def index_to_player(i: int, board: Board) -> Player:
            piece = board._game.board.searcher.position_pieces.get(i)
            if piece is None:
                return Player.NEUTRAL
            elif piece.player == 1:
                return Player.BLACK
            elif piece.player == 2:
                return Player.WHITE
            raise Exception

        return np.array([index_to_player(i, self.board).value for i in range(1,33)])

    def get_legal_actions(self) -> list[Action]:
        """
        Get a list of action indices corresponding to all the legal actions in the current position.
        """
        if len(self.current_jump) == 0:
            self.move_cache = [move.steps_move for move in self.board.legal_moves()]

        return [self._steps_move_to_action(steps_move) for steps_move in self.move_cache]

    def _action_to_steps_move(self, action: Action) -> list[int]:
        """Convert a model output action index to something usable by pydraughts"""
        to_pos = action % self.BOARD_SIZE
        from_pos = int((action - to_pos) / self.BOARD_SIZE)
        to_pos += 1
        from_pos += 1
        return [from_pos, to_pos]

    def _steps_move_to_action(self, steps_move: list[int]) -> Action:
        """Convert a pydraughts move to something usable by the model"""
        return (steps_move[0] - 1) * self.BOARD_SIZE + steps_move[1] - 1

    def play(self, action: Action) -> None:
        """
        Play a certain action on the board.
        Assumes that the action is legal, so ensure this first!
        """
        steps_move = self._action_to_steps_move(action)
        if steps_move not in self.move_cache:
            # Move is part of a jump chain
            new_move_cache: list[list[int]] = []
            for start_pos, end_pos, *further_jumps in self.move_cache:
                if [start_pos, end_pos] == steps_move:
                    new_move_cache.append([end_pos, *further_jumps])

            self.move_cache = new_move_cache
            self.current_jump.append(steps_move[0])
            return

        steps_move = self.current_jump + steps_move

        move = Move(self.board, steps_move=steps_move)
        self.board.push(move)

        self.current_jump = []
        self.move_cache = []

    def has_ended(self) -> bool:
        """
        Check if the game has ended.
        """
        return self.board.is_over()
    
    @staticmethod
    def _convert_player(draughts_player: int | None) -> Player:
        """Convert from pydraught player to our Player class"""
        if draughts_player == draughts.WHITE:
            return Player.WHITE
        elif draughts_player == draughts.BLACK:
            return Player.BLACK
        return Player.NEUTRAL


    def get_winner(self) -> Player:
        """
        If the game has ended, return the player who won (or NEUTRAL for a drawn game)
        """
        return self._convert_player(self.board.winner())

    def get_current_player(self) -> Player:
        """
        Return the player who is currently to move.
        Note: the same player may get multiple moves in a row
        """
        return self._convert_player(self.board.turn)

    def __str__(self) -> str:
        """
        Return a human-readable string that shows the board position and whose move it is
        """
        return self.board.__str__() + f'\n{self.get_current_player()._name_} to play'


if __name__ == '__main__':
    game = Game()
    print(game)
    print(game.get_state())
