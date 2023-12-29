import os
import numpy as np
from enum import Enum
from draughts import Board, Move
import draughts

Action = int
"""
An Action is an integer that corresponds to a single move, ranging from 0 to Game.NUM_ACTIONS
"""

class Player(Enum):
    WHITE = 1
    NEUTRAL = 0
    BLACK = -1


BOARD_SIZE = 50

VARIANT = 'standard'

NUM_ACTIONS = BOARD_SIZE ** 2
"""
The first 50 selects the "from" square, the last 50 selects the "to" square
So the move 11-15 is represented as the number 11*50+15
"""

GAMESTATE_SIZE = BOARD_SIZE * 2

class GameState:
    """
    TODO:
     - include the legal actions and current player in the gamestate class?
     - include the number of current repetitions? To avoid draw by repetition?
    """

    def __init__(self) -> None:
        self.piece_colors = np.full((BOARD_SIZE), Player.NEUTRAL.value)
        self.kings        = np.full((BOARD_SIZE), False)

    def _update_square(self, sq: int, player: Player, king: bool) -> None:
        self.piece_colors[sq - 1] = player.value
        self.kings[sq - 1] = king

    def _get_square(self, sq: int) -> tuple[Player, bool]:
        return Player(self.piece_colors[sq - 1]), self.kings[sq - 1]

    def to_array(self) -> np.ndarray:
        """
        Return an array of shape (GAMESTATE_SIZE,) containing the board position,
        to be used as input for a neural network.
        """
        kings_repr = self.kings.astype(int) * self.piece_colors
        return np.concatenate([self.piece_colors, kings_repr])

    def __str__(self) -> str:
        return str(self.piece_colors) + os.linesep + str(self.kings)


class Game:
    """
    Class to handle the logic for playing a game of checkers,
    and returning info about the game state and available moves
    to be used during self-play.
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
        self.board = Board(variant=VARIANT)
        self.move_cache: list[list[int]] = []
        self.current_jump: list[int] = []
        self.current_jump_end_pos: int | None = None

        self._move_cache_history: list[list[list[int]]] = []
        self._current_jump_history: list[list[int]] = []
        self._move_was_jump_chain_segment_history: list[bool] = []

    def get_state(self) -> GameState:
        """
        Return the game's state in a format that is usable by the NN as input.
        When the player is currently within an incomplete jump chain, we manually update the
        gamestate to reflect the position within the jump chain.
        """
        def piece_player(piece) -> Player:
            if piece is None:
                return Player.NEUTRAL
            elif piece.player == 1:
                return Player.BLACK
            elif piece.player == 2:
                return Player.WHITE
            raise Exception

        game_state = GameState()
        for i in range(1, BOARD_SIZE + 1):
            piece = self.board._game.board.searcher.position_pieces.get(i)
            king = False if piece is None else piece.king
            game_state._update_square(i, piece_player(piece), king)

        return self._add_jump_chain_to_state(game_state)

    def _add_jump_chain_to_state(self, game_state: GameState) -> GameState:
        """
        Check if game is inside an incomplete jump chain and manually update the gamestate,
        moving a piece along the jump chain squares and capturing pieces it jumps over.
        """
        if len(self.current_jump) == 0:
            assert self.current_jump_end_pos is None
            return game_state

        assert self.current_jump_end_pos is not None
        full_jump_chain: list[int] = self.current_jump + [self.current_jump_end_pos]
        while len(full_jump_chain) > 1:
            from_pos, to_pos, *_ = full_jump_chain
            full_jump_chain.pop(0)

            jumping_piece, jumping_king = game_state._get_square(from_pos)
            captured_pos = self._get_captured_piece_square(from_pos, to_pos, game_state)
            assert game_state._get_square(captured_pos)[0] != Player.NEUTRAL
            assert game_state._get_square(captured_pos)[0].value == -jumping_piece.value

            # Set the departure square to empty
            game_state._update_square(from_pos, Player.NEUTRAL, False)
            # Set the arrival square to the piece
            game_state._update_square(to_pos, jumping_piece, jumping_king)
            # Capture the piece in the middle
            game_state._update_square(captured_pos, Player.NEUTRAL, False)

        return game_state

    @staticmethod
    def _get_captured_piece_square(from_pos, to_pos, game_state: GameState) -> int:
        """
        Given the squares that a piece jumps from and to, return the square that is captured.
        This function only works when BOARD_SIZE is 50.
        """
        if BOARD_SIZE != 50:
            print('WARNING: game state logic currently only works for BOARD_SIZE == 50.')
            raise NotImplementedError

        def convert_square(square):
            """
            Convert squares from a system where only the black squares are counted
            (normal checkers notation) to counting both black and white squares,
            since this representation has nicer properties we can use.
            Namely, a step to the bottom-right / top-left always does +/- 11,
                and a step to the top-right / bottom-left always does +/- 9.
            """
            return 2 * square if square % 10 in [1,2,3,4,5] else 2 * square - 1

        def convert_back(square):
            """Convert back to standard form"""
            return square // 2 if square % 2 == 0 else (square + 1) // 2

        diff = convert_square(to_pos) - convert_square(from_pos)
        direction_down = diff > 0
        sign = 1 if direction_down else -1
        sw_ne_movement = (sign * diff) % 11 == 0
        assert sw_ne_movement or (sign * diff) % 9 == 0
        step = sign * (11 if sw_ne_movement else 9)

        # Find the square of the first piece along the move diagonal
        search_pos = convert_square(from_pos)
        while True:
            search_pos += step
            search_square = convert_back(search_pos)
            if game_state._get_square(search_square)[0] != Player.NEUTRAL:
                return search_square

    def get_legal_actions(self) -> list[Action]:
        """
        Get a list of action indices corresponding to all the legal actions in the current position.
        """
        if len(self.current_jump) == 0:
            self.move_cache = [move.steps_move for move in self.board.legal_moves()]

        return [self._steps_move_to_action(steps_move) for steps_move in self.move_cache]

    def _action_to_steps_move(self, action: Action) -> list[int]:
        """Convert a model output action index to something usable by pydraughts"""
        to_pos = action % BOARD_SIZE
        from_pos = int((action - to_pos) / BOARD_SIZE)
        to_pos += 1
        from_pos += 1
        return [from_pos, to_pos]

    def _steps_move_to_action(self, steps_move: list[int]) -> Action:
        """Convert a pydraughts move to something usable by the model"""
        return (steps_move[0] - 1) * BOARD_SIZE + steps_move[1] - 1

    def play(self, action: Action) -> None:
        """
        Play a certain action on the board.
        Assumes that the action is legal, so ensure this first!
        """
        if action not in self.get_legal_actions():
            raise Exception

        self._current_jump_history.append(self.current_jump[:])
        self._move_cache_history.append(self.move_cache[:])

        steps_move = self._action_to_steps_move(action)
        if steps_move not in self.move_cache:
            # Move is part of a jump chain
            new_move_cache: list[list[int]] = []
            for start_pos, end_pos, *further_jumps in self.move_cache:
                if [start_pos, end_pos] == steps_move:
                    new_move_cache.append([end_pos, *further_jumps])

            self.move_cache = new_move_cache
            self.current_jump.append(steps_move[0])
            self.current_jump_end_pos = steps_move[1]
            self._move_was_jump_chain_segment_history.append(True)
            return

        self._move_was_jump_chain_segment_history.append(False)
        steps_move = self.current_jump + steps_move

        move = Move(self.board, steps_move=steps_move)
        self.board.push(move)

        self.current_jump = []
        self.current_jump_end_pos = None
        self.move_cache = []

    def undo(self) -> None:
        """
        Undo the last move taken with game.play().
        """
        if not self._move_was_jump_chain_segment_history.pop():
            self.board.pop()

        self.current_jump = self._current_jump_history.pop()
        self.move_cache = self._move_cache_history.pop()

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
