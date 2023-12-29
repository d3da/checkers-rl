import numpy as np
from enum import Enum
from draughts import Board, Move
import draughts

GameState = np.ndarray
"""
A GameState should be an 10x10 array filled with Player.WHITE and Player.BLACK for the pieces
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

    NUM_ACTIONS = BOARD_SIZE ** 2
    """
    The first 50 selects the "from" square, the last 50 selects the "to" square
    So the move 11-15 is represented as the number 11*50+15
    """

    GAMESTATE_SIZE = BOARD_SIZE * 2

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
        self.current_jump_end_pos: int | None = None

        self._move_cache_history: list[list[list[int]]] = []
        self._current_jump_history: list[list[int]] = []
        self._move_was_jump_chain_segment_history: list[bool] = []

    def get_state(self) -> GameState:
        """
        Return the game's state in a format that is usable by the NN as input.
        When the player is currently within an incomplete jump chain, we manually update the
        gamestate to reflect the position within the jump chain.

        TODO:
            - Add something related to game history (so draw by repetition can be takes into account)
        """
        def piece_player(piece) -> Player:
            if piece is None:
                return Player.NEUTRAL
            elif piece.player == 1:
                return Player.BLACK
            elif piece.player == 2:
                return Player.WHITE
            raise Exception

        def is_king(piece) -> int:
            """1 for white king, -1 for black king, 0 otherwise"""
            if piece is None or not piece.king:
                return 0
            return piece_player(piece).value

        pieces = [self.board._game.board.searcher.position_pieces.get(i) \
                    for i in range(1, self.BOARD_SIZE + 1)]
        piece_colors = np.array([piece_player(piece).value for piece in pieces])
        piece_kings = np.array([is_king(piece) for piece in pieces])
        return self._add_jump_chain_to_state(np.concatenate([piece_colors, piece_kings]))

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

            captured_pos = self._get_captured_piece_square(from_pos, to_pos, game_state)
            jumping_piece = game_state[from_pos - 1]
            jumping_king = game_state[from_pos - 1 + self.BOARD_SIZE]
            assert game_state[captured_pos - 1] != 0
            assert game_state[captured_pos - 1] == -jumping_piece

            # Set the departure square to empty
            game_state[from_pos - 1] = Player.NEUTRAL.value
            game_state[from_pos - 1 + self.BOARD_SIZE] = Player.NEUTRAL.value
            # Set the arrival square to the piece
            game_state[to_pos - 1] = jumping_piece
            game_state[to_pos - 1 + self.BOARD_SIZE] = jumping_king
            # Capture the piece in the middle
            game_state[captured_pos - 1] = Player.NEUTRAL.value
            game_state[captured_pos - 1 + self.BOARD_SIZE] = Player.NEUTRAL.value

        return game_state

    @staticmethod
    def _get_captured_piece_square(from_pos, to_pos, game_state: GameState) -> int:
        """
        Given the squares that a piece jumps from and to, return the square that is captured.
        This function only works when Game.BOARD_SIZE is 50.
        """
        if Game.BOARD_SIZE != 50:
            print('WARNING: game state logic currently only works for Game.BOARD_SIZE == 50.')
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
            if game_state[convert_back(search_pos) - 1] != Player.NEUTRAL.value:
                return convert_back(search_pos)

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
