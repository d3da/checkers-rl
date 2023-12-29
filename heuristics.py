import numpy as np
from enum import Enum
from draughts import Board, Move
import draughts
import random

from game import Game, GameState, Player, Action, BOARD_SIZE

def get_count(game: Game, player: Player) -> int:
    """
    Counts the total number of pieces for WHITE or BLACK
    """
    if player == Player.WHITE :
        player = draughts.WHITE
    else:
        player = draughts.BLACK
    count = 0
    for i in range(1, BOARD_SIZE + 1):
        piece = game.board._game.board.searcher.position_pieces.get(i)
        if piece is None:
            count = count
        elif piece.player == player:
            count = count + 1
    return count

def kings_count(game: Game, player: Player) -> int:
    """
    Counts the number of kings for WHITE or BLACK
    """
    if player == Player.WHITE :
        player = draughts.WHITE
    else:
        player = draughts.BLACK
    count = 0
    for i in range(1, BOARD_SIZE + 1):
        piece = game.board._game.board.searcher.position_pieces.get(i)
        if piece is None:
            count = count
        elif piece.player == player:
            if piece.king :
                count = count + 1
    return count

def piece_count(game: Game, king_multiplier: int = 3) -> int:
    """
    Computes the piece count function:
    (w_pieces + 3*w_kings) - (b_pieces + 3*b_kings)
    WHITE is the maximizing player
    BLACK is the minimizing player
    """
    white = get_count(game, Player.WHITE)
    black = get_count(game, Player.BLACK)
    white_kings = kings_count(game, Player.WHITE)
    black_kings = kings_count(game, Player.BLACK)
    return (white - white_kings + king_multiplier * white_kings) - \
            (black - black_kings + king_multiplier * black_kings)

def select_piece_count_action(game: Game):
    """
    Plays the legal actions, evaluates them according to the piece count function and returns the best action for the current player
    """     

    scores = []
    actions = game.get_legal_actions()
    player = game.get_current_player()

    for action in actions :

        game.play(action)
        i = piece_count(game)
        scores.append(i)

        game.undo()

    if player == Player.WHITE:
        best_score = max(scores)
    else:
        best_score = min(scores)

    indices = [index for index, value in enumerate(scores) if value == best_score]
    pc = random.choice(indices)

    pc_action = actions[pc]

    return pc_action

if __name__ == '__main__':

    game = Game()
    action = select_piece_count_action(game)
    game.play(action)
    print(game)
        
