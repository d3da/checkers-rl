import draughts
import random

from draughts import Board, Move

from game import Game, Player, BOARD_SIZE

"""
This class is used to get heuristics from a Board (instead of Game).
To be deleted
"""

def get_count(board: Board, player: Player) -> int:
    """
    Counts the total number of pieces for WHITE or BLACK
    """
    if player == Player.WHITE :
        player = draughts.WHITE
    else:
        player = draughts.BLACK
    count = 0
    for i in range(1, BOARD_SIZE + 1):
        piece = board._game.board.searcher.position_pieces.get(i)
        if piece is None:
            count = count
        elif piece.player == player:
            count = count + 1
    return count

def kings_count(board: Board, player: Player) -> int:
    """
    Counts the number of kings for WHITE or BLACK
    """
    if player == Player.WHITE :
        player = draughts.WHITE
    else:
        player = draughts.BLACK
    count = 0
    for i in range(1, BOARD_SIZE + 1):
        piece = board._game.board.searcher.position_pieces.get(i)
        if piece is None:
            count = count
        elif piece.player == player:
            if piece.king:
                count = count + 1
    return count

def piece_count(board: Board, king_multiplier: int = 3) -> int:
    """
    Computes the piece count function:
    (w_pieces + 3*w_kings) - (b_pieces + 3*b_kings)
    WHITE is the maximizing player
    BLACK is the minimizing player
    """
    white = get_count(board, Player.WHITE)
    black = get_count(board, Player.BLACK)
    white_kings = kings_count(board, Player.WHITE)
    black_kings = kings_count(board, Player.BLACK)
    return (white - white_kings + king_multiplier * white_kings) - \
            (black - black_kings + king_multiplier * black_kings)


if __name__ == '__main__':
    from train import play_agent_game
    from agent import PieceCountHeuristicsAgent, UserInputAgent

    game = Game()
    play_agent_game(game, UserInputAgent(), PieceCountHeuristicsAgent())
    print(game)
    print(f'Winner: {game.get_winner()}')
