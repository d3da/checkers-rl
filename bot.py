import heuristics
import board_heu
from game import Game, Player, BOARD_SIZE
from draughts import Board, Move
import time

"""
TO DO:
- Clean up code
- For now, everything minimax-related is played on the Board (not Game) 
  --> change that
- Problem with moves that are evaluated the same: the model will have a tendency to play a little too much on the left (or right) 
  --> implement some randomness
- Minimax takes too long to compute 
  --> alpha-beta pruning

Minimax handles jump chains as one action.
"""

def minimax(board: Board, depth: int, max_player: bool):
    """
    Minimax algorithm
    Returns the score, the move and the resulting board of the best move to play according to heuristics
    """
    if depth == 0 or board.is_over():
        # return board_heu.piece_count(board), board.move_stack[-1].steps_move, board
        return board_heu.piece_count(board), board.move_stack[-1]
    if max_player:
        max_eval = float('-inf')
        best_board = None
        for child in get_children(board):
            evaluation = minimax(child, depth-1, False)[0]
            max_eval = max(max_eval, evaluation)
            if max_eval == evaluation:
                best_board = child
        # return max_eval, best_board.move_stack[-1].steps_move, best_board
        return max_eval, best_board.move_stack[-1]
    else:
        min_eval = float('inf')
        best_board = None
        for child in get_children(board):
            evaluation = minimax(child, depth-1, True)[0]
            min_eval = min(min_eval, evaluation)
            if min_eval == evaluation:
                best_board = child
        # return min_eval, best_board.move_stack[-1].steps_move, best_board
        return min_eval, best_board.move_stack[-1]
        

def get_children(parent_board: Board):
    """
    Returns a list of resulting (children) boards from a parent board by playing all legal moves
    """
    children = []
    moves = parent_board.legal_moves()
    for move in moves:
        child = parent_board.push(move)
        children.append(child.copy())
        child.pop() # goes back to parent board
    return children

if __name__ == '__main__':
    start_time = time.time()
    game = Game()
    print(minimax(game.board, 4, True))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The function took {elapsed_time} seconds to execute.")
