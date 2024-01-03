import tqdm
import random
from game import Game, GameState, Player, Action
import heuristics
import time
from draughts import Board, Move
import board_heu
import bot

"""
Compares minimax player with random player
"""

"""
TO DO:
- Clean up code
- Implement random choice of WHITE and BLACK for model_play in playbot
"""

"""
Observations:
- Game between heuristics bot (depth=3) and random bot lasts between 40 and 75 seconds approx (without pruning).
- 100 games between heuristics bot and random bot
  > win_rate = 1.0
  > lose_rate = 0.0
  > score_bot = 12.75
  > score_random = 0.19
  > Took 6332.291348695755 seconds to compute

- Game between heuristics bot (depth=3) and random bot lasts between 15 and 28 seconds approx (with pruning).
- 100 games between heuristics bot and random bot
  > win_rate = 1.0
  > lose_rate = 0.0
  > score_bot = 12.75
  > score_random = 0.27
  > Took 2263.570200920105 seconds to compute

"""
        
def botplay():
    """
    Play a game using the piece count function against an opponent playing random moves only.
    Returns two (mutually exclusive) booleans indicating whether the model player won, or lost
    Also returns the number of pieces left on the board for each opponent, at the end of the game
    
    For now, model_player only plays WHITE
    
    """
    
    model_player = 2
    random_player = 1
    board = Board()

    while True:
        current_player = board.turn

        if current_player == model_player:
            move = bot.minimax(board, 3, -1000, 1000, True)[1]
        else:
            move = random.choice(board.legal_moves())
        board.push(move)

        if board.is_over():
            print(board)
            return board.winner() == model_player, board.winner() == random_player, board_heu.get_count(board, Player.WHITE), board_heu.get_count(board, Player.BLACK)
        

def evaluate_pc_vs_random(num_games: int):
    
    """
    Plays a number of games using the piece count function against an opponent playing random moves only.
    Returns win and lose rate and the average of pieces left for the model and the random opponent.
    
    For now, model_player only plays WHITE
    
    """
    
    player_wins = 0
    draws = 0
    
    total_count_model = []
    total_count_random = []

    for _ in tqdm.trange(num_games, position=1, leave=False, desc='Playing against pc moves'):
        start_time = time.time()
        player_won, draw, count_model, count_random = botplay()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Game {_} took {elapsed_time} seconds to execute.")
        if player_won:
            player_wins += 1
        if draw:
            draws += 1
        total_count_model.append(count_model)
        total_count_random.append(count_random)

    win_rate = player_wins / num_games
    draw_rate = draws / num_games
    
    # not sure those are the best metrics to evaluate a game
    score_model = sum(total_count_model)/ num_games
    score_random = sum(total_count_random)/ num_games

    return win_rate, draw_rate, score_model, score_random
    
"""
def play(game: Game):

    # Randomly select a side
    model_player = random.choice([Player.WHITE, Player.BLACK])
    print(model_player)
    game.reset()
    
    if model_player == Player.WHITE:
        random_player = Player.BLACK
    else:
        random_player = Player.WHITE

    while True:
        state = game.get_state()
        current_player = game.get_current_player()
        legal_actions = game.get_legal_actions()

        if current_player == model_player:
            action = heuristics.select_piece_count_action(game)
        else:
            action = select_random_action(legal_actions)
        game.play(action)

        if game.has_ended():
            return game.get_winner() == model_player, game.get_winner() == Player.NEUTRAL, heuristics.get_count(game, model_player), heuristics.get_count(game, random_player)
"""

if __name__ == '__main__':
    print("starting...")
    start_time = time.time()
    print(evaluate_pc_vs_random(100))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The function took {elapsed_time} seconds to execute.")
    
    # For 10 games, depth=1
    # win_rate = 0.6, draw_rate = 0.2                                           
    # The function took 101.26401042938232 seconds to execute.
    
    # For 100 games, depth=1
    # win_rate = 0.54, draw_rate = 0.03                                                                                                                         
    # The function took 997.6490168571472 seconds to execute.
    
    # Updated minimax function takes 40 seconds with depth = 3 (without pruning)
    
    # For 10 games (minimax and depth = 3) : 1.0, 0.0                                                                                                               # The function took 545.5544471740723 seconds to execute.
    
