import draughts

from game import Game, Player, BOARD_SIZE

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


if __name__ == '__main__':
    from train import play_agent_game
    from agent import PieceCountHeuristicsAgent, UserInputAgent

    game = Game()
    play_agent_game(game,
                    UserInputAgent(),
                    PieceCountHeuristicsAgent(),
                    agent_b_kwargs=dict(depth=3))
    print(game)
    print(f'Winner: {game.get_winner()}')
        
