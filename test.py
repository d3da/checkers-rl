from game import Game


def test_undo_moves():
    game = Game()
    game.play(33*50+28) # 34-29
    game.play(18*50+22) # 19-23
    game.undo()
    game.play(18*50+22) # 19-23
    game.play(34*50+29) # 35-30
    game.undo()
    game.undo()
    game.play(18*50+22) # 19-23
    game.play(34*50+29) # 35-30
    # Now the only legal move is a jump chain (23-34-25)
    game.play(22*50+33) # 23-34
    game.undo()
    game.play(22*50+33) # 23-34
    game.play(33*50+24) # 34-25
    game.undo()
    game.play(33*50+24) # 34-25
    game.undo()
    game.undo()
    game.play(22*50+33) # 23-34
    game.play(33*50+24) # 34-25
    game.undo()
    game.undo()
    game.undo()
    game.play(34*50+29) # 35-30
    game.play(22*50+33) # 23-34
    game.play(33*50+24) # 34-25
    # jump chain completed, let's try one other move
    game.play(32*50+28) # 33-29
    game.undo()
    game.undo()
    game.play(33*50+24) # 34-25
    game.play(32*50+28) # 33-29
    # No exception == success!
    return

if __name__ == '__main__':
    test_undo_moves()
