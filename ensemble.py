import torch
import tqdm

from game import Game, Player
from agent import VModelAgent, VModelEnsembleAgent, PieceCountHeuristicsAgent, play_agent_game
from model import CheckersVModel
from train import VModelTrainRun, load_model_with_hyperparameters

model_names = [f"model_{n}" for n in range(1,4)]
models = []
win_draw_loss_rates: list[tuple[float, float, float]] = []

enemy_agent = PieceCountHeuristicsAgent()

for model_name in model_names:
    model, optimizer, trainrun = load_model_with_hyperparameters(CheckersVModel, torch.optim.SGD, VModelTrainRun, model_name)
    trainrun.load_model(model_name=model_name)
    model = trainrun.model
    models.append(model)
    win_draw_loss_rate = trainrun.evaluate_strength(33, enemy_agent=enemy_agent, enemy_agent_kwargs=dict(depth=2))
    win_draw_loss_rates.append(win_draw_loss_rate)
    print('Win, draw, loss for single model')
    print(win_draw_loss_rate)

ensemble = VModelEnsembleAgent(*models)
player_wins = draws = losses = 0
progress_bar = tqdm.trange(100, position=0, leave=False,
						   desc='Evaluating ensemble playing strength (against depth=3)')
for _ in progress_bar:
	eval_kwargs = dict(epsilon=0.05, depth=3)
	rl_player, winner, _ = play_agent_game(Game(),
										   ensemble,
										   enemy_agent,
										   eval_kwargs,
										   dict(depth=2))
	if winner == rl_player:
		player_wins += 1
	elif winner == Player.NEUTRAL:
		draws += 1
	else:
		losses += 1
	progress_bar.set_postfix_str(f'wins={player_wins}, draws={draws}, losses={losses}')

winrate = player_wins / 100
drawrate = draws / 100
lossrate = losses / 100

print(f'ENSEMBLE WIN/DRAW/LOSS: {winrate}, {drawrate}, {lossrate}')



