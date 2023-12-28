import torch
from game import Game

class CheckersQModel(torch.nn.Module):

    def __init__(self, num_hidden_layers, hidden_size, output_size):
        super().__init__()
        self.layers = torch.nn.Sequential()

        self.layers.append(torch.nn.Linear(in_features=Game.GAMESTATE_SIZE, out_features=hidden_size))
        self.layers.append(torch.nn.ReLU())

        for _ in range(num_hidden_layers):
            self.layers.append(torch.nn.Linear(in_features=hidden_size, out_features=hidden_size))
            self.layers.append(torch.nn.ReLU())

        self.layers.append(torch.nn.Linear(in_features=hidden_size, out_features=output_size))
        self.tanh = torch.nn.Tanh()
    

    def forward(self, inputs):
        X = self.layers(inputs)
        return self.tanh(X)

