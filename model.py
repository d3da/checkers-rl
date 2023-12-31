import torch
from game import GAMESTATE_SIZE, NUM_ACTIONS


class CheckersQModel(torch.nn.Module):

    def __init__(self, num_hidden_layers, hidden_size):
        super().__init__()
        self.layers = torch.nn.Sequential()

        self.layers.append(torch.nn.Linear(in_features=GAMESTATE_SIZE, out_features=hidden_size))
        self.layers.append(torch.nn.ReLU())

        for _ in range(num_hidden_layers):
            self.layers.append(torch.nn.Linear(in_features=hidden_size, out_features=hidden_size))
            self.layers.append(torch.nn.ReLU())

        self.layers.append(torch.nn.Linear(in_features=hidden_size, out_features=NUM_ACTIONS))
        self.tanh = torch.nn.Tanh()
    
    def forward(self, inputs):
        X = self.layers(inputs)
        return self.tanh(X)


class CheckersVModel(torch.nn.Module):

    def __init__(self, num_hidden_layers, hidden_size):
        super().__init__()
        self.layers = torch.nn.Sequential()
        self.layers.append(torch.nn.Linear(in_features=GAMESTATE_SIZE, out_features=hidden_size))
        self.layers.append(torch.nn.ReLU())

        for _ in range(num_hidden_layers):
            self.layers.append(torch.nn.Linear(in_features=hidden_size, out_features=hidden_size))
            self.layers.append(torch.nn.ReLU())

        self.layers.append(torch.nn.Linear(in_features=hidden_size, out_features=1))
        self.tanh = torch.nn.Tanh()
    

    def forward(self, inputs):
        X = self.layers(inputs)
        return self.tanh(X)

