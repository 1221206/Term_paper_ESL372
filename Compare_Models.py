import torch
import torch.nn as nn
from Model.Model_MLPtoKANs import KAN as Encoder
from Model.Model_MLPtoKANs import Predictor

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ResBlock(nn.Module):
    def __init__(self, input_channel, output_channel, stride):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channel, output_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(output_channel),
            nn.ReLU(),

            nn.Conv1d(output_channel, output_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(output_channel)
        )

        self.skip_connection = nn.Sequential()
        if output_channel != input_channel:
            self.skip_connection = nn.Sequential(
                nn.Conv1d(input_channel, output_channel, kernel_size=1, stride=stride),
                nn.BatchNorm1d(output_channel)
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.skip_connection(x) + out
        out = self.relu(out)
        return out


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.encoder = Encoder(input_dim=17, output_dim=32, num_hidden_layers=3, hidden_dim=60, dropout=0.2)
        self.predictor = Predictor(input_dim=32)

    def forward(self, x):
        x = self.encoder(x)
        x = self.predictor(x)
        return x


class LSTM(nn.Module):
    def __init__(self, input_dim=17, hidden_dim=60, output_dim=1, num_layers=2, dropout=0.2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.device = device

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)

        self.fc = nn.Linear(hidden_dim, output_dim)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        x = x.to(self.device)

        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        if x.dim() == 2:
            x = x.unsqueeze(1)

        out, _ = self.lstm(x, (h0, c0))

        out = self.fc(out[:, -1, :])
        return out


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = ResBlock(input_channel=1, output_channel=8, stride=1)
        self.layer2 = ResBlock(input_channel=8, output_channel=16, stride=2)
        self.layer3 = ResBlock(input_channel=16, output_channel=24, stride=2)
        self.layer4 = ResBlock(input_channel=24, output_channel=16, stride=1)
        self.layer5 = ResBlock(input_channel=16, output_channel=8, stride=1)
        self.layer6 = nn.Linear(8 * 5, 1)

    def forward(self, x):
        N, L = x.shape[0], x.shape[1]
        x = x.view(N, 1, L)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out.view(N, -1))
        return out.view(N, 1)


def count_parameters(model, model_name):
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('The {} model has {} trainable parameters'.format(model_name, count))

if __name__ == '__main__':
    print("The MLP model is part of a Physics-Informed Neural Network (PINN) framework.")
    print("The CNN and LSTM models are standard implementations and not part of the physics network.")
    print("\nNetwork Architectures:")
    print("MLP: Encoder (3 hidden layers, 60 hidden dimension), Predictor (1 hidden layer, 32 hidden dimension)")
    print("CNN: 5 ResBlock layers with output channels (8, 16, 24, 16, 8)")
    print("LSTM: 2 layers, 60 hidden dimension")
    print("\n")
    x = torch.randn(10, 17)

    mlp_model = MLP().to(device)
    cnn_model = CNN().to(device)
    lstm_model = LSTM().to(device)

    x = x.to(device)

    y1 = mlp_model(x)
    y2 = cnn_model(x)
    y3 = lstm_model(x)

    count_parameters(mlp_model, 'MLP')
    count_parameters(cnn_model, 'CNN')
    count_parameters(lstm_model, 'LSTM')