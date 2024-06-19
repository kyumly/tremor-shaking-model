import time
import torch
from tqdm import tqdm
from torch.nn import functional as F

from ulits import epoch_time
from torch import nn as nn

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=6, out_features=32, dtype=torch.float64)
        self.fc2 = nn.Linear(in_features=32, out_features=32, dtype=torch.float64)
        self.fc3 = nn.Linear(in_features=32, out_features=32, dtype=torch.float64)
        self.fc5 = nn.Linear(in_features=32, out_features=1, dtype=torch.float64)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc5(x)
        return x

class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers, device):
        super(LSTM, self).__init__()
        seq_length = 5
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.device = device

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size).to(self.device)

        c_0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size).to(self.device)

        # Propagate input through LSTM
        output, (h_n, c_n) = self.lstm(x, (h_0, c_0))

        h_n = h_n.view(-1, self.hidden_size).to(self.device)

        result = self.fc(h_n)

        return result


def train(model, data_loader, optim, criterion, device):
    model.train()
    losses = 0
    loss_list = []
    for d, v in tqdm(data_loader):
        data = d.to(device)
        v = v.to(device)
        optim.zero_grad()
        output_data = model(data)

        result = F.sigmoid(output_data)
        loss = criterion(result.reshape(-1, 1).to(torch.float64), v.reshape(-1, 1).to(torch.float64))
        loss_list.append(loss.item())

        loss.backward()
        optim.step()
        losses += loss.item()

    return losses / len(data_loader)


def evaluate(model, dataloader, criterion, device):
    epoch_loss = 0

    model.eval()
    with torch.no_grad():
        for d, v in tqdm(dataloader):

            data = d.to(device)
            v = v.to(device)

            output_data = model(data)
            result = F.sigmoid(output_data)
            loss = criterion(result.reshape(-1, 1).to(torch.float64), v.reshape(-1, 1).to(torch.float64))
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def trainer(model, num_epoch, dataloader_dict, optim, criterion, early_stop, device):
    EPOCHS = num_epoch
    lowest_epoch = 0
    best_valid_loss = float('inf')
    train_history, valid_history = [], []
    for epoch in range(EPOCHS):
        start_time = time.monotonic()
        train_loss = train(model, dataloader_dict['train'], optim, criterion, device)
        valid_loss = evaluate(model, dataloader_dict['valid'], criterion, device)
        print(valid_loss, train_loss)
        if valid_loss < best_valid_loss:
            lowest_epoch = epoch
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'save.pt')
        if early_stop > 0 and lowest_epoch + early_stop < epoch + 1:
            print("There is no improvement during last %d epochs." % early_stop)
            break
        train_history.append(train_loss)
        valid_history.append(valid_loss)

        end_time = time.monotonic()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Valid. Loss: {valid_loss:.3f}')

    return train_history, valid_history


def get_hyperparameters():
    weight_decay = [1e-4, 1e-3, 1e-2, 1e-1]
    lr = [1e-4, 1e-3, 1e-2, 1e-1]

    return weight_decay, lr
