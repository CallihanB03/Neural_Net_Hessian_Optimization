import torch
import torch.nn as nn
from utils.data_preprocessing import create_housing_data_loader



class RegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.l1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.l2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.l3 = nn.Linear(self.hidden_dim, self.output_dim)

        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.relu(self.l3(x))
        return x


def train_model(train_data, model, loss, optimizer, batch_size=64, error=10e-4):
    train_loader = create_housing_data_loader(train_data, batch_size=batch_size, shuffle=True)
    epoch_losses = []
    epoch = 1
    batch_loss = 0
    prev_epoch_loss = float("inf")

    
    while True:
        optimizer.zero_grad()
        try:
            train_batch = next(train_loader)

        except:
            StopIteration
            train_loader = create_housing_data_loader(train_data, batch_size=batch_size, shuffle=True)
            curr_epoch_loss = batch_loss
            epoch_losses.append(curr_epoch_loss)
            
            if abs(prev_epoch_loss - curr_epoch_loss) < error:
                model, epoch_losses
                return 
            
            epoch += 1
            continue

        X, y = train_batch[train_batch.columns[:-1]], train_batch[train_batch.columns[-1]]
        y_pred = model(X)

        batch_loss += loss(y, y_pred)
        batch_loss.backward()
        optimizer.step()
        


if __name__ == "__main__":
    pass