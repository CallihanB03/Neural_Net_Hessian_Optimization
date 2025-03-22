import torch.nn as nn
import torch.optim as optim
from utils.data_preprocessing import create_housing_data_loader
import matplotlib.pyplot as plt


class RegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.hidden_dim3 = hidden_dim3
        self.output_dim = output_dim

        self.l1 = nn.Linear(self.input_dim, self.hidden_dim1)
        self.l2 = nn.Linear(self.hidden_dim1, self.hidden_dim2)
        self.l3 = nn.Linear(self.hidden_dim2, self.hidden_dim3)
        self.l4 = nn.Linear(self.hidden_dim3, self.output_dim)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.relu(self.l3(x))
        x = self.relu(self.l4(x))
        return x


def train_model(train_data, target, model, loss, optimizer, batch_size=64, error=10e-4):
    train_loader = create_housing_data_loader(train_data, target=target, batch_size=batch_size, shuffle=False)
    epoch_losses = []
    epoch = 1
    curr_epoch_loss = 0
    prev_epoch_loss = float("inf")


    while True:
        optimizer.zero_grad()
        curr_epoch_loss = 0

        for X, y in train_loader:
            optimizer.zero_grad()
            y_pred = model(X)
            batch_loss = loss(y, y_pred)

            curr_epoch_loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()

        
        epoch_losses.append(curr_epoch_loss)
        print(f"Epoch: {epoch}, loss = {curr_epoch_loss:.4f}")

        if abs(prev_epoch_loss - curr_epoch_loss) < error:
            return model, epoch_losses
        
        prev_epoch_loss = curr_epoch_loss
        epoch += 1
        


if __name__ == "__main__":
    from utils.load_data import load_housing_data
    train_data, testing_data = load_housing_data()

    num_train_cols = train_data.shape[1]-1
    model1 = RegressionModel(input_dim=num_train_cols, 
                             hidden_dim1=64,
                             hidden_dim2=32,
                             hidden_dim3=1,
                             output_dim=1
                             )


    mse_loss = nn.MSELoss()
    adam_optimizer = optim.Adam(model1.parameters())




    model1, losses = train_model(train_data=train_data, target="median_house_value", model=model1, loss=mse_loss, optimizer=adam_optimizer)