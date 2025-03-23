from torchsummary import summary
import torch.nn as nn

class simple_classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, output_dim, dropout_rate):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, hidden_dim4)
        self.output = nn.Linear(hidden_dim4, output_dim)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout((dropout_rate))


    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)

        x = self.relu(self.fc2(x))
        x = self.dropout(x)

        x = self.relu(self.fc3(x))
        x = self.dropout(x)

        x = self.relu(self.fc4(x))
        x = self.dropout(x)

        x = self.logsoftmax(self.output(x))
        return x
    

if __name__ == "__main__":

    model = simple_classifier(
        input_dim=784,
        hidden_dim1=392,
        hidden_dim2=196,
        hidden_dim3=98,
        hidden_dim4=49,
        output_dim=10,
        dropout_rate=0.25
    )

    model_summary = summary(model, (64, 784))

