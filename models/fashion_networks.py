import torch
import torch.nn as nn
from torchsummary import summary

class Simple_classifier(nn.Module):
    def __init__(self, input_dim, 
                hidden_dim1,
                hidden_dim2, 
                hidden_dim3, 
                hidden_dim4, 
                output_dim, 
                dropout_rate):
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
    

class CNN_classifier(nn.Module):
    def __init__(self, 
                input_dim, 
                hidden_conv_dim1, 
                hidden_conv_dim2, 
                hidden_conv_dim3, 
                hidden_linear_dim1, 
                hidden_linear_dim2, 
                hidden_linear_dim3, 
                output_dim,
                kernel_size,
                pool_size,
                dropout_rate):
        
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=hidden_conv_dim1, kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(in_channels=hidden_conv_dim1, out_channels=hidden_conv_dim2, kernel_size=kernel_size)
        self.conv3 = nn.Conv2d(in_channels=hidden_conv_dim2, out_channels=hidden_conv_dim3, kernel_size=kernel_size)
        


        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=pool_size)
        self.flatten = nn.Flatten()
        self.logsoftmax = nn.LogSoftmax(dim=1)

        flattened_size = self._compute_flattened_size((1, 28, 28))

        self.fc1 = nn.Linear(flattened_size, hidden_linear_dim1)
        self.fc2 = nn.Linear(hidden_linear_dim1, hidden_linear_dim2)
        self.fc3 = nn.Linear(hidden_linear_dim2, hidden_linear_dim3)
        self.fc4 = nn.Linear(hidden_linear_dim3, output_dim)
        self.dropout = nn.Dropout((dropout_rate))

    def _compute_flattened_size(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            out = self.relu(self.conv1(dummy_input))
            out = self.max_pool(out)
            out = self.relu(self.conv2(out))
            out = self.max_pool(out)
            out = self.relu(self.conv3(out))
            out = self.flatten(out)
            return out.shape[1]

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.max_pool(x)
        x = self.dropout(x)

        
        x = self.relu(self.conv2(x))
        x = self.max_pool(x)
        x = self.dropout(x)

        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.dropout(x)


        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.logsoftmax(self.fc4(x))
        
        return x

    
def train_simple_classifier(model, loss_fn, optimizer, error, train_loader, val_loader, device):
    train_losses = []
    val_losses = []
    prev_epoch_loss = float("inf")
    epoch = 1

    while True:
        model.train()
        train_epoch_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            images = images.view(images.shape[0], -1)
            train_preds = model(images)
            train_batch_loss = loss_fn(train_preds, labels)
            optimizer.zero_grad()
            train_batch_loss.backward()
            optimizer.step()
            train_epoch_loss += train_batch_loss.item()

        else:
            train_epoch_loss = train_batch_loss / len(train_loader)

            with torch.no_grad():
                val_epoch_acc = 0
                val_epoch_loss = 0
                model.eval()

                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    images = images.view(images.shape[0], -1)
                    val_preds = model(images)
                    val_batch_loss = loss_fn(val_preds, labels)
                    val_epoch_loss += val_batch_loss.item()

                    proba = torch.exp(val_preds)
                    _, pred_labels = proba.topk(1, dim=1)

                    result = pred_labels == labels.view(pred_labels.shape)
                    batch_acc = torch.mean(result.type(torch.FloatTensor))
                    val_epoch_acc += batch_acc.item()

                else:
                    val_epoch_loss = val_epoch_loss / len(val_loader)
                    val_epoch_acc = val_epoch_acc / len(val_loader)
                    train_losses.append(train_epoch_loss.item())
                    val_losses.append(val_epoch_loss)

                    print(f"Epoch: {epoch} -> train_loss: {train_epoch_loss:.6f}, val_loss = {val_epoch_loss:.6f}, val_acc: {val_epoch_acc*100:.4f}%")
        

        if abs(train_epoch_loss - prev_epoch_loss) < error:
            return model, train_losses, val_losses
        
        epoch += 1
        prev_epoch_loss = train_epoch_loss


def evaluate_simple_classifier(model, loss_fn, test_loader, device):
    with torch.no_grad():
        test_loss = 0
        test_acc = 0
        model.eval()

        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            images = images.view(images.shape[0], -1)
            test_preds = model(images)
            test_batch_loss = loss_fn(test_preds, labels)
            test_loss += test_batch_loss.item()

            proba = torch.exp(test_preds)
            _, pred_labels = proba.topk(1, dim=1)

            result = pred_labels == labels.view(pred_labels.shape)
            batch_acc = torch.mean(result.type(torch.FloatTensor))
            test_acc += batch_acc.item()
        
        else:
            test_loss = test_loss / len(test_loader)
            test_acc = test_acc / len(test_loader)

            print(f"test_loss: {test_loss:.6f}, test_acc: {test_acc * 100:.4f}")
    return round(test_loss, 6)


            




if __name__ == "__main__":
    simple_classifier = Simple_classifier(
        input_dim=784,
        hidden_dim1=392,
        hidden_dim2=196,
        hidden_dim3=98,
        hidden_dim4=49,
        output_dim=10,
        dropout_rate=0.25
    )


    simple_classifier_summary = summary(simple_classifier, (64, 784))
    print(simple_classifier_summary)



    cnn_classifier = CNN_classifier(
        input_dim=1,
        hidden_conv_dim1=32,
        hidden_conv_dim2=64,
        hidden_conv_dim3=64,
        hidden_linear_dim1=250,
        hidden_linear_dim2=125,
        hidden_linear_dim3=60,
        output_dim=10,
        kernel_size=3,
        pool_size=2,
        dropout_rate=0.2
    )

    cnn_classifier_summary = summary(cnn_classifier, (1, 28, 28))
    print(cnn_classifier_summary)




