import torch
import torch.nn as nn
import torch.optim as optim
from utils.load_data import load_fashion
from utils.data_preprocessing import create_fashion_dataloaders
from utils.create_plots import plot_training_and_validation_loss
from models.fashion_networks import CNN_classifier
from models.lbfgs_optimizer import RegularizedLBFGS
import numpy as np
import pandas as pd

    

def train_cnn_classifier(model, loss_fn, optimizer, error, train_loader, val_loader, device):
    train_losses = []
    val_losses = []
    val_acc = []
    prev_epoch_loss = float("inf")
    epoch = 1

    while True:
        model.train()
        train_epoch_loss = 0

        for images, labels in train_loader:

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            if isinstance(optimizer, torch.optim.LBFGS):
                optimizer_type = "LBFGS"
                def closure():
                    train_preds = model(images)
                    train_batch_loss = loss_fn(train_preds, labels)
                    train_batch_loss.backward()
                    return train_batch_loss
                train_batch_loss = optimizer.step(closure)

            else:
                optimizer_type = "Adam"
                train_preds = model(images)
                train_batch_loss = loss_fn(train_preds, labels)
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
                    val_acc.append(val_epoch_acc*100)

                    print(f"Epoch: {epoch} -> train_loss: {train_epoch_loss:.6f}, val_loss = {val_epoch_loss:.6f}, val_acc: {val_epoch_acc*100:.4f}%")
        

        if abs(train_epoch_loss - prev_epoch_loss) < error:
            return model, train_losses, val_losses, val_acc, optimizer_type
        
        epoch += 1
        prev_epoch_loss = train_epoch_loss


def evaluate_cnn_classifier(model, loss_fn, test_loader, device):
    with torch.no_grad():
        test_loss = 0
        test_acc = 0
        model.eval()

        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
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
    return round(test_loss, 6), round(test_acc, 6)


if __name__ == "__main__":
    # Fashion MNIST Dataset
    train_data, test_data = load_fashion()

    train_loader, val_loader, test_loader = create_fashion_dataloaders(
      train_data=train_data,
      test_data=test_data,
      val_size=0.2,
      batch_size=64  
    )


    # Input Shape 1 x 28 x 28
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


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss_fn = nn.NLLLoss()
    optimizer_adam = optim.Adam(cnn_classifier.parameters(), lr=0.001, weight_decay=0.0025)
    optimizer_lgfbs = torch.optim.LBFGS(cnn_classifier.parameters(), lr=0.01, max_iter=10, history_size=10)



    cnn_classifier, train_losses, val_losses, val_acc, optimizer_type = train_cnn_classifier(
        model=cnn_classifier,
        loss_fn=loss_fn,
        optimizer=optimizer_lgfbs,
        error=1e-5,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )

    training_data_matrix = np.array([train_losses, val_losses, val_acc])
    training_data_rotated = np.zeros((training_data_matrix.shape[1], training_data_matrix.shape[0]))
    for i in range(training_data_matrix.shape[0]):
        training_data_rotated[:, i] = training_data_matrix[i, :]
    training_df = pd.DataFrame(training_data_rotated, columns=["training loss", "validation loss", "testing loss"])


    test_loss, test_acc = evaluate_cnn_classifier(
        model=cnn_classifier,
        loss_fn=loss_fn,
        test_loader=test_loader,
        device=device
    )

    if optimizer_type == "LBFGS":
        training_df.to_csv(f"training_data/LBFGS/test_loss_{test_loss}_test_acc_{test_acc}.csv")
    else:
        training_df.to_csv(f"training_data/Adam/test_loss_{test_loss}_test_acc_{test_acc}.csv")

    plot_training_and_validation_loss(
        train_losses=train_losses,
        val_losses=val_losses,
        y_label="Negative Log Likelihood",
        show=False,
        save=True,
        relative_save_path=f"/figures/cnn/test_loss_{test_loss}.png"
    )


