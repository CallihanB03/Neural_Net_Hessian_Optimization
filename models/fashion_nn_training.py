import torch
import torch.nn as nn
import torch.optim as optim
from utils.load_data import load_fashion
from utils.data_preprocessing import create_fashion_dataloaders
from utils.create_plots import plot_training_and_validation_loss
from models.fashion_networks import Simple_classifier

    
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
    # Fashion MNIST Dataset
    train_data, test_data = load_fashion()

    train_loader, val_loader, test_loader = create_fashion_dataloaders(
      train_data=train_data,
      test_data=test_data,
      val_size=0.2,
      batch_size=64  
    )


    # Input Shape 64 x 784
    simple_classifier = Simple_classifier(
        input_dim=784,
        hidden_dim1=392,
        hidden_dim2=196,
        hidden_dim3=98,
        hidden_dim4=49,
        output_dim=10,
        dropout_rate=0.25
    )


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss_fn = nn.NLLLoss()
    optimizer = optim.Adam(simple_classifier.parameters(), lr=0.001, weight_decay=0.05)

    simple_classifier, train_losses, val_losses = train_simple_classifier(
        model=simple_classifier,
        loss_fn=loss_fn,
        optimizer=optimizer,
        error=1e-5,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )

    test_loss = evaluate_simple_classifier(
        model=simple_classifier,
        loss_fn=loss_fn,
        test_loader=test_loader,
        device=device
    )

    plot_training_and_validation_loss(
        train_losses=train_losses,
        val_losses=val_losses,
        y_label="Negative Log Likelihood",
        show=False,
        save=True,
        relative_save_path=f"/figures/simple_nn/simple_classifier_test_loss_{test_loss}.png"
    )
