import matplotlib.pyplot as plt
import numpy as np
import os

def show_clothing(dataloader, ind=0):
    for _ in range(ind+1):
        image, label = next(iter(dataloader))
    
    classes = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']

    print(classes[label[0].item()])
    plt.title(classes[label[0].item()])
    plt.imshow(image[0].numpy().squeeze(), cmap='gray')
    plt.show()
    plt.close()


def plot_training_and_validation_loss(train_losses, val_losses, y_label, show, save, relative_save_path):
    if save and not relative_save_path:
        assert ValueError, "must have value for argument save_path if save=True"

    epochs = np.arange(1, len(train_losses)+1)
    plt.plot(epochs, train_losses)
    plt.scatter(epochs, val_losses, color="red")
    plt.legend(["Train Loss", "Validation Loss"])
    plt.xlabel("Epochs")
    plt.ylabel(y_label)
    if save:
        plt.savefig("".join([os.getcwd(), relative_save_path]))

    if show:
        plt.show()

    plt.close()