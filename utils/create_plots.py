import matplotlib.pyplot as plt

def show_clothing(dataloader, ind=0):
    for _ in range(ind+1):
        image, label = next(iter(dataloader))
    
    classes = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']

    print(classes[label[0].item()])
    plt.title(classes[label[0].item()])
    plt.imshow(image[0].numpy().squeeze(), cmap='gray')
    plt.show()
    plt.close()


