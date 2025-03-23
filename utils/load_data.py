from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import transforms
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5), (0.5))]
                                )


def load_digits():
    train_data = MNIST(root="~/Academics/Projects/data", 
                       train=True, 
                       download=False, 
                       transform=transforms.ToTensor()
                       )
    
    test_data =  MNIST(root="~/Academics/Projects/data", 
                       train=False, 
                       download=False, 
                       transform=transforms.ToTensor()
                       )
    
    return train_data, test_data


def load_fashion():
    train_data = FashionMNIST(root="~/Academics/Projects/data",
                              download=True,
                              train=True,
                              transform=transform
                              )
    test_data = FashionMNIST(root="~/Academics/Projects/data",
                              download=True,
                              train=False,
                              transform=transform
                              )
    return train_data, test_data


def load_housing_data():
    # Create df
    data_bunch = fetch_california_housing()
    col_names = data_bunch["feature_names"]
    data_arr = data_bunch["data"]
    df = pd.DataFrame(data=data_arr, columns=col_names)
    df["median_house_value"] = data_bunch["target"]


    # train test split
    X, y = df[df.columns[:-1]], df[df.columns[-1]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    return train_data, test_data





if __name__ == "__main__":
    fashion_train, fashion_test = load_fashion()
