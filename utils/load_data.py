from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from sklearn.datasets import fetch_california_housing
import pandas as pd



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

def load_housing_data():
    data_bunch = fetch_california_housing()
    col_names = data_bunch["feature_names"]
    data_arr = data_bunch["data"]
    df = pd.DataFrame(data=data_arr, columns=col_names)
    return df



if __name__ == "__main__":
    pass
