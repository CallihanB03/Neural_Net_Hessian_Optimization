import torch
from torch.utils.data import DataLoader, sampler
import numpy as np


class HousingDataLoader():
    def __init__(self, data, target, batch_size, shuffle):
        self.ind = 0
        self.target = [target]

        if shuffle:
            self.data = data.sample(frac=1)
        else:
            self.data = data

        self.batch_size = batch_size

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.ind + self.batch_size >= self.data.shape[0]:
            raise StopIteration
        
        df = self.data[self.ind: self.ind+self.batch_size]
        self.ind += self.batch_size

        train_cols = df.columns.difference(self.target)
        X, y = torch.from_numpy(df[train_cols].values).to(torch.float32), torch.from_numpy(df[self.target].values).to(torch.float32)

        return X, y
    


def create_housing_data_loader(data, target, batch_size=64, shuffle=True):
    data_loader = HousingDataLoader(data, target=target, batch_size=batch_size, shuffle=shuffle)
    return data_loader


def train_validation_split(train_data_size, val_size):
    assert  0 < val_size < 1, "val_size must be between 0 and 1"
    train_size = 1 - val_size
    
    indices = list(range(train_data_size))
    np.random.shuffle(indices)
    split = int(np.floor(train_size * train_data_size))
    train_indices, val_indices = indices[:split], indices[split:]
    return train_indices, val_indices

def create_fashion_dataloaders(train_data, test_data, val_size=0.2, batch_size=64):
    if val_size:
        train_size = len(train_data)
        train_indices, val_indices = train_validation_split(train_size, val_size=val_size)


        train_sampler = sampler.SubsetRandomSampler(train_indices)
        val_sampler = sampler.SubsetRandomSampler(val_indices)

        train_data_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)

        val_data_loader = DataLoader(train_data, batch_size=batch_size, sampler=val_sampler)

        test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

        return train_data_loader, val_data_loader, test_data_loader


    train_data_loader = DataLoader(train_data, batch_size)

    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_data_loader, None, test_data_loader



if __name__ == "__main__":
    from load_data import *

    # Housing Dataset
    train_data, test_data = load_housing_data()
    train_loader = create_housing_data_loader(train_data, "median_house_value")

    show = True
    while True:
        try: 
            X, y = next(train_loader)
            if show:
                print(X, y)
                show = False
        
        except StopIteration:
            break

    print("Iterated through all batches")

    # Fashion MNIST Dataset
    train_data, test_data = load_fashion()

    train_loader, val_loader, test_loader = create_fashion_dataloaders(
      train_data=train_data,
      test_data=test_data,
      val_size=0.2,
      batch_size=64  
    )

    print(type(train_loader))
    print(type(val_loader))
    print(type(test_loader))
