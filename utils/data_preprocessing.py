import torch
from torch.utils.data import DataLoader


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


if __name__ == "__main__":
    from load_data import *

    housing_train_data, housing_test_data = load_housing_data()

    housing_traind_data_loader = create_housing_data_loader(housing_train_data, "median_house_value")

    show = True
    while True:
        try: 
            X, y = next(housing_traind_data_loader)
            if show:
                print(X, y)
                show = False
        
        except StopIteration:
            break

    print("Iterated through all batches")