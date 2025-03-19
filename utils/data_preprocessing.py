from torch.utils.data import DataLoader

def create_digits_loader(data, batch_size=64, shuffle=True):
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)


class HousingDataLoader():
    def __init__(self, data, batch_size, shuffle):
        self.ind = 0

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
        return df
    


def create_housing_data_loader(data, batch_size=64, shuffle=True):
    data_loader = HousingDataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return data_loader


if __name__ == "__main__":
    from load_data import *

    housing_train_data, housing_test_data = load_housing_data()

    housing_traind_data_loader = create_housing_data_loader(housing_train_data)

    while True:
        try: 
            housing_training_batch = next(housing_traind_data_loader)
        
        except StopIteration:
            break

    print("Iterated through all batches")