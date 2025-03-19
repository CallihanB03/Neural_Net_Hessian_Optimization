from torch.utils.data import DataLoader
from load_data import load_digits

def create_digits_loader(training=True, batch_size=64, shuffle=True):
    if training:
        data, _ = load_digits()
    else:
        _, data = load_digits()

    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)


class HousingDataLoader():
    def __init__(self, data, batch_size, shuffle):
        self.ind = 0

        if shuffle:
            self.data = data.shuffle(frac=1)
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
    


def create_housing_data_loader(training=True, batch_size=64, shuffle=True):
    pass


if __name__ == "__main__":
    pass