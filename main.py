import numpy as np
import struct
from functools import reduce

class SVM:
    def __init__(self):
        self.X = np.array(0)
        self.Y = np.array(0)
        self.X_train = np.array(0)
        self.X_test = np.array(0)

    def _idx_loader(self, file_path:str):
        with open(file_path, 'rb') as input_file:
            magic = input_file.read(4)
            dims = int(magic[3])
            sizes = [struct.unpack('>L', input_file.read(4))[0] for _ in range(dims)]
            size = reduce(lambda x, y: x * y, sizes)
            data = np.array(list(input_file.read(size)), dtype=float)
            data = data.reshape(sizes)
            return data

    def load_data_idx(self, path_images:str, path_lables:str):
        images = self._idx_loader(path_images)
        features = images.reshape((images.shape[0], images.shape[1] * images.shape[2])) / 128 - 1.0
        labels = self._idx_loader('labels.idx')
        self.X = features
        self.Y = labels

        self.X = self._correct(self.X)
        self.Y = self._correct(self.Y)
        
    def _correct(self, X):
        if X.shape[0]>X.shape[-1]:
            X = X.T
        return X

    def _get_size(self):
        self.X = self._correct(self.X)
        return self.X.shape[1]

    def extract_sample(self, train_size:int, test_size:int):
        assert train_size <= self._get_size() and test_size <= self._get_size(), "No enough data.."
        self.X_train = self.X[:, :train_size]
        self.X_test = self.X[:, -test_size:]

    def print_info(self):
        print(f"X shape: {self.X.shape}")
        print(f"Y shape: {self.Y.shape}")
        print(f"X train shape: {self.X_train.shape}")
        print(f"X test shape: {self.X_test.shape}")
        #add some print options


def main():
    model = SVM()
    model.load_data_idx(path_images='images.idx', path_lables='labels.idx')
    model.extract_sample(50000, 5000)
    model.print_info()

if __name__ == '__main__':
    main()
