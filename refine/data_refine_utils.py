import hickle as hkl
import numpy as np

class DataHandle(object):
    def __init__(self, data_file1, data_file2, label_file, batch_size, shuffle=False, seed=42, N_seq=None):
        self.X1 = hkl.load(data_file1)
        self.X2 = hkl.load(data_file2)
        self.Y = hkl.load(label_file)
        
        self.im_shape = self.X1[0].shape
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.possible_starts = np.arange(self.X1.shape[0])

        if shuffle:
            self.possible_starts = np.random.permutation(self.possible_starts)

        if N_seq is not None:
            self.possible_starts = self.possible_starts[:N_seq]

        self.N_sequences = len(self.possible_starts)
        self.index_array = np.arange(self.N_sequences)
        self.N_idx = self.N_sequences / self.batch_size

    def next_batch(self, idx):
        if idx == 0 and self.shuffle:
            self.possible_starts = np.random.permutation(self.possible_starts)
        batch_x1 = np.zeros((self.batch_size,) + self.im_shape, np.float32)
        batch_x2 = np.zeros((self.batch_size,) + self.im_shape, np.float32)
        batch_y = np.zeros((self.batch_size,) + self.im_shape, np.float32)
        index_array = self.index_array[self.batch_size * idx: self.batch_size * (idx + 1)]
        for i, idx1 in enumerate(index_array):
            idx2 = self.possible_starts[idx1]
            batch_x1[i] = self.preprocess(self.X1[idx2])
            batch_x2[i] = self.preprocess(self.X2[idx2])
            batch_y[i] = self.preprocess(self.Y[idx2])
        return batch_x1, batch_x2, batch_y

    def preprocess(self, X):
        return X.astype(np.float32) / 127.5 - 1

    def create_all(self):
        X1_all = np.zeros((self.N_sequences,) + self.im_shape, np.float32)
        X2_all = np.zeros((self.N_sequences,) + self.im_shape, np.float32)
        Y_all = np.zeros((self.N_sequences,) + self.im_shape, np.float32)
        for i, idx in enumerate(self.possible_starts):
            X1_all[i] = self.preprocess(self.X1[idx])
            X2_all[i] = self.preprocess(self.X2[idx])
            Y_all[i] = self.preprocess(self.Y[idx])
        return X1_all, X2_all, Y_all
