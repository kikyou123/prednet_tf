import hickle as hkl
import numpy as np

class DataHandle(object):
    def __init__(self, data_file, source_file, nt, batch_size=8, shuffle=False, seed=None, output_mode='error', sequence_start_mode='all', N_seq=None):
        self.X = hkl.load(data_file)
        self.sources = hkl.load(source_file)
        self.nt = nt
        self.batch_size = batch_size
        self.sequence_start_mode = sequence_start_mode
        self.output_mode = output_mode
        self.im_shape = self.X[0].shape
        self.shuffle = shuffle

        if self.sequence_start_mode == 'all':
            self.possible_starts = np.array([i for i in range(self.X.shape[0] - nt) if self.sources[i] == self.sources[i + self.nt - 1]])
        elif self.sequence_start_mode == 'unique':
            curr_location = 0
            possible_starts = []
            while curr_location < self.X.shape[0] - self.nt + 1:
                if self.sources[curr_location] == self.sources[curr_location + self.nt - 1]:
                    possible_starts.append(curr_location)
                    curr_location += self.nt
                else:
                    curr_location += 1
            self.possible_starts = possible_starts

        if shuffle:
            self.possible_starts = np.random.permutation(self.possible_starts)

        if N_seq is not None and len(self.possible_starts) > N_seq:
            self.possible_starts = self.possible_starts[:N_seq]

        self.N_sequences = len(self.possible_starts)
        self.index_array = np.arange(self.N_sequences)
        self.N_idx = self.N_sequences / self.batch_size

    def next_batch(self, idx):
        if idx == 0 and self.shuffle:
            self.possible_starts = np.random.permutation(self.possible_starts)

        batch_x = np.zeros((self.batch_size, self.nt) + self.im_shape, np.float32)
        index_array = self.index_array[self.batch_size * idx: self.batch_size * (idx + 1)]
        for i, idx1 in enumerate(index_array):
            idx2 = self.possible_starts[idx1]
            batch_x[i] = self.preprocess(self.X[idx2: idx2 + self.nt])
        if self.output_mode == 'error':
            batch_y = np.zeros(self.batch_size, np.float32)
        elif self.output_mode == 'prediction':
            batch_y = batch_x
        return batch_x, batch_y

    def preprocess(self, X):
        return X.astype(np.float32) / 255
    
    def create_all(self):
        X_all = np.zeros((self.N_sequences, self.nt) + self.im_shape, np.float32)
        for i, idx in enumerate(self.possible_starts):
            X_all[i] = self.preprocess(self.X[idx: idx + self.nt])
        return X_all
