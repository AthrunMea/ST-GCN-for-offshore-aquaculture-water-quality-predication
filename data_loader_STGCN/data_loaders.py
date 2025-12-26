import torch
from torch.utils.data import Dataset
import os
import numpy as np


class DatasetProcessing(Dataset):
    def __init__(self, data_path, task, phase, transform=None, device='cuda'):
        self.transform = transform
        self.phase = phase
        self.device = device
        self.npz_data = np.load(os.path.join(data_path, task, phase+'.npz'))
        self.data_in = self.npz_data['x']
        self.data_out = self.npz_data['y']
        if self.transform is not None:
            self.data_in = self.simple_normalization(self.data_in, 1)
            self.data_out = self.simple_normalization(self.data_out, 1)

    def __getitem__(self, index):
        x = torch.from_numpy(np.float32(self.data_in[index])).to(self.device)
        y = torch.from_numpy(np.float32(self.data_out[index])).to(self.device)
        return x, y

    def __len__(self):
        return self.data_in.shape[0]

    def simple_normalization(self, x, dim):
        eps = 1e-5
        x_mean = torch.mean(x, dim=dim, keepdim=True)
        x_var = torch.mean((x - x_mean) ** 2, dim=dim, keepdim=True)
        x_hat = (x - x_mean) / torch.sqrt(x_var + eps)
        return x_hat

    def random_shuffle(self):
        shuffle_ix = np.random.permutation(np.arange(self.data_in.shape[0]))
        self.data_in = self.data_in[shuffle_ix]
        self.data_out = self.data_out[shuffle_ix]


def data_generator_np(data_path, task, batch_size, device='cuda'):
    train_dataset = DatasetProcessing(data_path, task, 'train', device=device)
    test_dataset = DatasetProcessing(data_path, task, 'test', device=device)

    #train_loader = 0 #pred用
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=False,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0)

    return train_loader, test_loader
