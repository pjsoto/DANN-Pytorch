import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
class CustomDataset(Dataset):
    def __init__(self, x, y, args):
        self.args = args
        self.x = x
        self.y = y

        if self.args.phase == "train":
            if self.args.task == "classification":
                self.mask = np.ones((self.y.shape[0],1))
            elif self.args.task == "domain_adaptation":
                self.mask = np.ones((self.y.shape[0],1))
                indexs = np.transpose(np.array(np.where(self.y == -1)))
                self.mask[indexs,0] = 0
                self.y[indexs] = 0
                self.d = self.mask.copy()
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        x_sample = self.x.transpose(0,3,1,2)[idx, :, :, :].astype('float32')/255
        y_sample = self.y[idx]
        x_sample_t = torch.from_numpy(np.asarray(x_sample)) 
        y_sample_t = torch.nn.functional.one_hot(torch.as_tensor(y_sample).long(), num_classes = 10).float()
        if self.args.phase == "train":
            m_sample = self.mask[idx, :]
            m_sample_t = torch.from_numpy(np.asarray(m_sample))
            if self.args.task == "classification":
                return {'x': x_sample_t, 'y': y_sample_t, 'm': m_sample_t}
            elif self.args.task == "domain_adaptation":
                d_sample = self.d[idx,:]
                #d_sample_t = torch.nn.functional.one_hot(torch.as_tensor(d_sample).long(), num_classes = 2).float().squeeze()
                d_sample_t = torch.from_numpy(np.asarray(d_sample))
                return {'x': x_sample_t, 'y': y_sample_t, 'm': m_sample_t, 'd': d_sample_t}
        elif self.args.phase == "test":
            return {'x': x_sample_t, 'y': y_sample_t}

        
