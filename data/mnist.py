import os
import gzip
import numpy as np

class MNIST():
    def __init__(self, args):
        self.datapath = args.datapath + 'MNIST/'
        self.x_trainpath = self.datapath + 'x_train.npy'
        self.y_trainpath = self.datapath + 'y_train.npy'
        self.x_testpath = self.datapath + 'x_test.npy'
        self.y_testpath = self.datapath + 'y_test.npy'
    def get_train(self):
        self.x_train = np.load(self.x_trainpath)
        self.y_train = np.load(self.y_trainpath)
        return self.x_train, self.y_train
    def get_test(self):
        self.x_test = np.load(self.x_testpath)
        self.y_test = np.load(self.y_testpath)
        return self.x_test, self.y_test