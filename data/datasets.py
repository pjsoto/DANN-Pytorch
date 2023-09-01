import os
import sys
import math
import numpy as np
import pandas as pd
from data.mnist import *
from data.mmnist import *
from data.custom_dataset import *


class Data():
    def __init__(self, args):
        self.args = args
        self.mnist = MNIST(args)
        self.mmnist = MMNIST(args)
        
        x_train_mnist, y_train_mnist = self.mnist.get_train()
        x_test_mnist, y_test_mnist = self.mnist.get_test()
        
        x_train_mnist = x_train_mnist.reshape((60000, 28, 28, 1))
        x_train_mnist = x_train_mnist.repeat(3, -1)

        self.mmnist.set_labels(y_train_mnist, y_test_mnist)
        
        x_train_mmnist, y_train_mmnist = self.mmnist.get_train()
        x_test_mmnist, y_test_mmnist = self.mmnist.get_test()
        
        if self.args.phase == "train":
            # Shuffling the data
            num_samples = x_train_mnist.shape[0]
            index = np.arange(num_samples)
            np.random.shuffle(index)
            x_train_mnist = x_train_mnist[index, :, :, :]
            y_train_mnist = y_train_mnist[index]

            num_samples = x_train_mmnist.shape[0]
            index = np.arange(num_samples)
            np.random.shuffle(index)
            x_train_mmnist = x_train_mmnist[index, :, :, :]
            y_train_mmnist = y_train_mmnist[index]

            
            num_samples = x_train_mnist.shape[0]
            num_valsamples = int((self.args.validation_portion * num_samples)/100)
            x_valid_mnist = x_train_mnist[:num_valsamples,:,:,:]
            y_valid_mnist = y_train_mnist[:num_valsamples]
            x_train_mnist = x_train_mnist[num_valsamples:,:,:,:]
            y_train_mnist = y_train_mnist[num_valsamples:]

            
            num_samples = x_train_mmnist.shape[0]
            num_valsamples = int((self.args.validation_portion * num_samples)/100)
            x_valid_mmnist = x_train_mmnist[:num_valsamples,:,:,:]
            y_valid_mmnist = y_train_mmnist[:num_valsamples]
            x_train_mmnist = x_train_mmnist[num_valsamples:,:,:,:]
            y_train_mmnist = y_train_mmnist[num_valsamples:]
            if self.args.task == "classification":
               if self.args.source == "mnist":
                    self.x_train = x_train_mnist
                    self.y_train = y_train_mnist
                    self.x_valid = x_valid_mnist
                    self.y_valid = y_valid_mnist
               if self.args.source == "mmnist":
                    self.x_train = x_train_mmnist
                    self.y_train = y_train_mmnist
                    self.x_valid = x_valid_mmnist
                    self.y_valid = y_valid_mmnist
            elif self.args.task == "domain_adaptation":
                if self.args.target == "mmnist" and self.args.source == "mnist":
                    self.x_train = np.concatenate((x_train_mnist, x_train_mmnist), axis = 0)
                    self.y_train = np.concatenate((y_train_mnist, np.zeros((y_train_mmnist.shape[0],)) - 1), axis = 0)
                    self.x_valid = np.concatenate((x_valid_mnist, x_valid_mmnist), axis = 0)
                    self.y_valid = np.concatenate((y_valid_mnist, np.zeros((y_valid_mmnist.shape[0],)) - 1), axis = 0)
                elif self.args.target == "mnist" and self.args.source == "mmnist":
                    self.x_train = np.concatenate((x_train_mmnist, x_train_mnist), axis = 0)
                    self.y_train = np.concatenate((y_train_mmnist, np.zeros((y_train_mnist.shape[0],)) - 1), axis = 0)
                    self.x_valid = np.concatenate((x_valid_mmnist, x_valid_mnist), axis = 0)
                    self.y_valid = np.concatenate((y_valid_mmnist, np.zeros((y_valid_mnist.shape[0],)) - 1), axis = 0)
                else:
                    print("Please select a valid configuration among domains")
                    sys.exit()

        elif self.args.phase == "test":
            if self.args.target == "mnist":
                self.x_test = (x_test_mnist.reshape((x_test_mnist.shape[0], 28, 28, 1))).repeat(3, -1)
                self.y_test = y_test_mnist
            if self.args.target == "mmnist":
                self.x_test = x_test_mmnist
                self.y_test = y_test_mmnist

        self.Create()

    def Create(self):
        if self.args.phase == "train":
            self.TRAIN_DATASET = CustomDataset(self.x_train, self.y_train, self.args)
            self.VALID_DATASET = CustomDataset(self.x_valid, self.y_valid, self.args)

            self.train_loader = torch.utils.data.DataLoader(self.TRAIN_DATASET,
                                                            shuffle = True,
                                                            batch_size=self.args.batch_size,
                                                            num_workers=self.args.num_workers,
                                                            )
            self.valid_loader = torch.utils.data.DataLoader(self.VALID_DATASET,
                                                            shuffle = True,
                                                            batch_size=self.args.batch_size,
                                                            num_workers=self.args.num_workers,
                                                            )
        if self.args.phase == "test":
            self.TEST_DATASET = CustomDataset(self.x_test, self.y_test, self.args)

            self.test_loader = torch.utils.data.DataLoader(self.TEST_DATASET,
                                                            shuffle = False,
                                                            batch_size=self.args.batch_size,
                                                            num_workers=self.args.num_workers,
                                                            )