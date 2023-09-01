import os
import gzip
import pickle
import numpy as np
import urllib.request

class MMNIST():
    def __init__(self, args):
        self.url = "https://github.com/VanushVaswani/keras_mnistm/releases/download/1.0/keras_mnistm.pkl.gz"
        self.datapath = args.datapath + 'MMNIST/'
        self.raw_folder = "raw"
        self.processed_folder = "processed"
        # make data dirs
        if not os.path.exists(os.path.join(self.datapath, self.raw_folder)):
            os.makedirs(os.path.join(self.datapath, self.raw_folder))
        if not os.path.exists(os.path.join(self.datapath, self.processed_folder)):
            os.makedirs(os.path.join(self.datapath, self.processed_folder))
        # download pkl files
        print("Downloading " + self.url)
        filename = self.url.rpartition("/")[2]
        self.file_path = os.path.join(self.datapath, self.raw_folder, filename)
        if not os.path.exists(self.file_path.replace(".gz", "")):
            data = urllib.request.urlopen(self.url)
            with open(self.file_path, "wb") as f:
                f.write(data.read())
            with open(self.file_path.replace(".gz", ""), "wb") as out_f, gzip.GzipFile(self.file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(self.file_path)
            
        print("Processing MNIST Modified...")
        # load MNIST-M images from pkl file
        with open(self.file_path.replace(".gz", ""), "rb") as f:
            mnist_m_data = pickle.load(f, encoding="bytes")
            self.x_train = mnist_m_data[b"train"]
            self.x_test = mnist_m_data[b"test"]
            print(np.shape(self.x_train))     
    def set_labels(self, y_train, y_test):
        self.y_train = y_train
        self.y_test = y_test
    
    def get_train(self):
        return self.x_train, self.y_train

    def get_test(self):
        return self.x_test, self.y_test