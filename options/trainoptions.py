import argparse
from options.baseoptions import BaseOptions


class TrainOptions():
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """
    def __init__(self):
        super(TrainOptions, self).__init__()

        self.parser = argparse.ArgumentParser(description='')

    def initialize(self):
        self.parser = BaseOptions(self.parser).initialize()
        self.parser.add_argument("--training_graphs", type = eval, choices = [True, False], default = True, help = "Defines if graphs with the training behaviour will be saved at the end of the training")
        self.parser.add_argument("--continue_training", type = eval, choices= [True, False], default = False, help = "Decides if the training will continue from an specific point")
        self.parser.add_argument("--validation_portion", type = int, default = 20, help = "Defines the percentage of training set used in validation set")
        self.parser.add_argument("--phase", type = str, default = "train", help = "Decide the model purpose between train|test")
        self.parser.add_argument("--runs", type = int, default = 5, help="Set the number of times the algorithm will be executed")
        self.parser.add_argument("--batch_size", type = int, default = 32, help="Set the number of samples in each batch")
        self.parser.add_argument("--learningrate", type = float, default=1e-2, help="Defines the learning rate value during training")
        self.parser.add_argument("--learningrate_decay", type = eval, choices = [True, False], default = True, help = "Defines if learning rate decay during training will be applied")
        self.parser.add_argument("--starting_epoch", type = int, default = 0, help = "Help in the continuation of trainings be defining the starting epoch")
        self.parser.add_argument("--max_epochs", type = int, default = 100, help = "Defines the max number of training epochs")
        self.parser.add_argument("--bestf1", type = float, default = 0, help = "Help in the continuation of trainings be defining the starting value of the monitored metric")
        self.parser.add_argument("--patience", type = int, default = 10, help = "Defines the number of epochs the algorithm has to wait before finishing without any improvement in the watched score")
        self.parser.add_argument("--optimizer", type = str, default ="adam", help = "Defines the optimizer to be used during the training")
        return self.parser.parse_args()