import os
import argparse



class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    """
        The following implementation has been supported on the cyclegan code available at:

    """

    def __init__(self, parser):
        """Reset the class; indicates the class hasn't been initailized"""
        self.parser = parser

    def initialize(self):
        # Architecture parameters
        self.parser.add_argument("--labelpredictor_classes", type = int, default = 10, help = "Defines the number of classes in the classification task")
        self.parser.add_argument("--architecture", type = str, default = "ganin", help = "Defines the architecture")
        # Dataset hyperparameters
        self.parser.add_argument('--num_workers', type = int, default = 4, help = "Defines the number of cpus will be used in the dataloader")
        self.parser.add_argument("--task", type = str, default = "domain_adaptation", help = "Defines the task among classification|domain_adaptation")
        self.parser.add_argument("--source", type = str, default = "mnist", help = "Defines the source domain")
        self.parser.add_argument("--target", type = str, default = "mmnist", help = "Defines the target domain")
        self.parser.add_argument('--experiment_mainpath', type = str, default = '/home/psoto/Data/WORK/LaTIM/EXPERIMENTS/')
        self.parser.add_argument("--overall_projectname", type = str, default = "/DANN_MNIST", help = "Set the Global project name")
        self.parser.add_argument("--experiment_name", type = str, default = "/S_MNIST_T_MMNIST_DA", help = "Set the name of the expriment")
        self.parser.add_argument("--datapath",type = str, default = "/home/psoto/Data/WORK/LaTIM/DATA/", help = "Defines the root path of the images used to train the model")
        return self.parser