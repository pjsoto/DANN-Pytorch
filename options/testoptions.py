import argparse
from options.baseoptions import BaseOptions


class TestOptions():
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """
    def __init__(self):
        super(TestOptions, self).__init__()

        self.parser = argparse.ArgumentParser(description='')

    def initialize(self):
        self.parser = BaseOptions(self.parser).initialize()
        self.parser.add_argument("--phase", type = str, default = "test", help = "Decide the model purpose between train|test")
        self.parser.add_argument("--batch_size", type = int, default = 1, help="Set the number of samples in each batch")
        return self.parser.parse_args()