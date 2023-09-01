import argparse
from options.baseoptions import BaseOptions


class MetricsOptions():
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """
    def __init__(self):
        super(MetricsOptions, self).__init__()

        self.parser = argparse.ArgumentParser(description='')

    def initialize(self):
        self.parser = BaseOptions(self.parser).initialize()
        self.parser.add_argument("--plots", type = eval, choices=[True, False], default = True, help="Decide if plots of metrics will be created")
        return self.parser.parse_args()