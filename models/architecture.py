import os
import torch
import numpy as np

from models.ganinarch import *
class Architectures():
    def __init__(self, args) -> None:
        super(Architectures, self).__init__()
        self.args = args
        if self.args.architecture == "ganin":
            self.model = GANINARCH(args)

            if self.args.phase == 'test':
                state_dict = torch.load(self.args.trainedweight_filepath)
                msg = self.model.load_state_dict(state_dict, strict = True)
                print('Classification model weights found at {} and loaded with msg: {}'.format(args.trainedweight_filepath, msg))
                print("[!] Model loaded sucessfully")