import os
import sys
import json
import torch
import logging
import numpy as np
from datetime import datetime
from data.datasets import Data
from options.trainoptions import TrainOptions

from models.image_classification import *
def main():
    args = TrainOptions().initialize()

    args.pin_memory = torch.cuda.is_available()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device_number = torch.cuda.device_count()
    
    # Setting the folder where checkpoints will be stored
    args.checkpoints_savepath_ = args.experiment_mainpath + args.overall_projectname + args.experiment_name + "/checkpoints"
    if not os.path.exists(args.checkpoints_savepath_):
        os.makedirs(args.checkpoints_savepath_)

    print("Creating the Custom Dataset")
    DATASET = Data(args)
    print("Dataset Created Successfully")
    
    for r in range(args.runs):
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        args.checkpoints_savepath = args.checkpoints_savepath_ + "/model_" + dt_string + "/"
        if not os.path.exists(args.checkpoints_savepath):
            os.makedirs(args.checkpoints_savepath)
        
        with open(args.checkpoints_savepath + 'commandline_args.txt', 'w') as f:
            for i in args.__dict__:
                print(str(i) + ": " ,getattr(args, i))
                f.write(str(i) + ": " + str(getattr(args, i)) + "\n")
        
        model = IMGCLASSIFIER(args, DATASET)
        model.Train()


if __name__ == "__main__":
    main()
    