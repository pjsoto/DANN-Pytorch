import os
import sys
import json
import glob
import torch
import logging
import numpy as np
from datetime import datetime

from data.datasets import Data
from models.image_classification import *
from options.testoptions import TestOptions
def main():
    args = TestOptions().initialize()
    
    args.pin_memory = torch.cuda.is_available()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device_number = torch.cuda.device_count()

    args.checkpoints_savepath = args.experiment_mainpath + args.overall_projectname + args.experiment_name + "/checkpoints"
    if not os.path.exists(args.checkpoints_savepath):
        print("The current folder: " + args.checkpoints_savepath + "doesn't exists")
        print("Please, make sure you are addressing the right checkpoint folders")
        sys.exit()
    
    training_folders = os.listdir(args.checkpoints_savepath)
    if len(training_folders) == 0:
        print("The current folder: " + args.checkpoints_savepath + "doesn't contains trained models")
        print("Please, make sure you are addressing the right checkpoint folders")
        sys.exit()
    else:
        trainedweights_files = glob.glob(args.checkpoints_savepath + '/**/*.pth', recursive = True)
        print(f"{len(trainedweights_files)} .pth files found in " + args.checkpoints_savepath + " directory.")
        if len(trainedweights_files) == 0:
            print("No trained weight were stored in this address")
            sys.exit()
        else:
            args.results_savepath = args.experiment_mainpath + args.overall_projectname + args.experiment_name + "/results/"
            if not os.path.exists(args.results_savepath):
                os.makedirs(args.results_savepath)

    print("Creating the Custom Dataset")
    DATASET = Data(args)
    print("Dataset Created Successfully")
    for trainedweight_file in trainedweights_files:
        args.results_savepath = args.experiment_mainpath + args.overall_projectname + args.experiment_name + "/results/"
        training_folder = trainedweight_file.split("/")[-2]
        args.trainedweight_filepath = trainedweight_file
        args.results_savepath = args.results_savepath + training_folder + "/"
        if not os.path.exists(args.results_savepath):
            os.makedirs(args.results_savepath)
        
        model = IMGCLASSIFIER(args, DATASET)
        model.Evaluates()
        
if __name__ == "__main__":
    main()