import os
import sys
import json
import glob
import torch
import logging
import numpy as np
import pandas as pd
from datetime import datetime

#from data.LCDM_TP import Data
#from models.Models import Models
from options.metricoptions import MetricsOptions
from utils.tools import get_metrics_fth, createbarplot
def main():
    RECALL = []
    F1SCORE = []
    ACCURACY = []
    PRECISION = []
    
    args = MetricsOptions().initialize()
        
    args.results_savepath = args.experiment_mainpath + args.overall_projectname + args.experiment_name + "/results/"

    if not os.path.exists(args.results_savepath):
        print("The current folder: " + args.results_savepath + "doesn't exists")
        print("Please, make sure you are addressing the right results folders")
        sys.exit()

    results_folders = os.listdir(args.results_savepath)
    if len(results_folders) == 0:
        print("The current folder: " + args.results_folders + "doesn't contains evaluated models")
        print("Please, make sure you are addressing the right results folders")
        sys.exit()
    else:
        csvresults_files = glob.glob(args.results_savepath + '/**/*.csv', recursive = True)
        print(f"{len(csvresults_files)} .csv files found in " + args.results_savepath + " directory.")
        if len(csvresults_files) == 0:
            print("No result csv files were stored in this address")
            sys.exit()
    
    for csvfile in csvresults_files:
        #reading the csv file
        df = pd.read_csv(csvfile)
        df = df.sample(frac=1)

        metrics = get_metrics_fth(df)
        RECALL.append(metrics['Recall']), F1SCORE.append(metrics['F1score']), ACCURACY.append(metrics['Accuracy']), PRECISION.append(metrics['Precision'])            
       
    #Computing and saving global metrics for threshold 0.5
    RESULTS_FTH = {'Accuracy': ACCURACY, 'Precision': PRECISION, 'Recall': RECALL, 'F1score': F1SCORE}
    AVG = []
    STD = []
    METRICS = []
    for i in RESULTS_FTH:
        AVG.append(np.mean(RESULTS_FTH[i]))
        STD.append(np.std(RESULTS_FTH[i]))
        METRICS.append(i)

    dfr = pd.DataFrame({
            'Metrics': METRICS,
            'AVG Scores': AVG,
            'STD': STD
    })
      
    dfr.to_csv(args.results_savepath + 'Results_fth.csv')
    if args.plots:
        createbarplot(dfr, args.results_savepath, "Scores_fth")

if __name__ == "__main__":
    main()