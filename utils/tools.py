import os
import numpy as np
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

def ComputeMetrics(true_labels, predicted_labels):
    accuracy = 100*accuracy_score(true_labels, predicted_labels)
    f1score = 100*f1_score(true_labels, predicted_labels, average = 'macro')
    recall = 100*recall_score(true_labels, predicted_labels, average = 'macro')
    prescision = 100*precision_score(true_labels, predicted_labels, average = 'macro')

    return accuracy, prescision, recall, f1score

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i,round(y[i], 2),round(y[i], 2))

def createplot(yt, yv, x, savepath, title):

    plt.figure()

    plt.plot(x, yt, color = 'red', label = 'train')
    plt.plot(x, yv, color = 'blue',label = 'validation')
    plt.xlabel("Number of epochs")
    plt.ylabel("Scores")
    plt.title(title)
    plt.legend()
    plt.show()
    plt.savefig(savepath + title + ".jpg")
    plt.close()

def createplotda(yt1, yv1, yt2, yv2, lambd, x, savepath, title):
    plt.figure()

    plt.plot(x, yt1, color = 'red', label = 'train c')
    plt.plot(x, yv1, color = 'blue',label = 'validation c')

    plt.plot(x, yt2, color = 'red', linestyle='dashed', label = 'train d')
    plt.plot(x, yv2, color = 'blue',linestyle='dashed', label = 'validation d')

    plt.plot(x, lambd, color = 'black', label = 'lambda')

    plt.xlabel("Number of epochs")
    plt.ylabel("Scores")
    plt.title(title)
    plt.legend()
    plt.show()
    plt.savefig(savepath + title + ".jpg")
    plt.close()

def createbarplot(df, savepath, title):
    sns.barplot(data = df, x = "Metrics", y = "AVG Scores")
    plt.errorbar(df['Metrics'].values, df['AVG Scores'].values, df['STD'].values, fmt = 'ko')
    addlabels(df['Metrics'].values, df['AVG Scores'].values)
    plt.ylim(0, 1.2)
    plt.show()
    plt.savefig(savepath + title + ".jpg")
    plt.close()

def get_metrics_fth(df):

    y_true = df["TrueLabels"].values
    y_pred = df["Predictions"].values

    recall = recall_score(y_true, y_pred, average=None)
    precision = precision_score(y_true, y_pred, average=None)
    accuracy = accuracy_score(y_true, y_pred)
    f1= f1_score(y_true, y_pred, average=None)

    return {'Accuracy': accuracy , 'Recall': recall, 'Precision': precision, 'F1score': f1}

