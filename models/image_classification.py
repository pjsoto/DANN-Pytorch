import os
import sys
import torch
import numpy as np
import pandas as pd
from progress.bar import Bar
from models.architecture import Architectures
from utils.tools import *
class IMGCLASSIFIER():
    def __init__(self, args, DATASET):
        self.lambd = 0
        self.args = args
        self.dataset = DATASET
        self.model = Architectures(args).model.to(self.args.device).float()
        if self.args.phase == 'train':
            self.optimizer = torch.optim.SGD(self.model.parameters(), self.args.learningrate, momentum=0.9)
        if self.args.phase == 'test':
            self.model.eval()

    def Loss(self):
        lc = -1 * self.labels*torch.log(torch.maximum(self.class_pred, 1e-3 * torch.ones_like(self.class_pred)))
        
        loss_sum = torch.sum(lc, dim = 1)
        loss_masked = self.mask[:,0] * loss_sum
        
        self.classification_loss = torch.sum(loss_masked)/torch.sum(self.mask)
        if self.args.task == "classification":
            return self.classification_loss
        elif self.args.task == "domain_adaptation":
            ld = -1 * (self.domain_labels * torch.log(torch.maximum(self.domain_pred, 1e-3 * torch.ones_like(self.domain_pred))) + (1 - self.domain_labels) * torch.log(torch.maximum(1 - self.domain_pred, 1e-3 * torch.ones_like(self.domain_pred))))
            #ld = -1 * self.domain_labels * torch.log(torch.maximum(self.domain_pred, 1e-3 * torch.ones_like(self.domain_pred)))
            
            loss_sum_d = torch.sum(ld, dim = 1)
            self.domainclassifier_loss = loss_sum_d.mean()
            return self.classification_loss + self.domainclassifier_loss
    
    def LearningrateDecayScheduler(self):
        p = self.epoch/self.args.max_epochs
        lr_ = self.optimizer.param_groups[0]['lr']
        self.optimizer.param_groups[0]['lr'] = lr_ / (1. + 10 * p)**0.75
    
    def LambdaScheduler(self):
        p = self.epoch/self.args.max_epochs
        self.lambd = 2./(1 + np.exp(-20 * p)) - 1
        self.model.set_lambda(self.lambd)
    
    def Train(self):
        
        BestEpoch = 0
        Patience = 0
        BestF1_Validation = 0
        BestLD_Validation = 0

        TrainC_Loss = []
        ValidC_Loss = []

        TrainD_Loss = []
        ValidD_Loss = []

        Train_Ac = []
        Valid_Ac = []
        
        Train_Pr = []
        Valid_Pr = []
        
        Train_Re = []
        Valid_Re = []
        
        Train_F1 = []
        Valid_F1 = []

        LAMBDAS = [] 

        self.epoch = 0
        self.model.set_lambda(0)
        LAMBDAS.append(0)
        while self.epoch < self.args.max_epochs:
            # Open a file in order to save the training history
            f = open(self.args.checkpoints_savepath + "Log.txt","a")
            print("-" * 100)
            print(f"epoch {self.epoch}/{self.args.max_epochs}")
            print("Training...")
            f.write("-" * 50 + "\n")
            f.write(f"epoch {self.epoch}/{self.args.max_epochs}\n")
            f.write("Training...\n")

            self.model.train()
            c_epoch_loss = []
            d_epoch_loss = []
            groundtruth_array = []
            predictions_array = []
            batch_counter = 0
            bar =  Bar('Processing', max = len(self.dataset.train_loader))
            for batch_data in self.dataset.train_loader:
                 data, self.labels, self.mask = batch_data['x'].to(self.args.device), batch_data['y'].to(self.args.device), batch_data['m'].to(self.args.device)
                 self.optimizer.zero_grad()
                 if self.args.task == "classification":
                     self.class_pred = self.model(data.float())
                 elif self.args.task == "domain_adaptation":
                     self.domain_labels = batch_data['d'].to(self.args.device)
                     self.class_pred, self.domain_pred = self.model(data.float())
                     
                 loss = self.Loss()
                 loss.backward()
                 self.optimizer.step()
                 if batch_counter == 0:
                    predictions_array = np.argmax(self.class_pred.detach().cpu().numpy(), 1)
                    groundtruth_array = np.argmax(self.labels.cpu().numpy(), 1)
                    domainmasks_array = self.mask.cpu().numpy()
                 else:
                    predictions_array = np.concatenate((predictions_array, np.argmax(self.class_pred.detach().cpu().numpy(), 1)), axis = 0)
                    groundtruth_array = np.concatenate((groundtruth_array, np.argmax(self.labels.cpu().numpy(), 1)), axis = 0)
                    domainmasks_array = np.concatenate((domainmasks_array, self.mask.cpu().numpy()), axis = 0)
                 
                 c_epoch_loss.append(self.classification_loss.item())
                 if self.args.task == "domain_adaptation":
                     d_epoch_loss.append(self.domainclassifier_loss.item())
                 batch_counter += 1

                 bar.next()
            bar.finish()
            
            # Computing the validation loss and metrics
            # Taking out the target data
            source_indexs = np.transpose(np.array(np.where(domainmasks_array == 1)))
            C_Loss = (np.sum(c_epoch_loss))/len(c_epoch_loss)
            
            Acc, Pre, Rec, F1 = ComputeMetrics(groundtruth_array[source_indexs[:,0]], predictions_array[source_indexs[:,0]])
            if self.args.task == "classification":
                print(f"epoch {self.epoch} average loss: {C_Loss:.4f} accuracy: {Acc:.2f} precision: {Pre:.2f} recall: {Rec:.2f} F1-Score: {F1:.2f}")
                f.write(f"epoch {self.epoch} average loss: {C_Loss:.4f} accuracy: {Acc:.2f} precision: {Pre:.2f} recall: {Rec:.2f} F1-Score: {F1:.2f}\n")
            elif self.args.task == "domain_adaptation":
                D_Loss = (np.sum(d_epoch_loss))/len(d_epoch_loss)
                TrainD_Loss.append(D_Loss)
                print(f"epoch {self.epoch} c average loss: {C_Loss:.4f} d average loss: {D_Loss:.4f} accuracy: {Acc:.2f} precision: {Pre:.2f} recall: {Rec:.2f} F1-Score: {F1:.2f}")
                f.write(f"epoch {self.epoch} average loss: {C_Loss:.4f} accuracy: {Acc:.2f} precision: {Pre:.2f} recall: {Rec:.2f} F1-Score: {F1:.2f}\n")

            #Storing the training results
            TrainC_Loss.append(C_Loss)
            Train_Ac.append(Acc/100)
            Train_Pr.append(Pre/100)
            Train_Re.append(Rec/100)
            Train_F1.append(F1/100)

            # Computing the validation metrics
            print("Validating...")
            f.write("Validating...\n")
            self.model.eval()
            c_epoch_loss = []
            d_epoch_loss = []
            groundtruth_array = []
            predictions_array = []
            batch_counter = 0
            bar =  Bar('Processing', max = len(self.dataset.valid_loader))
            for batch_data in self.dataset.valid_loader:
                data, self.labels, self.mask = batch_data['x'].to(self.args.device), batch_data['y'].to(self.args.device), batch_data['m'].to(self.args.device)
                with torch.no_grad():
                    if self.args.task == "classification":
                        self.class_pred = self.model(data.float())
                    elif self.args.task == "domain_adaptation":
                        self.domain_labels = batch_data['d'].to(self.args.device)
                        self.class_pred, self.domain_pred = self.model(data.float())
                        
                    loss = self.Loss()
                    if batch_counter == 0:
                        predictions_array = np.argmax(self.class_pred.cpu().numpy(), 1)
                        groundtruth_array = np.argmax(self.labels.cpu().numpy(), 1)
                        domainmasks_array = self.mask.cpu().numpy()
                    else:
                        predictions_array = np.concatenate((predictions_array, np.argmax(self.class_pred.cpu().numpy(), 1)), axis = 0)
                        groundtruth_array = np.concatenate((groundtruth_array, np.argmax(self.labels.cpu().numpy(), 1)), axis = 0)
                        domainmasks_array = np.concatenate((domainmasks_array, self.mask.cpu().numpy()), axis = 0)
                    c_epoch_loss.append(self.classification_loss.item())
                    if self.args.task == "domain_adaptation":
                        d_epoch_loss.append(self.domainclassifier_loss.item())
                batch_counter += 1
                bar.next()
            bar.finish()
            source_indexs = np.transpose(np.array(np.where(domainmasks_array == 1)))
            C_Loss = (np.sum(c_epoch_loss))/len(c_epoch_loss)
            Acc, Pre, Rec, F1 = ComputeMetrics(groundtruth_array[source_indexs[:,0]], predictions_array[source_indexs[:,0]])

            if self.args.task == "classification":
                print(f"epoch {self.epoch} average loss: {C_Loss:.4f} accuracy: {Acc:.2f} precision: {Pre:.2f} recall: {Rec:.2f} F1-Score: {F1:.2f}")
                f.write(f"epoch {self.epoch} average loss: {C_Loss:.4f} accuracy: {Acc:.2f} precision: {Pre:.2f} recall: {Rec:.2f} F1-Score: {F1:.2f}\n")
            elif self.args.task == "domain_adaptation":
                D_Loss = (np.sum(d_epoch_loss))/len(d_epoch_loss)
                ValidD_Loss.append(D_Loss)
                print(f"epoch {self.epoch} c average loss: {C_Loss:.4f} d average loss: {D_Loss:.4f} accuracy: {Acc:.2f} precision: {Pre:.2f} recall: {Rec:.2f} F1-Score: {F1:.2f}")
                f.write(f"epoch {self.epoch} average loss: {C_Loss:.4f} accuracy: {Acc:.2f} precision: {Pre:.2f} recall: {Rec:.2f} F1-Score: {F1:.2f}\n")
            

            #Storing the training results
            ValidC_Loss.append(C_Loss)
            Valid_Ac.append(Acc/100)
            Valid_Pr.append(Pre/100)
            Valid_Re.append(Rec/100)
            Valid_F1.append(F1/100)

            if self.args.task == "classification":
                if np.isnan(C_Loss):
                    print("Nan value detected! Please verify your code...")
                    sys.exit()
                else:
                    if F1 > BestF1_Validation:
                        BestF1_Validation = F1
                        BestEpoch = self.epoch
                        Patience = 0
                        torch.save(self.model.state_dict(), self.args.checkpoints_savepath + self.args.architecture + ".pth")
                        print("[!]Best model saved\n")
                        f.write("[!]Best model saved\n")
                    else:
                        Patience += 1
                        if Patience > self.args.patience:
                            print(f"Resume: Training finished with best model's F1-score: {BestF1_Validation:.2F} obtained at epoch: {BestEpoch}")
                            f.write(f"Resume: Training finished with best model's F1-score: {BestF1_Validation:.2F} obtained at epoch: {BestEpoch}\n")
                            f.close()
                            break
            elif self.args.task == "domain_adaptation":
                if np.isnan(C_Loss) or np.isnan(D_Loss):
                    print("Nan value detected! Please verify your code...")
                    sys.exit()
                else:
                    if self.lambd != 0:
                        FLAG = False
                        if BestLD_Validation < D_Loss and D_Loss < 1:
                            if BestF1_Validation < F1:
                                BestLD_Validation = D_Loss
                                BestF1_Validation = F1
                                BestEpoch = self.epoch
                                print('[!]Saving the best ideal model...\n')
                                f.write('[!]Saving the best ideal model...\n')
                                FLAG = True
                            elif np.abs(BestF1_Validation - F1) < 2:
                                BestLD_Validation = D_Loss
                                BestEpoch = self.epoch
                                print('[!]Saving best model according best Discriminator loss...\n')
                                f.write('[!]Saving best model according best Discriminator loss...\n')
                                FLAG = True
                        elif (BestF1_Validation < F1) and (np.abs(BestLD_Validation - D_Loss) < 0.2):
                            BestF1_Validation = F1
                            BestEpoch = self.epoch
                            print('[!]Saving best model according best Classifier perfoemance...\n')
                            f.write('[!]Saving best model according best Classifier performance...\n')
                            FLAG = True

                        if FLAG:
                            Patience = 0
                            torch.save(self.model.state_dict(), self.args.checkpoints_savepath + self.args.architecture + ".pth")
                            print("[!]Best model saved\n")
                            f.write("[!]Best model saved\n")
                        else:
                            Patience += 1
                            if Patience > self.args.patience:
                                print(f"Resume: Training finished with best model's F1-score: {BestF1_Validation:.2F} obtained at epoch: {BestEpoch}")
                                f.write(f"Resume: Training finished with best model's F1-score: {BestF1_Validation:.2F} obtained at epoch: {BestEpoch}\n")
                                f.close()
                                break
                            print('[!] The Model has not been considered as suitable for saving procedure.')
                    else:
                        print('Model was warming up! No saving procedure accomplished during this stage')


            self.epoch += 1
            f.close()
            self.LearningrateDecayScheduler()
            self.LambdaScheduler()
            LAMBDAS.append(self.lambd)
        
        if self.args.training_graphs:
            if self.args.task == 'classification':
                createplot(TrainC_Loss, 
                           ValidC_Loss, 
                           np.arange(len(TrainC_Loss)), 
                           self.args.checkpoints_savepath,
                           "Cross Entropy Loss")
            elif self.args.task == 'domain_adaptation':
                createplotda(TrainC_Loss,
                             ValidC_Loss,
                             TrainD_Loss,
                             ValidD_Loss,
                             LAMBDAS,
                             np.arange(len(TrainC_Loss)),
                             self.args.checkpoints_savepath,
                             "Losses"
                             )
            
            createplot(Train_F1, 
                       Valid_F1, 
                       np.arange(len(Train_F1)), 
                       self.args.checkpoints_savepath,
                       "F1 Score")

            createplot(Train_Re, 
                       Valid_Re, 
                       np.arange(len(Train_Re)), 
                       self.args.checkpoints_savepath,
                       "Recall")

            createplot(Train_Pr, 
                       Valid_Pr, 
                       np.arange(len(Train_Pr)), 
                       self.args.checkpoints_savepath,
                       "Precision")
            
            createplot(Train_Ac, 
                       Valid_Ac, 
                       np.arange(len(Train_Ac)), 
                       self.args.checkpoints_savepath,
                       "Accuracy")
    
    def Evaluates(self):

        batch_counter = 0
        groundtruth_array = []
        predictions_array = []
        bar =  Bar('Evaluating test set', max = len(self.dataset.test_loader))
        for batch_data in self.dataset.test_loader:
                data, labels = batch_data['x'].to(self.args.device), batch_data['y'].to(self.args.device)
                with torch.no_grad():
                    if self.args.task == "classification":
                        predictions = self.model(data.float())
                    elif self.args.task == "domain_adaptation":
                        predictions, _ = self.model(data.float())
                    probs = predictions.cpu().numpy()
                    label = labels.cpu().numpy()
                    if batch_counter == 0:
                        predictions_array = np.argmax(probs, 1)
                        groundtruth_array = np.argmax(label, 1)
                    else:
                        predictions_array = np.concatenate((predictions_array, np.argmax(probs, 1)), axis = 0)
                        groundtruth_array = np.concatenate((groundtruth_array, np.argmax(label, 1)), axis = 0)
                   
                batch_counter += 1
                bar.next()
        bar.finish()
             
        df = pd.DataFrame({
            'TrueLabels': groundtruth_array,
            'Predictions': predictions_array,
        })

        df.to_csv(self.args.results_savepath + 'Predictions.csv')