import os
import sys
import torch
import numpy as np
import torch.nn as nn
from torch import tensor
from collections import OrderedDict
from typing import Callable, Sequence, Type, Union



class GradReverse(torch.autograd.Function):
    lambd = 0
    @staticmethod
    def forward(ctx, x, lambd):
        lambd_ = torch.tensor(lambd, requires_grad = False)
        ctx.save_for_backward(lambd_)
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        lambd = ctx.saved_tensors[0]
        return -lambd * grad_output, None

def grad_reverse(x, lambd):
    
    return GradReverse.apply(x, lambd)

class GANINARCH(nn.Module):
    def __init__(self, args):
        super(GANINARCH, self).__init__()
        self.args = args
        self.lambd = 0
        in_channels = 3
        init_features = 32
        out_channels_lp = self.args.labelpredictor_classes
        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", nn.Conv2d(in_channels, init_features, kernel_size=5, stride=1, padding=3, bias=False)),
                    ("relu0", nn.ReLU()),
                    ("pool0", nn.MaxPool2d(kernel_size=2, stride=2, padding=1)),
                    ("conv1", nn.Conv2d(init_features, 48, kernel_size=5, stride=1, padding=3, bias=False)),
                    ("relu1", nn.ReLU()),
                    ("pool1", nn.MaxPool2d(kernel_size=2, stride=2, padding=1)),
                    ("flatten", nn.Flatten()),
                ]
            )
        )

        self.label_predictor = nn.Sequential(
            OrderedDict(
                [
                    ("lplinear0", nn.Linear(in_features = 4800, out_features = 100, bias = False)),
                    ("lprelu0", nn.ReLU()),
                    ("lplinear1", nn.Linear(in_features = 100, out_features = 100, bias = False)),
                    ("lprelu0", nn.ReLU()),
                    ("lplinear2", nn.Linear(in_features = 100, out_features = out_channels_lp, bias = False)),
                    ("softmax_lp", nn.Softmax(dim = 1)),
                ]
            )
        )

        self.domain_classifier = nn.Sequential(
            OrderedDict(
                [   ("dropout", nn.Dropout(p=0.25)),
                    ("dclinear0", nn.Linear(in_features = 4800, out_features = 100, bias = False)),
                    ("dcrelu0", nn.ReLU()),
                    ("dclinear1", nn.Linear(in_features = 100, out_features = 1, bias = False)),
                    ("softmax_cd", nn.Sigmoid()),
                ]
            )
        )
    def set_lambda(self, lambd):
        self.lambd = lambd
    def grad_reverse_(self, grad):
        grad_clone = grad.clone()
        return - self.lambd * grad_clone
    
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        features_vector = self.features(x)
        class_predictions = self.label_predictor(features_vector)
        if self.args.task == "classification":
            return class_predictions
        elif self.args.task == "domain_adaptation":
            features_vector_ = grad_reverse(features_vector, self.lambd)
            domain_predictions = self.domain_classifier(features_vector_)
            return class_predictions, domain_predictions

