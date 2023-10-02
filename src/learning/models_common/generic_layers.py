"""
Generic layers used across models.
"""
import torch
from torch.nn import functional


class SoftmaxMixLayers(torch.nn.Linear):
    def forward(self, input):
        # the weight vector is out_dim x in_dim.
        # so we want to softmax along in_dim.
        weight = functional.softmax(self.weight, dim=1)
        return functional.linear(input, weight, self.bias)
