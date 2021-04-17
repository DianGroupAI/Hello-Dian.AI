import numpy as np

from .modules import Module


class Sigmoid(Module):

    def forward(self, x):

    def backward(self, delta):


class Tanh(Module):

    def forward(self, x):

    def backward(self, delta):


class ReLU(Module):

    def forward(self, x):

    def backward(self, delta):


class Softmax(Module):

    def forward(self, x):

    def backward(self, delta):


class Loss(object):
    """
    >>> criterion = CrossEntropyLoss(n_classes)
    >>> ...
    >>> for epoch in n_epochs:
    ...     ...
    ...     probs = model(x)
    ...     loss = criterion(probs, target)
    ...     model.backward(loss.backward())
    ...     ...
    """
    def __init__(self, n_classes):
        self.n_classes = n_classes
    
    def __call__(self, probs, targets):
        self.probs = probs
        self.targets = targets
        ...
    
    def backward(self):


class SoftmaxLoss(Loss):

    def __call__(self, probs, targets):
        super(SoftmaxLoss, self).__call__(probs, targets)

    def backward(self, delta):


class CrossEntropyLoss(Loss):

    def __call__(self, probs, targets):
        super(SoftmaxLoss, self).__call__(probs, targets)

    def backward(self):
