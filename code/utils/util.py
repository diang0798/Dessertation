import torch
from torch.autograd import Variable


def get_variable(x):
    x = Variable(x)
    return x.cuda() if torch.cuda.is_available() else x
