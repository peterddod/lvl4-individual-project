import torch
from torch import nn
from torch.nn.functional import mse_loss, unfold
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import random


def lazyautoencode(X, h_size, c=0.1):
    a_in = orthrandom(size=(X.size()[1], h_size))
    b_in = orthrandom(h_size)

    temp = X.mm(a_in)                                          # input * weights
    h = torch.add(temp, b_in)                               # aX + b

    h_pinv = regpinv(h, c=c)                               # H^-1
    beta = h_pinv.mm(X)                                  # output weights

    return a_in, b_in, beta, h 


def autoencode(X, h_size, l, c=0.1):
    j=1

    a_in = orthrandom(size=(X.size()[1], h_size))
    b_in = orthrandom(h_size)

    h = torch.add(X.mm(a_in), b_in)

    a_out = regpinv(h, c=c).mm(X)
    b_out = torch.sqrt(mse_loss(h.mm(a_out), X))

    while j<l:
        a_in = torch.transpose(a_out, 0, 1)
        b_in = b_out

        h = torch.add(X.mm(a_in), b_in)

        a_out = regpinv(h, c=c).mm(X)
        b_out = torch.sqrt(mse_loss(h.mm(a_out), X))

        j += 1

    return a_in, b_in, a_out, h


def pretrain(X, a, b, sb=0.5, sc=0.5):
    if not (sb > 0):
        raise Exception('sb less than or equal to')
    elif not (sc >= 0 and sc < 2*sb):
        raise Exception('sc less than 0 or greater than 2*sb')

    for i in range(len(b)):
        w = a[:,i]  # points to the ith column of a
        bi = b[i]  # points to the ith element of b

        temp = torch.matmul(X, w[:,None])
        s = torch.add(temp, bi)
        s.sort()

        t_min = np.random.uniform(-sb, sb)
        t_max = np.random.uniform(-sb, sb)

        while not (t_max-t_min > sc):
            t_min = np.random.uniform(-sb, sb)
            t_max = np.random.uniform(-sb, sb)

        mean = (t_min+t_max)/2
        std = (t_max-t_min)/4

        ti = np.random.normal(mean, std, size=len(s))
        ti.sort()

        ti = torch.tensor(ti).float()

        e = torch.linalg.lstsq(s, ti).solution

        a[:,i] = w*e
        b[i] = bi*e

    return a, b


def filtersynth(X, k, s=3, stride=1, padding=0) :
    # input tensor (num_samples, channels, height, width)
    in_size = X.size()

    # create clustering algorithm with k clusters
    kmeans = MiniBatchKMeans(k*in_size[1])

    # cluster every s[1] x s[1] square on each image 
    for c in range(0,in_size[1]):
        for h in range(0,in_size[2]-s,stride):
            for w in range(0,in_size[3]-s,stride):
                block = torch.reshape(X[:, c, h:h+s, w:w+s], (in_size[0], s*s)).detach()

                kmeans = kmeans.partial_fit(block)

    # use centroids as kernels - (out_channels, in_channels, kH, kW)
    weights = torch.reshape(torch.tensor(kmeans.cluster_centers_).float(), (k, in_size[1], s, s))

    return weights


def regpinv(X, c=0.1):
    X = torch.nan_to_num(X)
    Xt = torch.transpose(X, 0, 1)
    out = []

    try:
        out = torch.pinverse(torch.add(Xt.mm(X), c*torch.eye(len(Xt)))).mm(Xt)
    except:
        out = torch.pinverse(X)

    return out


def resetseed(seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def dropout(a, b=None, p=0.5, device=None):
    weights = nn.init.uniform_(torch.empty(a.size()[1], device=device), a=0, b=1)

    b_dropout = b!=None

    for i, weight in enumerate(weights):
        if weight<p:
            a[:,i] = 0

            if b_dropout:
                b[i] = 0
    
    if b_dropout:
        return a, b

    return a


def orthrandom(size=(3,3), range=(0,1), device=None):
    if type(size)==int:
        x = nn.init.normal_(torch.empty(size, device=device), std=0.5, mean=0.5)
        return x / torch.linalg.norm(x)

    return nn.init.orthogonal_(nn.init.uniform_(torch.empty(size[0], size[1], device=device), a=-range[0], b=range[1]))