import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans


def autoencode(X, a, b, out_weights=True):
    temp = X.mm(a)                                          # input * weights
    ae_H = torch.add(temp, b)                               # aX + b

    if not out_weights:
        return ae_H

    ae_H_pinv = torch.pinverse(ae_H)                        # H^-1
    beta = ae_H_pinv.mm(X)                                  # output weights

    return torch.pinverse(beta), ae_H                       # direct links, hidden layer output


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


def filtersynth(X, k, s=3, stride=1):
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