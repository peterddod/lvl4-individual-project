from sunau import Au_read
from keras.datasets import mnist, cifar10, fashion_mnist
import tensorflow_datasets as tfds
import torchvision.transforms.functional as VF
from torchvision.transforms import AutoAugmentPolicy, AutoAugment
import numpy as np
from torch import reshape, from_numpy
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import fbeta_score
import torch


def scale(X):
    return (X - X.mean().item())/(X.std().item()*10)

def label_vectoriser(labels, classes, label_smoothing=0.1):
    n = len(labels)

    y = torch.zeros([n,classes])

    for i in range(len(y)):
        for j in range(len(y[i])):
            onehot = 0
            if j == labels[i]:
                onehot = 1

            y[i,j] = (1-label_smoothing)*onehot + label_smoothing/classes

    return y

def zca_whitening_matrix(X):
    """
    Function to compute ZCA whitening matrix (aka Mahalanobis whitening).
    INPUT:  X: [M x N] matrix.
        Rows: Variables
        Columns: Observations
    OUTPUT: ZCAMatrix: [M x M] matrix
    """
    X = X.view((X.size()[0], -1))
    # Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / N
    sigma = np.cov(X, rowvar=True) # [M x M]
    # Singular Value Decomposition. X = U * np.diag(S) * V
    U,S,V = np.linalg.svd(sigma)
        # U: [M x M] eigenvectors of sigma.
        # S: [M x 1] eigenvalues of sigma.
        # V: [M x M] transpose of U
    # Whitening constant: prevents division by zero
    epsilon = 1e-5
    # ZCA Whitening matrix: U * Lambda * U'
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T)) # [M x M]
    return ZCAMatrix

def load_mnist(scaled=True, label_smoothing=0.1, augment=False, zca_whitening=False):
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    train_X = reshape(from_numpy(train_X), (-1, 1, 28, 28))
    test_X = reshape(from_numpy(test_X), (-1, 1, 28, 28))

    if augment:
        augmenter = AutoAugment(AutoAugmentPolicy.SVHN)
        train_X =  augmenter(train_X)

    train_X = train_X.float()
    test_X = test_X.float()

    if scaled:
        train_X = scale(train_X)
        test_X = scale(test_X)

    if zca_whitening:
        zca_m = zca_whitening_matrix(train_X)
        train_X = zca_m.mm(train_X)
        zca_m = zca_whitening_matrix(test_X)
        train_X = zca_m.mm(test_X)

    y = label_vectoriser(train_y, 10, label_smoothing)

    return {
        'train_X': train_X, 
        'train_y': y, 
        'test_X': test_X, 
        'test_y': test_y,
    }

def load_cifar10(scaled=True, label_smoothing=0.1, augment=False, zca_whitening=False):
    (train_X, train_y), (test_X, test_y) = cifar10.load_data()

    train_X = reshape(from_numpy(train_X), (-1, 3, 32, 32))
    test_X = reshape(from_numpy(test_X), (-1, 3, 32, 32))

    if augment:
        augmenter = AutoAugment(AutoAugmentPolicy.CIFAR10)
        train_X =  augmenter(train_X)

    train_X = train_X.float()
    test_X = test_X.float()

    if scaled:
        train_X = scale(train_X)
        test_X = scale(test_X)

    if zca_whitening:
        zca_m = zca_whitening_matrix(train_X)
        train_X = zca_m.mm(train_X)
        zca_m = zca_whitening_matrix(test_X)
        train_X = zca_m.mm(test_X)

    y = label_vectoriser(train_y, 10, label_smoothing)

    return {
        'train_X': train_X, 
        'train_y': y, 
        'test_X': test_X, 
        'test_y': test_y,
    }

def load_fashionmnist(scaled=True, label_smoothing=0.1, augment=True, zca_whitening=False):
    (train_X, train_y), (test_X, test_y) = fashion_mnist.load_data()

    train_X = reshape(from_numpy(train_X), (-1, 1, 28, 28))
    test_X = reshape(from_numpy(test_X), (-1, 1, 28, 28))

    if augment:
        augmenter = AutoAugment(AutoAugmentPolicy.SVHN)
        train_X =  augmenter(train_X)

    train_X = train_X.float()
    test_X = test_X.float()

    if scaled:
        train_X = scale(train_X)
        test_X = scale(test_X)

    if zca_whitening:
        zca_m = zca_whitening_matrix(train_X)
        train_X = zca_m.mm(train_X)
        zca_m = zca_whitening_matrix(test_X)
        train_X = zca_m.mm(test_X)

    y = label_vectoriser(train_y, 10, label_smoothing)

    return {
        'train_X': train_X, 
        'train_y': y, 
        'test_X': test_X, 
        'test_y': test_y,
    }

def load_norb(scaled=True, label_smoothing=0.1, augment=False, zca_whitening=False):
    ds = tfds.load('smallnorb', split=['train', 'test'], as_supervised=True, batch_size=-1)

    train_X = VF.resize(reshape(from_numpy(ds[0][0].numpy()), (-1, 1, 96, 96)).type(torch.uint8), (32,32))
    test_X = VF.resize(reshape(from_numpy(ds[1][0].numpy()), (-1, 1, 96, 96)).type(torch.uint8), (32,32))

    if augment:
        augmenter = AutoAugment(AutoAugmentPolicy.SVHN)
        train_X =  augmenter(train_X)

    train_X = train_X.float()
    test_X = test_X.float()

    if scaled:
        train_X = scale(train_X)
        test_X = scale(test_X)

    if zca_whitening:
        zca_m = zca_whitening_matrix(train_X)
        train_X = zca_m.mm(train_X)
        zca_m = zca_whitening_matrix(test_X)
        train_X = zca_m.mm(test_X)

    y = label_vectoriser(ds[0][1].numpy(), 5, label_smoothing)

    return {
        'train_X': train_X, 
        'train_y': y, 
        'test_X': test_X, 
        'test_y': ds[1][1].numpy(),
    }

def load_yaleface(scaled=True, label_smoothing=0.1):
    (train_X, train_y), (test_X, test_y) = cifar10.load_data()


    train_X = reshape(from_numpy(train_X), (50000, 3, 32, 32))
    test_X = reshape(from_numpy(test_X), (10000, 3, 32, 32))

    if scaled:
        train_X = scale(train_X)
        test_X = scale(test_X)

    y = label_vectoriser(train_y, 10, label_smoothing)

    return {
        'train_X': train_X, 
        'train_y': y, 
        'test_X': test_X, 
        'test_y': test_y,
    }

def evaluation_summary(description, predictions, true_labels):
    print("Evaluation for: " + description)
    precision = precision_score(predictions, true_labels, average="macro")
    recall = recall_score(predictions, true_labels, average="macro")
    accuracy = accuracy_score(predictions, true_labels)
    f1 = fbeta_score(predictions, true_labels, beta=1, average="macro")
    print("Classifier '%s' has Acc=%0.3f P=%0.3f R=%0.3f F1=%0.3f" % (description,accuracy,precision,recall,f1))
    print(classification_report(predictions, true_labels, digits=3))
    print('\nConfusion matrix:\n',confusion_matrix(true_labels, predictions))