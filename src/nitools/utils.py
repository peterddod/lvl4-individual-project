from keras.datasets import mnist, cifar10
import numpy as np
from torch import reshape, from_numpy
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import fbeta_score


def scale(X):
    return (X - X.mean())/(X.std()*10)

def load_mnist(scaled=True):
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    train_X = np.reshape(train_X.ravel(), (-1,784))
    test_X = np.reshape(test_X.ravel(), (-1,784))

    if scaled:
        train_X = scale(train_X)
        test_X = scale(test_X)

    y = np.zeros([60000,10])

    for i in range(len(y)):
        for j in range(len(y[i])):
            if j == train_y[i]:
                y[i,j] = 1
            else:
                y[i,j] 

    return {
        'train_X': train_X, 
        'train_y': y, 
        'test_X': test_X, 
        'test_y': test_y,
    }

def load_cifar10(scaled=True):
    (train_X, train_y), (test_X, test_y) = cifar10.load_data()


    train_X = reshape(from_numpy(train_X), (50000, 3, 32, 32))
    test_X = reshape(from_numpy(test_X), (10000, 3, 32, 32))

    if scaled:
        train_X = scale(train_X)
        test_X = scale(test_X)

    y = np.zeros([50000,10])

    for i in range(len(y)):
        for j in range(len(y[i])):
            if j == train_y[i]:
                y[i,j] = 1
            else:
                y[i,j] 

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