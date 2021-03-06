from iter import Model3c
import numpy as np
import torch
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from cifar import CIFAR10
import time as t

PATH = '/weights/'

if __name__ == '__main__':
    batch_size = 16
    cifar_train = CIFAR10(train=True)
    cifar_test = CIFAR10(train=False)
    train_loader = DataLoader(cifar_train, batch_size=batch_size)
    test_loader = DataLoader(cifar_test, batch_size=batch_size)
    model = Model3c()
    sgd = Adam(model.parameters(), lr=0.001)
    cost = MSELoss()
    SEEDS = [22, 432, 63, 754, 3456]#, 5, 6677, 876, 213, 5444]

    for epoch in [50]:
        for seed in SEEDS:
            start = t.time()
            for _epoch in range(epoch):
                model.train()
                for idx, (train_x, train_label) in enumerate(train_loader):
                    label_np = np.zeros((train_label.shape[0], 10))
                    sgd.zero_grad()
                    predict_y = model(train_x.float())
                    loss = cost(predict_y, train_label.float())
                    if idx % 10 == 0:
                        print('idx: {}, loss: {}'.format(idx, loss.sum().item()))
                    loss.backward()
                    sgd.step()

                correct = 0
                _sum = 0
                model.eval()
                for idx, (test_x, test_label) in enumerate(test_loader):
                    predict_y = model(test_x.float()).detach()
                    predict_ys = np.argmax(predict_y, axis=-1)
                    label_np = test_label.numpy()
                    _ = predict_ys == test_label
                    correct += np.sum(_.numpy(), axis=-1)
                    _sum += _.shape[0]

                acc = correct / _sum
                print(acc)
                print('accuracy: {:.2f}'.format(np.mean(acc)))
            end = t.time() 

            torch.save(model[0].state_dict(), f'{PATH}{seed}-1.pth')
            torch.save(model[3].state_dict(), f'{PATH}{seed}-2.pth')