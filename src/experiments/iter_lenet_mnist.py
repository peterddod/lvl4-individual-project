from iter import model
import numpy as np
import torch
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from mnist import MNIST
import time as t

if __name__ == '__main__':
    batch_size = 16
    mnist_train = MNIST(train=True)
    mnist_test = MNIST(train=False)
    train_loader = DataLoader(mnist_train, batch_size=batch_size)
    test_loader = DataLoader(mnist_test, batch_size=batch_size)
    model = model.Model()
    sgd = Adam(model.parameters(), lr=0.01)
    cost = MSELoss()
    SEEDS = [22, 432, 63, 754, 3456, 5, 6677, 876, 213, 5444]
    RESULTS_HEADER = 'name, epochs, seed, acc, train_time, test_time'

    with open('results.csv', 'w') as f:
        f.write(RESULTS_HEADER + '\n')
        f.close()

    for epoch in [1,2,3]:
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
                print('accuracy: {:.2f}'.format(acc))
            end = t.time() 
            result = f'lenet5,{epoch},{seed},{acc},{end-start},0'
            with open('results.csv', 'a') as f:
                f.write(result + '\n')
                f.close()