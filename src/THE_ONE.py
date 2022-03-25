from datasets import MNIST, FSHN_MNIST, NORB, CIFAR10
from iter import Model
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch.nn import MSELoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from datasets import MNIST, FSHN_MNIST, NORB
from nitools.operations import resetseed
from nitools.models import LeNetPlus, LRF_ELM
from itertools import product
import time as t


class FileWriter():

    def __init__(self, filename, header):
        self.filename = filename
        self.header = header

        with open(f'{self.filename}.csv', 'w') as f:
            f.write(header + '\n')
            f.close()

    def __call__(self, info):
        with open(f'{self.filename}.csv', 'a') as f:
            f.write(info + '\n')
            f.close()


def iter_dispatch(name, model, dataset, dataset_id, seed, file):
    batch_size = 16

    data_train = dataset(train=True)
    data_test = dataset(train=False)
    train_loader = DataLoader(data_train, batch_size=batch_size)
    test_loader = DataLoader(data_test, batch_size=batch_size)

    for epoch in [5]:
        resetseed(seed)
        model = Model(out_size=5)
        sgd = SGD(model.parameters(), lr=0.01)

        cost = MSELoss()
        
        train_time = 0
        test_time = 0
        for _epoch in range(epoch):
            start = t.time()
            model.train()
            for idx, (train_x, train_label) in enumerate(train_loader):
                label_np = np.zeros(train_label.size())
                sgd.zero_grad()
                predict_y = model(train_x.float())
                loss = cost(predict_y, train_label.float())
                if idx % 10 == 0:
                    print('idx: {}, loss: {}'.format(idx, loss.sum().item()))
                loss.backward()
                sgd.step()
            end = t.time() 
            train_time += end-start

            start = t.time()
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
            end = t.time() 
            test_time += end-start
            acc = correct / _sum
            
            print('accuracy: {:.2f}'.format(acc))
        
            result = f'{name},{dataset_id},{_epoch+1},{seed},{acc},{train_time},{test_time}'
            
            print(file.header)
            print(result)

            file(result)


def ours_dispatch(name, model, h_size, subnets, dataset, dataset_id, seed, file):
    data = dataset().data

    X = data['train_X'].float()
    y = data['train_y'].float()
    tX = data['test_X'].float()

    for _h_size, _subnets in product(h_size, subnets):
        try:
            print('\n===============================')
            print(f'{name} - {seed} - {dataset_id} {_h_size} {_subnets}')

            resetseed(seed)

            model_i = model(h_size=_h_size, subnets=_subnets)

            # train model
            print('Starting training...')
            train_start = t.time()

            model_i.train(X, y)

            train_end = t.time()
            print('Training finished!')

            # run prediction 
            print('Making predictions...')
            pred_start = t.time()

            pred = model_i.predict(tX)

            pred_end = t.time()
            
            pred_arg = torch.zeros(len(tX))

            for i in range(len(pred)):
                pred_arg[i] = torch.argmax(pred[i])

            # get accuracy
            acc =  accuracy_score(pred_arg, data['test_y'])

            # create string of tuple ('name,dataset,h_size,subnets,seed,acc,train_time,test_time')
            results = f'{name},{dataset_id},{_h_size},{_subnets},{seed},{acc},{train_end-train_start},{pred_end-pred_start}'
            print(file.header)
            print(results)

            file(results)
        except:
            results = f'{name},{dataset_id},{_h_size},{_subnets},{seed},failed,failed,failed'
            file(results)


def elm_lrf_dispatch(name, model, dataset, dataset_id, seed, file):
    print('\n===============================')
    print(f'{name} - {seed} - {dataset_id}')

    data = dataset().data

    X = data['train_X'].float()
    y = data['train_y'].float()
    tX = data['test_X'].float()


    resetseed(seed)

    model_i = model()

    # train model
    print('Starting training...')
    train_start = t.time()

    model_i.train(X, y)

    train_end = t.time()
    print('Training finished!')

    # run prediction 
    print('Making predictions...')
    pred_start = t.time()

    pred = model_i.predict(tX)

    pred_end = t.time()
    
    pred_arg = torch.zeros(len(tX))

    for i in range(len(pred)):
        pred_arg[i] = torch.argmax(pred[i])

    # get accuracy
    acc =  accuracy_score(pred_arg, data['test_y'])

    # create string of tuple ('name,dataset,seed,acc,train_time,test_time')
    results = f'{name},{dataset_id},{seed},{acc},{train_end-train_start},{pred_end-pred_start}'
    
    print(file.header)
    print(results)

    file(results)


def run(seeds):
    iter_file = FileWriter('iter_results', 'name,dataset,epochs,seed,acc,train_time,test_time')
    ours_file = FileWriter('ours_results', 'name,dataset,h_size,subnets,seed,acc,train_time,test_time')
    elmlrf_file = FileWriter('elmlrf_results', 'name,dataset,seed,acc,train_time,test_time')

    for seed in seeds:
        for i, dataset in enumerate([MNIST, FSHN_MNIST, NORB]):     
            ours_dispatch('ours-lenet', LeNetPlus, [100,200,400,800], [1,2,4,6,8], dataset, i, seed, ours_file)
            try:
                iter_dispatch('iterative-lenet', Model, dataset, i, seed, iter_file)
            except:
                result = f'iterative-lenet,{i},fail,{seed},fail,fail,fail'
                iter_file(result)
            
            try:
                elm_lrf_dispatch('elm-lrf-lenet', LRF_ELM, dataset, i, seed, elmlrf_file)
            except:
                results = f'elm-lrf-lenet,{i},{seed},fail,fail,fail'
                elmlrf_file(results)

        # cifar 10
            # run iterative resnet8 training
            # run non iterative resnet8 training
                # run for different classifier setups
                # run with and without l1 penalty
            # run elm-lrf training
                # 128 receptive fields
