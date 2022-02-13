import torch
from nitools.classifiers import PAE_RVFL, AE_ELM, ELM 
from nitools.models.LeNet5 import LeNet5
from nitools.operations import resetseed
from nitools.utils import load_mnist 
import time as t


if __name__ == '__main__':
    SEEDS = [22, 432, 63, 754, 3456, 5, 6677, 876, 213, 5444]
    RESULTS_HEADER = 'name, parameters, neurons, seed, acc, train_time, test_time'
    neurons = [250,500,1000,2000,4000]

    models = {
        'baseline-elm': {
            'model': None,
            'classifier': ELM.ELM,
            'parameters': {
                'h_size': neurons,
            },
        },
        'lenet-ae-elm': {
            'model': LeNet5,
            'classifier': AE_ELM.AE_ELM,
            'parameters': {
                'h_size': neurons,
                'weight_train': [True, False],
            },
        },
        'lenet-elm': {
            'model': LeNet5,
            'classifier': AE_ELM.AE_ELM,
            'parameters': {
                'h_size': neurons,
                'weight_train': [True, False],
            },
        },
        'lenet-pae-rvfl': {
            'model': LeNet5,
            'classifier': PAE_RVFL.PAE_RVFL,
            'parameters': {
                'h_size': [75,150,300,600,1200],
                'subnets': [1,2,3,4,6,8,10],
                'weight_train': [True, False],
            },
        },
    }

    mnist = load_mnist()

    mnist_in = 784
    mnist_class = 10

    model = object
    classifier = object
    name = string
    params = {}

    X = torch.from_numpy(mnist['train_X']).float()[:60000]
    y = torch.from_numpy(mnist['train_y']).float()[:60000]

    for seed in SEEDS:
        entry = None

        if model == None:
            entry = classifier
        else:
            entry = 

        print('===============================')
        print(f'{name} - {seed} - {params}')

        # train model
        print('Starting training...')
        train_start = t.time_ns()

        model.train()

        train_end = t.time_ns()
        print('Training finished!')

        # run prediction 
        print('Making predictions...')
        pred_start = t.time_ns()

        model.predict()

        pred_end = t.time_ns()

        # get accuracy
        acc = 8

        # create string of tuple (name, parameters, neurons, seed, acc, train_time, test_time)
        results = f'{name}, {params}, {neurons}, {seed}, {acc}, {train_end-train_start}, {pred_end-pred_start}'

        print(f'''#### RESULTS
        {RESULTS_HEADER}
        {results}''')

        