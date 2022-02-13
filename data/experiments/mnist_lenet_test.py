from sklearn.metrics import accuracy_score
import torch
from nitools.classifiers import PAE_RVFL, AE_ELM, ELM 
from nitools.models.LeNet5 import LeNet5
from nitools.operations import resetseed
from nitools.utils import load_mnist 
import time as t


def str_params(params):
    out = ""

    for param, val in params.items():
        out += f'{param}:{val};'
    
    return out

if __name__ == '__main__':
    SEEDS = [22, 432, 63, 754, 3456, 5, 6677, 876, 213, 5444]
    RESULTS_HEADER = 'name, parameters, neurons, seed, acc, train_time, test_time'
    neurons = [250,500,1000,2000,4000]

    models = {
        'lenet-ae-elm': {
            'classifier': AE_ELM.AE_ELM,
            'parameters': {
                'h_size': neurons,
            },
        },
        'lenet-elm': {
            'classifier': AE_ELM.AE_ELM,
            'parameters': {
                'h_size': neurons,
            },
        },
        'lenet-pae-rvfl': {
            'classifier': PAE_RVFL.PAE_RVFL,
            'parameters': {
                'h_size': [75,150,300,600,1200],
                'subnets': [1,2,3,4,6,8,10],
            },
        },
    }

    with open('results.csv', 'w') as f:
        f.write(RESULTS_HEADER + '\n')
        f.close()

    mnist = load_mnist()

    mnist_in = 784
    mnist_class = 10

    X = torch.from_numpy(mnist['train_X']).float()[:60000]
    y = torch.from_numpy(mnist['train_y']).float()[:60000]
    tX = torch.from_numpy(mnist['test_X']).float()

    for name, model in models.items():
        classifier = model['classifier']
        params = {}

        for seed in SEEDS:
            for weight_train in [True,False]:
                model = LeNet5(classifier(params), weight_train=weight_train)

                print('\n===============================')
                print(f'{name} - {seed} - {str_params(params)} - weight_train: {weight_train}')

                # train model
                print('Starting training...')
                train_start = t.time()

                model.train(X, y)

                train_end = t.time()
                print('Training finished!')

                # run prediction 
                print('Making predictions...')
                pred_start = t.time()

                pred = model.predict(tX)

                pred_end = t.time()

                # get accuracy
                acc =  accuracy_score(pred, mnist['test_y'])

                # create string of tuple (name, parameters, neurons, seed, acc, train_time, test_time)
                results = f'{name}, {str_params(params)}, {neurons}, {seed}, {acc}, {train_end-train_start}, {pred_end-pred_start}'

                print(f'''#### RESULTS
                {RESULTS_HEADER}
                {results}''')

                # append results to file
                with open('results.csv', 'a') as f:
                    f.write(results + '\n')
                    f.close()

        