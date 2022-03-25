from itertools import product
from sklearn.metrics import accuracy_score
import torch
import nitools
from nitools.classifiers import PAE_RVFL, PAE_ELM 
from nitools.models.LeNet5 import LeNet5
from nitools.operations import resetseed
from nitools.utils import load_mnist 
import time as t

FILENAME = 'LOCAL_conv_weight_results.csv'

def str_params(params):
    out = ""

    for param, val in params.items():
        out += f'{param}:{val};'
    
    return out

def get_params(params):
    # seperate parameters into lists 
    names = []
    params_list = []

    for name, values in params.items():
        names.append(name)
        params_list.append(values)

    # find cartisian product of parameters
    params_product = product(*params_list)

    # return parameters and parameters names
    return params_product, names

def combine_params(params, name, def_params):
    params_dict = {}

    for i in range(len(params)):
        params_dict[name[i]] = params[i]

    return {**def_params, **params_dict}

def run_experiment(model, name, seed, params, weight_init):
    print('\n===============================')
    print(f'{name} - {seed} - {str_params(params)}')

    resetseed(seed)

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
    
    pred_arg = torch.zeros(10000)

    for i in range(len(pred)):
        pred_arg[i] = torch.argmax(pred[i])

    # get accuracy
    acc =  accuracy_score(pred_arg, mnist['test_y'])

    # create string of tuple (name, subnets, neurons, seed, acc, train_time, test_time)
    results = f'{name},{seed},{acc},{train_end-train_start},{pred_end-pred_start},{weight_init}'

    print(f'''#### RESULTS
    {RESULTS_HEADER}
    {results}''')

    # append results to file
    with open(FILENAME, 'a') as f:
        f.write(results + '\n')
        f.close()


if __name__ == '__main__':
    # Initialise variables
    SEEDS = [22, 432, 63, 754, 3456, 5, 6677, 876, 213, 5444]
    RESULTS_HEADER = 'name,seed,acc,train_time,test_time,weight_init'

    # Initialise modles
    models = {
        'lenet-pae-elm': {
            'classifier': PAE_ELM,
            'parameters': {
                'h_size': [600],
                'subnets': [3],
            },
        },
    }

    # Write result headers to output file
    with open(FILENAME, 'w') as f:
        f.write(RESULTS_HEADER + '\n')
        f.close()

    # Load dataset
    mnist = load_mnist()

    mnist_in = 400
    mnist_class = 10

    X = torch.from_numpy(mnist['train_X']).float()[:60000]
    y = torch.from_numpy(mnist['train_y']).float()[:60000]
    tX = torch.from_numpy(mnist['test_X']).float()


    for name, model in models.items():
        classifier = model['classifier']

        params_list, param_names = get_params(model['parameters'])

        def_params = {
            'c': 10,
            'in_size': mnist_in,
            'out_size': mnist_class
        }

        for params in params_list:
            in_params = combine_params(params, param_names, def_params)

            for seed in SEEDS:
                # Pre-trained weights
                model_full = LeNet5(classifier(**in_params), weight_train=False)

                model_full[0].load_state_dict(torch.load(f'weights/{seed}-1.pth'))
                model_full[3].load_state_dict(torch.load(f'weights/{seed}-2.pth'))

                run_experiment(model_full, name, seed, in_params, 'pre_train')

                # Random weights
                model_full = LeNet5(classifier(**in_params), weight_train=False)
                run_experiment(model_full, name, seed, in_params, 'ran')

                # Random orthographic weights
                model_full = LeNet5(classifier(**in_params), weight_train=True)
                run_experiment(model_full, name, seed, in_params, 'ortho_ran')