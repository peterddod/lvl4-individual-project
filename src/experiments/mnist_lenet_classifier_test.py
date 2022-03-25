from itertools import product
from sklearn.metrics import accuracy_score
import torch
from nitools.classifiers import PAE_ELM
from nitools.models import LeNetPlus
from nitools.operations import resetseed
from nitools.utils import load_mnist, load_fashionmnist, load_norb
import time as t

FILENAME = 'results.csv'

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

def run_experiment(model, name, seed, params, data, dataset, n=10000, nt=10000):
    print('\n===============================')
    print(f'{name} - {seed} - {str_params(params)}')

    X = data['train_X'].float()[:n]
    y = data['train_y'].float()[:n]
    tX = data['test_X'].float()[:nt]

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
    
    pred_arg = torch.zeros(len(tX))

    for i in range(len(pred)):
        pred_arg[i] = torch.argmax(pred[i])

    # get accuracy
    acc =  accuracy_score(pred_arg, data['test_y'][:nt])

    # create string of tuple ('name,subnets,h_size,seed,acc,train_time,test_time')
    results = f'{name},{dataset},{params["subnets"]},{params["h_size"]},{seed},{acc},{train_end-train_start},{pred_end-pred_start}'

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
    RESULTS_HEADER = 'name,dataset,subnets,h_size,seed,acc,train_time,test_time'

    # Initialise modles
    models = {
        'lenet-pae-elm': {
            'parameters': {
                'h_size': [100,300,900],
                'subnets': [1,2,3,5,7,9],
            },
        },
    }

    # Write result headers to output file
    with open(FILENAME, 'w') as f:
        f.write(RESULTS_HEADER + '\n')
        f.close()

    for i, loader in enumerate([load_mnist, load_fashionmnist, load_norb]):
        if loader in [load_mnist, load_norb]:
            continue

        data = loader()

        for name, model in models.items():
            params_list, param_names = get_params(model['parameters'])

            def_params = {
                'c': 10,
            }

            for params in params_list:
                in_params = combine_params(params, param_names, def_params)

                for seed in SEEDS:
                    model_full = LeNetPlus(**in_params, device=torch.device('cpu'))

                    run_experiment(model_full, name, seed, in_params, data, i, n=25000, nt=25000)


        