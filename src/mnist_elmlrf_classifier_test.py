from itertools import product
from sklearn.metrics import accuracy_score
import torch
from nitools.models import LRF_ELM
from nitools.operations import resetseed
from nitools.utils import load_mnist, load_fashionmnist, load_norb
import time as t


FILENAME = 'results.csv'

def run_experiment(model, name, seed, data, dataset, n=10000, nt=10000):
    print('\n===============================')
    print(f'{name} - {seed}')

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

    # create string of tuple ('name,dataset,seed,acc,train_time,test_time')
    results = f'{name},{dataset},{seed},{acc},{train_end-train_start},{pred_end-pred_start}'

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
    RESULTS_HEADER = 'name,dataset,seed,acc,train_time,test_time'

    # Write result headers to output file
    with open(FILENAME, 'w') as f:
        f.write(RESULTS_HEADER + '\n')
        f.close()

    for i, loader in enumerate([load_mnist, load_fashionmnist, load_norb]):
        data = loader()

        for seed in SEEDS:
            model_full = LRF_ELM(device=torch.device('cpu'))
            run_experiment(model_full, 'ELM-LRF', seed, data, i, n=25000, nt=25000)


        