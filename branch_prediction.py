
import os
import pickle
import random
import argparse

from satml import fast_cnf, models, datasets, training
import satml.models.graph
import satml.models.lstm
import satml.datasets.graph
import satml.datasets.lstm

import torch
import torch.utils.data as data


parser = argparse.ArgumentParser('Trains a branch predictor')

parser.add_argument('--data', type=str, help='the pickled data file generated by `data_generator.py`')
parser.add_argument('--model', type=str, help='type of model to use', choices=['lstm', 'graph'])
parser.add_argument('--epochs', type=int, help='how many epochs to run for', default=150)
parser.add_argument('--batch-size', type=int, help='self explanatory', default=64)
parser.add_argument('--checkpoint', type=str, help='the checkpoint directory', default='checkpoint')
parser.add_argument('--log-interval', type=int, help='when to print status', default=25)
parser.add_argument('--num-formulas', type=int, help='restrict the number of formulas used for training')
parser.add_argument('--train-split', type=float, help='the training split [0, 1]', default=0.8)

parser.add_argument('--graph-embedding-size', type=int, help='the graph embedding size', default=50)
parser.add_argument('--graph-iterations', type=int, help='the graph iterations', default=5)

parser.add_argument('--lstm-num-layers', type=int, help='num lstm layer', default=3)
parser.add_argument('--lstm-embedding-size', type=int, help='the lstm embedding size', default=50)
parser.add_argument('--lstm-linear-size', type=int, help='the lstm linear layer size', default=50)
parser.add_argument('--lstm-token-vars', type=int, help='the maximum number of variables supported')

args = parser.parse_args()


print("Loading decisions... Might take a while.")
with open(args.data, 'rb') as f:
    decisions = pickle.load(f)
assert decisions is not None

print(f"✅ Loaded {len(decisions)} decisions")
if args.num_formulas:
    decisions = decisions[:args.num_formulas]
    print(f"Restricted to {len(decisions)} decisions by --num-formulas")

num_training = int(len(decisions) * args.train_split)
decisions_train, decisions_val = decisions[:num_training], decisions[num_training:]
print(f"Have {len(decisions_train)} training examples, {len(decisions_val)} validation examples")

dataset_constr, dataset_args, model, collate_fn = None, None, None, None

if args.model == 'lstm':
    dataset_constr = datasets.lstm.LSTMDataset
    dataset_args = {'max_var': args.lstm_token_vars}

    dataset_train = dataset_constr(decisions_train, **dataset_args)

    sample, _ = dataset_train[0]
    model = models.lstm.LSTM(
        sample.shape[1],
        args.lstm_embedding_size,
        [args.lstm_linear_size, args.lstm_linear_size, dataset_train.max_label],
        args.lstm_num_layers
    )

    collate_fn = datasets.lstm.collator
elif args.model == 'graph':
    dataset_constr = datasets.graph.AdjacencyDataset
    dataset_train = dataset_constr(decisions_train)

    model = models.graph.GraphEmbeddingLSTM(args.graph_embedding_size, args.graph_iterations)

    collate_fn = datasets.graph.collator

dataset_val = dataset_constr(decisions_val, **dataset_args)

loader_opts = {
    'collate_fn': collate_fn,
    'batch_size': args.batch_size,
    'shuffle': True,
    'num_workers': 4
}

def val_loader_make(subset=0.1):
    if subset == 1:
        return data.DataLoader(dataset_val, **loader_opts)
    
    size = int(len(decisions_val) * subset)
    indices = random.sample(list(range(len(decisions_val))), k=size)

    sampled_dec = []
    for i in indices:
        sampled_dec.append(decisions_val[i])

    return data.DataLoader(dataset_constr(sampled_dec, **dataset_args), **loader_opts)


loader_train = data.DataLoader(dataset_train, **loader_opts)


print(model)
print('➡️  Starting training...')
training.train_model(model, loader_train, val_loader_make, args.epochs, args.log_interval, args.checkpoint)
