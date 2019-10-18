
from satml import fast_cnf

import torch
import torch.nn as nn
import torch.utils.data as data


def one_hot(index, length):
    v = torch.zeros(length)
    v[index] = 1
    return v


def tokenize(formula, max_var):
    token_length = max_var + 2 # For AND and OR

    tokenized = []
    for i, clause in enumerate(formula):
        # Intersperse conjunct tokens
        if i > 0:
            tokenized.append(one_hot(-1, token_length))
        
        for j, literal in enumerate(clause):
            # Intersperse disjunct tokens
            if j > 0:
                tokenized.append(one_hot(-2, token_length))
            
            tokenized.append(one_hot(literal - 1, token_length))

    return torch.stack(tokenized)


def collator(batch):
    """Expects `batch` to be a list of (X, y) pairs."""
    X, y = zip(*batch)
    padded_sequences = nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value=-1)
    return padded_sequences, torch.LongTensor(y)


class LSTMDataset(data.Dataset):
    def __init__(self, decisions, max_var=None):
        self.decisions = decisions
        
        # Extract the maximum number of variables in the dataset, we use this
        # for labeling.
        if max_var is None:
            self.max_var = max((fast_cnf.max_var(f) for f, _ in decisions))
        else:
            self.max_var = max_var
        
    def __len__(self):
        return len(self.decisions)
    
    def __getitem__(self, index):
        formula, (var, branch) = self.decisions[index]
        label = 0
        if branch is True:
            label = var - 1
        else:
            label = self.max_var + (var - 1)
    
        return tokenize(formula, self.max_var), label

    @property
    def max_label(self):
        return self.max_var * 2
