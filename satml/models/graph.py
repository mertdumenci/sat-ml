
import torch
import torch.nn as nn

from satml.models.net import Net


class GraphEmbeddingLSTM(nn.Module):
    """An implementation of a modified version of the NeuroSAT formula classifier (Selsam et al)."""
    def __init__(self, dimension, iterations):
        super(GraphEmbeddingLSTM, self).__init__()

        self.dimension = dimension
        self.iterations = iterations

        # https://github.com/ryanzhangfan/NeuroSAT/blob/master/src/neurosat.py
        #
        # We learn the initialization vectors, hence their being linear transformations
        # (to make Pytorch happy to learn these vectors)
        self.l_init = nn.Linear(1, dimension)
        self.c_init = nn.Linear(1, dimension)

        # The message nets
        self.l_msg = Net(dimension, [dimension, dimension, dimension])
        self.c_msg = Net(dimension, [dimension, dimension, dimension])

        # The update LSTM
        self.l_u = nn.LSTM(dimension * 2, dimension)
        self.c_u = nn.LSTM(dimension, dimension)

        # The classifier
        self.l_cls = Net(dimension, [dimension, dimension, 1])

    def flip(self, embedding_matrix):
        halfway = int(embedding_matrix.shape[0] / 2)
        return torch.cat((embedding_matrix[halfway:], embedding_matrix[:halfway]))

    def forward(self, X_tuple):
        adj_matrix, num_vars = X_tuple
        # Get our L_init and C_init vectors.
        x = torch.ones(1)
        if torch.cuda.is_available():
            x = x.cuda()

        x.requires_grad = False
        L_init = self.l_init(x)
        C_init = self.c_init(x)

        # Embedding matrices start out by tiling L_init, C_init.
        n_lits, n_clauses = adj_matrix.shape
        L_t = L_init.repeat(n_lits, 1)
        C_t = C_init.repeat(n_clauses, 1)

        # Hidden states for the update LSTMs
        L_h = torch.zeros((1, n_lits, self.dimension))
        L_0 = torch.zeros((1, n_lits, self.dimension))
        C_h = torch.zeros((1, n_clauses, self.dimension))
        C_0 = torch.zeros((1, n_clauses, self.dimension))

        if torch.cuda.is_available():
            L_h, L_0, C_h, C_0 = L_h.cuda(), L_0.cuda(), C_h.cuda(), C_0.cuda()
        
        for i in range(self.iterations):
            l_agg = torch.matmul(adj_matrix.t(), self.l_msg(L_t))
            c_agg = torch.matmul(adj_matrix, self.c_msg(C_t))
            
            # Clause update uses literal embedding aggregations
            _, (C_h, _) = self.c_u(l_agg.unsqueeze(0), (C_h, C_0))
            # Literal update uses clause embedding aggregations
            l_input = torch.cat((c_agg, self.flip(L_t)), dim=1)
            _, (L_h, _) = self.l_u(l_input.unsqueeze(0), (L_h, L_0))

            # Unpack the hidden states to go back in the loop
            C_t = C_h.squeeze(0)
            L_t = L_h.squeeze(0)

        # Classify the literal embeddings
        L_imp = self.l_cls(L_t)
        L_imp = L_imp.squeeze(1)
        
        # Unpack the weights into (de-normalized) probability vectors
        L_imp_unpacked = []
        
        total_vars = sum(num_vars)
        max_vars = max(num_vars)

        currently_seen_vars = 0
        for i, n in enumerate(num_vars):
            lits = torch.zeros(max_vars * 2)
            if torch.cuda.is_available():
                lits = lits.cuda()

            lits[:n] = L_imp[currently_seen_vars:currently_seen_vars + n]
            lits[n:2 * n] = L_imp[total_vars + currently_seen_vars:total_vars + currently_seen_vars + n]

            L_imp_unpacked.append(lits)
            currently_seen_vars += n

        return torch.stack(L_imp_unpacked)
