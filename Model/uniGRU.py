import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import precision_recall_fscore_support
from collections import defaultdict
import random
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

class uniGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,word_embeddings, dropout, device, seed):
        super(uniGRU, self).__init__()
        self.device = device
        torch.manual_seed(seed)

        self.embedding_layer = nn.Embedding.from_pretrained(torch.from_numpy(word_embeddings).float(), freeze=True)
        self.dropout = nn.Dropout(dropout)

        self.input_dim = input_size
        self.hidden_dim = hidden_size
        self.output_dim = output_size
        self.concat_dim = input_size + hidden_size
        
        self.linear_candidate = nn.Linear(self.concat_dim, self.hidden_dim)
        self.linear_relevance = nn.Linear(self.concat_dim, self.hidden_dim)
        self.linear_update = nn.Linear(self.concat_dim, self.hidden_dim)
        self.linear_prediction = nn.Linear(self.hidden_dim, self.output_dim)

    def gru_cell(self, xt, c_prev):
        xt = self.dropout(xt) # dropout 
        concat_gate = torch.cat((c_prev, xt), 1)
        rt = torch.sigmoid(self.linear_relevance(concat_gate))
        ut = torch.sigmoid(self.linear_update(concat_gate))
        concat_candidate = torch.cat((rt*c_prev, xt), 1)
        cct= torch.tanh(self.linear_candidate(concat_candidate))
        cct = self.dropout(cct) # dropout
        c_next = ut*cct + (1-ut)*c_prev
        c_next = self.dropout(c_next) # add dropout 
        yt_pred = self.linear_prediction(c_next)

        return c_next, yt_pred

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.to(self.device)
        x = self.embedding_layer(x)
        sequence_length = x.shape[1]
        
        c_next = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        c = torch.zeros(batch_size, sequence_length, self.hidden_dim).to(self.device)
        y_pred = torch.zeros(batch_size, sequence_length, self.output_dim).to(self.device)

        for t in range(sequence_length):
            c_next, yt = self.gru_cell(x[:,t,:], c_next)
            c[:,t,:] = c_next
            y_pred[:,t,:] = yt
        return y_pred     
        # return y_pred, c
    print("done")