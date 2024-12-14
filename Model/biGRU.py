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

#BiGRU
class biGRU(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, output_size,
                 word_embeddings, dropout, device, seed):
        super(biGRU, self).__init__()
        self.device = device
        torch.manual_seed(seed)

        self.embedding_layer = nn.Embedding.from_pretrained(torch.from_numpy(word_embeddings).float(), freeze=True)
        self.dropout = nn.Dropout(dropout)

        self.input_dim = input_size
        self.hidden_dim = hidden_size
        self.output_dim = output_size
        self.concat_dim = input_size + hidden_size
        # self.sequence_length = sequence_length
        
        self.linear_candidate_forward = nn.Linear(self.concat_dim, self.hidden_dim)
        self.linear_relevance_forward = nn.Linear(self.concat_dim, self.hidden_dim)
        self.linear_update_forward = nn.Linear(self.concat_dim, self.hidden_dim)

        self.linear_candidate_backward = nn.Linear(self.concat_dim, self.hidden_dim)
        self.linear_relevance_backward = nn.Linear(self.concat_dim, self.hidden_dim)
        self.linear_update_backward = nn.Linear(self.concat_dim, self.hidden_dim)
        
        self.linear_prediction = nn.Linear(self.hidden_dim*2, self.output_dim)

    def gru_cell(self, xt, c_prev, linear_candidate, linear_relevance, linear_update):
        concat_gate = torch.cat((c_prev, xt), 1)
        rt = torch.sigmoid(linear_relevance(concat_gate))
        ut = torch.sigmoid(linear_update(concat_gate))
        
        concat_candidate = torch.cat((rt*c_prev, xt), 1)
        cct= torch.tanh(linear_candidate(concat_candidate))
        c_next = ut*cct + (1-ut)*c_prev

        return c_next 


    def forward(self, x):
        x = x.to(self.device)
        x = self.embedding_layer(x)
        batch_size = x.shape[0]
        sequence_length = x.shape[1]
        
        c_next_forward = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        c_forward = torch.zeros(batch_size, sequence_length, self.hidden_dim).to(self.device)
        c_next_backward = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        c_backward = torch.zeros(batch_size, sequence_length, self.hidden_dim).to(self.device)
        
        y_pred = torch.zeros(batch_size, sequence_length, self.output_dim).to(self.device)

        for t in range(sequence_length):
            c_next_forward = self.gru_cell(x[:,t,:], c_next_forward, self.linear_candidate_forward, self.linear_relevance_forward, self.linear_update_forward)
            c_forward[:,t,:] = c_next_forward

        for t in reversed(range(sequence_length)):
            c_next_backward = self.gru_cell(x[:,t,:], c_next_backward, self.linear_candidate_backward, self.linear_relevance_backward, self.linear_update_backward)
            c_backward[:,t,:] = c_next_backward

        a = torch.cat((c_forward, c_backward), 2)
        a = self.dropout(a)
        for t in range(sequence_length):
            y_pred[:, t, :] = self.linear_prediction(a[:, t, :])        
            
            
        return y_pred