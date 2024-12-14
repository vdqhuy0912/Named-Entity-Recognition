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

class biLSTM(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, output_size,
                 word_embeddings, dropout, device, seed):
        super(biLSTM, self).__init__()
        self.device = device
        torch.manual_seed(seed)

        self.embedding_layer = nn.Embedding.from_pretrained(torch.from_numpy(word_embeddings).float(), freeze=True)
        self.dropout = nn.Dropout(dropout)

        self.input_dim = input_size
        self.hidden_dim = hidden_size
        self.output_dim = output_size
        self.concat_dim = input_size + hidden_size
        
        self.linear_candidate_forward = nn.Linear(self.concat_dim, self.hidden_dim)
        self.linear_forget_forward = nn.Linear(self.concat_dim, self.hidden_dim)
        self.linear_update_forward = nn.Linear(self.concat_dim, self.hidden_dim)
        self.linear_output_forward = nn.Linear(self.concat_dim, self.hidden_dim)

        self.linear_candidate_backward = nn.Linear(self.concat_dim, self.hidden_dim)
        self.linear_forget_backward = nn.Linear(self.concat_dim, self.hidden_dim)
        self.linear_update_backward = nn.Linear(self.concat_dim, self.hidden_dim)
        self.linear_output_backward = nn.Linear(self.concat_dim, self.hidden_dim)
        
        self.linear_prediction = nn.Linear(self.hidden_dim*2, self.output_dim)

    def lstm_cell(self, xt, a_prev, c_prev, linear_candidate, linear_forget, linear_update, linear_output):
        concat = torch.cat((a_prev, xt), 1)
        
        ft = torch.sigmoid(linear_forget(concat))
        it = torch.sigmoid(linear_update(concat))
        ot = torch.sigmoid(linear_output(concat))
        cct= torch.tanh(linear_candidate(concat))
        
        c_next = ft*c_prev + it*cct
        a_next = ot*torch.tanh(c_next)
        
        return a_next, c_next

    def forward(self, x):
        x = x.to(self.device)
        x = self.embedding_layer(x)
        batch_size = x.shape[0]
        sequence_length = x.shape[1]
        
        a_next_forward = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        c_next_forward = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        a_forward = torch.zeros(batch_size, sequence_length, self.hidden_dim).to(self.device)


        a_next_backward = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        c_next_backward = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        a_backward = torch.zeros(batch_size, sequence_length, self.hidden_dim).to(self.device)
        
        y_pred = torch.zeros(batch_size, sequence_length, self.output_dim).to(self.device)

        
        for t in range(sequence_length):
            a_next_forward, c_next_forward = self.lstm_cell(x[:,t,:], a_next_forward, c_next_forward, self.linear_candidate_forward, self.linear_forget_forward, self.linear_update_forward, self.linear_output_forward)
            a_forward[:,t,:] = a_next_forward

        for t in reversed(range(sequence_length)):
            a_next_backward, c_next_backward = self.lstm_cell(x[:,t,:], a_next_backward, c_next_backward, self.linear_candidate_backward, self.linear_forget_backward, self.linear_update_backward, self.linear_output_backward)
            a_backward[:,t,:] = a_next_backward

        a = torch.cat((a_forward, a_backward), 2)
        a = self.dropout(a)
        for t in range(sequence_length):
            y_pred[:, t, :] = self.linear_prediction(a[:, t, :])        
            
            
        return y_pred