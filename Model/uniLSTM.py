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

class uniLSTM(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, output_size,
                 word_embeddings, dropout, device, seed):
        super(uniLSTM, self).__init__()
        self.device = device
        torch.manual_seed(seed)

        self.embedding_layer = nn.Embedding.from_pretrained(torch.from_numpy(word_embeddings).float(), freeze=True)
        self.dropout = nn.Dropout(dropout)

        self.input_dim = input_size
        self.hidden_dim = hidden_size
        self.output_dim = output_size
        self.concat_dim = input_size + hidden_size
        
        self.linear_candidate = nn.Linear(self.concat_dim, self.hidden_dim)
        self.linear_forget = nn.Linear(self.concat_dim, self.hidden_dim)
        self.linear_update = nn.Linear(self.concat_dim, self.hidden_dim)
        self.linear_output = nn.Linear(self.concat_dim, self.hidden_dim)
        self.linear_prediction = nn.Linear(self.hidden_dim, self.output_dim)


    def lstm_cell_forward(self, xt, a_prev, c_prev):
        concat = torch.cat((a_prev, xt), 1)
        ft = torch.sigmoid(self.linear_forget(concat))
        it = torch.sigmoid(self.linear_update(concat))
        ot = torch.sigmoid(self.linear_output(concat))
        cct = torch.tanh(self.linear_candidate(concat))
        
        c_next = ft * c_prev + it * cct
        a_next = ot * torch.tanh(c_next)
        
        # Apply dropout to the activation output
        a_next = self.dropout(a_next)
        
        yt_pred = self.linear_prediction(a_next)

        return a_next, c_next, yt_pred

    def forward(self, x):
        x = self.embedding_layer(x)
        sequence_length = x.shape[1]
        batch_size = x.shape[0]
        x = x.to(self.device)
        a_next = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        c_next = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        a = torch.zeros(batch_size, sequence_length, self.hidden_dim).to(self.device)
        c = torch.zeros(batch_size, sequence_length, self.hidden_dim).to(self.device)
        y_pred = torch.zeros(batch_size, sequence_length, self.output_dim).to(self.device)

        for t in range(sequence_length):
            a_next, c_next, yt = self.lstm_cell_forward(x[:, t, :], a_next, c_next)
            a[:, t, :] = a_next
            c[:, t, :] = c_next
            y_pred[:, t, :] = yt
            
        return y_pred
