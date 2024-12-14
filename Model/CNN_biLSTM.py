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

class CNN_biLSTM(nn.Module):
    def __init__(self, vocab_size, word_emb_dim, word_embeddings, case_emb_dim, case_embeddings,
                 char_size, char_emb_dim, conv_out_channels, conv_kernel_size,
                 lstm_hidden_size, num_labels, dropout, dropout_recurrent, device):  
        super(CNN_biLSTM, self).__init__()
        self.device = device
        
        # Word Embedding
        self.word_embedding = nn.Embedding.from_pretrained(torch.from_numpy(word_embeddings).float(), freeze=True)

        # Casing Embedding
        self.case_embedding = nn.Embedding.from_pretrained(torch.from_numpy(case_embeddings).float(), freeze=True)

        # Character Embedding
        self.char_embedding = nn.Embedding(char_size, char_emb_dim)
        nn.init.uniform_(self.char_embedding.weight, a=-0.5, b=0.5)
        
        # CNN cho character
        self.char_conv = nn.Conv1d(in_channels=char_emb_dim,         # Kích thước của mỗi phần tử trong chuỗi
                                   out_channels=conv_out_channels,   # No.Filters
                                   kernel_size=conv_kernel_size, 
                                   padding='same')  
        self.char_activation = nn.Tanh()
        self.char_maxpool = nn.AdaptiveMaxPool1d(1)  # Giảm chiều xuống 1

        # BiLSTM
        self.bilstm = nn.LSTM(
            input_size=word_emb_dim + case_emb_dim + conv_out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=1,  
            batch_first=True,
            bidirectional=True,
            dropout=dropout_recurrent
        )

        # Fully Connected Layer
        self.fc = nn.Linear(2 * lstm_hidden_size, num_labels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, words, casing, chars):
        # Word Embedding
        words, casing, chars = words.to(self.device), casing.to(self.device), chars.to(self.device)
        word_emb = self.word_embedding(words)  # [batch_size, seq_len, word_emb_dim]
        
        # Casing Embedding
        case_emb = self.case_embedding(casing)  # [batch_size, seq_len, case_emb_dim]
        
        # Character Embedding
        char_emb = self.char_embedding(chars)  # [batch_size, seq_len, max_char_len, char_emb_dim]
        char_emb = self.dropout(char_emb)
        
        batch_size, seq_len, max_char_len, char_emb_dim = char_emb.size()
        char_emb = char_emb.view(-1, max_char_len, char_emb_dim)  # [batch_size*seq_len, max_char_len, char_emb_dim]
        char_emb = char_emb.permute(0, 2, 1)  # [batch_size*seq_len, char_emb_dim, max_char_len]
        char_conv = self.char_conv(char_emb)  # [batch_size*seq_len, conv_out_channels, max_char_len]
        
        char_conv = self.char_activation(char_conv)
        
        char_conv = self.char_maxpool(char_conv).squeeze(-1)  # [batch_size*seq_len, conv_out_channels]
        char_conv = self.dropout(char_conv)
        
        char_conv = char_conv.view(batch_size, seq_len, -1)  # [batch_size, seq_len, conv_out_channels]
        
        # Concatenate embeddings
        embeddings = torch.cat([word_emb, case_emb, char_conv], dim=-1)  # [batch_size, seq_len, word_emb_dim + case_emb_dim + conv_out_channels]
        embeddings = self.dropout(embeddings)
        
        # BiLSTM
        lstm_out, _ = self.bilstm(embeddings)  # [batch_size, seq_len, 2*lstm_hidden_size]
        
        # Fully Connected
        fc_out = self.fc(lstm_out)  # [batch_size, seq_len, num_labels]
        
        return fc_out
