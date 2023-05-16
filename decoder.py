import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, vocab_size):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.Wa = nn.Linear(hidden_dim * 2, hidden_dim)
        self.Wb = nn.Linear(hidden_dim, hidden_dim)
        self.Ws = nn.Linear(hidden_dim, hidden_dim)
        self.Wu = nn.Linear(input_dim, hidden_dim)
        self.Wout = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, ut_1, st_1, encoder_outputs, attention_mask):
        ut_1 = self.embedding(ut_1)
        st, (ht, ct) = self.lstm(ut_1, st_1)

        # Compute attention weights
        energy = torch.tanh(self.Wa(torch.cat((ht[-1], encoder_outputs), dim=1)))
        attention_scores = torch.sum(self.Wb(energy) * attention_mask, dim=1)
        alpha = F.softmax(attention_scores, dim=0)
        ct = torch.sum(encoder_outputs * alpha.unsqueeze(-1), dim=0)

        # Compute copy probabilities
        copy_probs = torch.sum(alpha.unsqueeze(-1) * (ut_1 == encoder_outputs), dim=0)

        # Compute generation probabilities
        p_gen = torch.sigmoid(self.Ws(ht[-1]) + self.Wu(ut_1[-1]) + self.Wb(ct))

        # Compute final probability distribution
        p_vocab = F.softmax(self.Wout(torch.cat((ht[-1], ct), dim=1)), dim=1)
        p_copy = torch.zeros_like(p_vocab)
        p_copy[:, :encoder_outputs.shape[0]] = copy_probs
        p_final = p_gen * p_vocab + (1 - p_gen) * p_copy

        return p_final, st
