import torch
import torch.nn as nn


class CharLSTM(nn.Module):
    def __init__(self, vocab_size, emb_size=256, hidden_size=512, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(emb_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        emb = self.embedding(x)
        out, hidden = self.lstm(emb, hidden)
        logits = self.linear(out)
        return logits, hidden

    def init_hidden(self, batch_size, device):
        return (
            torch.zeros(
                self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=device
            ),
            torch.zeros(
                self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=device
            ),
        )
