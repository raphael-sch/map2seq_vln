import torch
from torch import nn

from utils import padding_idx


class BiLSTM(nn.Module):
    def __init__(self, opts, embedding, num_layers=2, hidden_size=256, dropout=0.3, input_size=32):
        super(BiLSTM, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.opts = opts
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0,
                            bidirectional=True)
        self.dropout_out = nn.Dropout(dropout)
        if self.opts.config.use_layer_norm:
            self.layer_norm_out = nn.LayerNorm(hidden_size * 2, eps=1e-6)

    def _forward(self, x, lengths, cat_features=None):
        x_embed = self.embedding(x)  # [batch_size, max_length, 32]

        if cat_features is not None:
            x_embed = torch.cat([x_embed] + cat_features, dim=-1)

        x_packed = nn.utils.rnn.pack_padded_sequence(input=x_embed,
                                                     lengths=lengths,
                                                     batch_first=True,
                                                     enforce_sorted=False)
        output, (h_n, c_n) = self.lstm(x_packed)

        # unpack output
        seq_unpacked, lens_unpacked = nn.utils.rnn.pad_packed_sequence(output, batch_first=False)
        if self.opts.config.use_layer_norm:
            seq_unpacked = self.layer_norm_out(seq_unpacked)
        seq_unpacked = self.dropout_out(seq_unpacked)
        # seq_unpacked (max_len, batch_size, hidden*2)

        return (seq_unpacked, lens_unpacked), (h_n, c_n)


class InstructionEncoder(BiLSTM):
    def __init__(self, opts, vocab_size, embedding_dim=32, hidden_size=256, num_layers=2, dropout=0.3):
        embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx)

        super(InstructionEncoder, self).__init__(opts,
                                                 embedding,
                                                 num_layers=num_layers,
                                                 hidden_size=hidden_size,
                                                 dropout=dropout,
                                                 input_size=embedding_dim)

        self.bridge_ht = nn.Linear(in_features=hidden_size*2, out_features=hidden_size)
        self.bridge_ct = nn.Linear(in_features=hidden_size*2, out_features=hidden_size)
        self.dropout_first_ht = nn.Dropout(dropout)
        self.dropout_first_ct = nn.Dropout(dropout)
        if self.opts.config.use_layer_norm:
            self.layer_norm_first_ht = nn.LayerNorm(hidden_size, eps=1e-6)
            self.layer_norm_first_ct = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, x, lengths):
        (seq_unpacked, lens_unpacked), (_, c_n) = self._forward(x, lengths)
        last_c = torch.cat([c_n[-2], c_n[-1]], dim=-1)

        # used to initialize steps rnn
        first_ht = torch.sigmoid(self.bridge_ht(last_c)).unsqueeze(0)  # [1, batch_size, 256]
        first_ct = torch.sigmoid(self.bridge_ct(last_c)).unsqueeze(0)  # [1, batch_size, 256]

        if self.opts.config.use_layer_norm:
            first_ht = self.layer_norm_first_ht(first_ht)
            first_ct = self.layer_norm_first_ct(first_ct)

        first_ht = self.dropout_first_ht(first_ht)
        first_ct = self.dropout_first_ct(first_ct)

        return (seq_unpacked, lens_unpacked), (first_ht, first_ct)
