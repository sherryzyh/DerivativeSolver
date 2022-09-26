from typing import Tuple, List
import numpy as np
import torch
import math
import time
from io import open
import unicodedata
import string
import re
import random
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from config import *

class DerivativeEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size=128, hidden_size=256):
        super(DerivativeEncoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, emb_size, PAD_TOKEN_IDX)
        # self.gru = nn.LSTM(hidden_size, hidden_size)
        self.lstm = nn.LSTM(
            input_size=emb_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True)

    def forward(self, encoder_input):
        output = self.embedding(encoder_input)
        output, hidden = self.lstm(output)
        return output, hidden


class DerivativeDecoder(nn.Module):
    def __init__(self, vocab_size, emb_size=128, hidden_size=256, max_length=MAX_SEQUENCE_LENGTH):
        super(DerivativeDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.max_length = max_length

        self.embedding = nn.Embedding(self.vocab_size, self.emb_size)
        self.lstm = nn.LSTM(
            input_size=emb_size,
            hidden_size=hidden_size,
            num_layers=4,
            batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, decoder_input, encoder_outputs, encoder_hidden):
        output = self.embedding(decoder_input)
        output, last_hidden = self.lstm(output, encoder_hidden)
        output = self.linear(output)
        return output, last_hidden

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, emb_size=128, encoder_hidden_dim=256, decoder_hidden_dim=256):
        super(Seq2Seq,self).__init__()
        self.vocab_size = vocab_size
        self.encoder = DerivativeEncoder(vocab_size, emb_size, encoder_hidden_dim)
        self.decoder = DerivativeDecoder(vocab_size, emb_size, decoder_hidden_dim, max_length=MAX_SEQUENCE_LENGTH)
    
    def forward(self, encoder_input, gt_decoder_input=None, mode="train"):
        batchsize = encoder_input.size(0)
        if gt_decoder_input is not None:
          batch_max_seq_len = gt_decoder_input.size(1)
        else:
          batch_max_seq_len = MAX_SEQUENCE_LENGTH + 1

        # print(f"gt_decoder_input: {gt_decoder_input}")
        # print(f"mode: {mode}")

        encoder_output, encoder_hidden = self.encoder(encoder_input) # (batchsize, seq_len, hidden_size)
        decoder_input = torch.LongTensor([[SOS_TOKEN_IDX] for _ in range(batchsize)]).to(encoder_input.device)
        decoder_output = torch.empty((batchsize, 1, self.vocab_size), dtype=torch.int64).to(encoder_input.device)

        for i in range(batch_max_seq_len):
            pred_logits, _ = self.decoder(decoder_input, encoder_output, encoder_hidden) # (batchsize, seq_len(decoder_input_len), vocab_size)
            pred = torch.argmax(pred_logits[:, -1, :], dim=1) # (batchsize)

            decoder_output = torch.cat([decoder_output, pred_logits[:, -1, :].view(batchsize, 1, self.vocab_size)], axis=1)
            if i == batch_max_seq_len - 1:
              break

            if random.random() <= 0.2 and mode == "train": # use teacher forcing
              # print(f"decoder_input concat gt label")
              decoder_input = torch.cat([decoder_input, gt_decoder_input[:, i + 1].view(batchsize, 1)], axis=1)
            else: # in "test" or not use teacher forcing
              # print(f"decoder_input concat prev pred")
              decoder_input = torch.cat([decoder_input, pred.view(batchsize, 1)], axis=1)

        return decoder_output[:, 1:]
