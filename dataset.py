from typing import Tuple, List
import numpy as np
import torch
from io import open
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from config import *

class DerivativeDataset(Dataset):
    def __init__(self, input: List[str], target: List[str], tokenizer, isTrain) -> None:
        super().__init__()
        self.input = np.array(input)
        self.target = np.array(target)
        self.tokenizer = tokenizer
        self.isTrain = isTrain
    
    def __len__(self) -> int:
        return len(self.input)

    def __getitem__(self, index):
        function = self.input[index]
        tokenized_function = np.array([self.tokenizer.tokenize(ch) for ch in function] + [EOS_TOKEN_IDX])
        derivative = self.target[index]
        
        if self.isTrain:
          tokenized_decoder_input = np.array([SOS_TOKEN_IDX] + self.tokenizer.tokenize_seq(derivative))
          tokenized_target = np.array(self.tokenizer.tokenize_seq(derivative) + [EOS_TOKEN_IDX])
          return (tokenized_function, tokenized_decoder_input, tokenized_target)
        else:
          return (tokenized_function, derivative)
      
    def collate_fn_train(self, batch: List[Tuple]):
        batch_input = [torch.from_numpy(input) for input, decoder_input, label in batch]
        batch_decoder_input = [torch.from_numpy(decoder_input) for input, decoder_input, label in batch]
        batch_label = [torch.from_numpy(label) for input, decoder_input, label in batch]
        batch_input = nn.utils.rnn.pad_sequence(batch_input, batch_first=True, padding_value=PAD_TOKEN_IDX)
        batch_decoder_input = nn.utils.rnn.pad_sequence(batch_decoder_input, batch_first=True, padding_value=PAD_TOKEN_IDX)
        batch_label = nn.utils.rnn.pad_sequence(batch_label, batch_first=True, padding_value=-100)
        return batch_input, batch_decoder_input, batch_label
    
    def collate_fn_val(self, batch):
        batch_input = [torch.from_numpy(input) for input, true_derivatives in batch]
        batch_true_derivatives = [true_derivatives for input, true_derivatives in batch]
        batch_input = nn.utils.rnn.pad_sequence(batch_input, batch_first=True, padding_value=PAD_TOKEN_IDX)
        return batch_input, batch_true_derivatives