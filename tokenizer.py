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

MAX_SEQUENCE_LENGTH = 30
TRAIN_URL = "https://drive.google.com/file/d/1ND_nNV5Lh2_rf3xcHbvwkKLbY2DmOH09/view?usp=sharing"

PAD_TOKEN_IDX = 0
SOS_TOKEN_IDX = 1
EOS_TOKEN_IDX = 2

class DerivativeTokenizer:
    def __init__(self) -> None:
        self.token2idx, self.idx2token = self.build_dict()

    def tokenize(self, token: str) -> List[int]:
        return self.token2idx[token]
    
    def tokenize_seq(self, sequence):
        p = 0
        l = len(sequence)
        tokenized_sequence = []
        while p < l:
          head = p + 1
          if sequence[p].isalpha():
              while head < l:
                # sequece[head] is a letter
                # print(head, sequence[head], ord(sequence[head])-ord("a"))
                if head < l and sequence[head].isalpha():
                    head += 1
                else:
                    break
          
          curr = sequence[p: head]
          if curr not in self.token2idx:
            for ch in curr:
              # print(f"{ch} -> {self.token2idx[ch]}")
              tokenized_sequence.append(self.token2idx[ch])
          else:
            # print(f"{curr} -> {self.token2idx[curr]}")
            tokenized_sequence.append(self.token2idx[curr])
          p = head
        
        return tokenized_sequence
              
    def detokenize(self, idx):
        return self.idx2token[idx]
        
    def vocab_size(self):
        return len(self.token2idx)

    def build_dict(self):
      token_dict = {"PADDING": PAD_TOKEN_IDX, "SOS": SOS_TOKEN_IDX, "EOS": EOS_TOKEN_IDX}
      for i in range(10):
          token_dict[str(i)] = i + 3

      token_dict.update({
          "+": 13,
          "-": 14, 
          "*": 15,
          "/": 16,
          "(": 17,
          ")": 18,
          "^": 19,
      })

      token_dict.update({
          "exp": 20,
          "sin": 21,
          "cos": 22,
      })

      for i in range(26):
          ch = chr(ord("a") + i)
          token_dict[ch] = i + 23
      
      idx_to_token = dict()
      for k, v in token_dict.items():
        idx_to_token[v] = k

      return token_dict, idx_to_token
