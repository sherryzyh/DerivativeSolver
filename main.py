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

def load_file(file_path: str) -> Tuple[Tuple[str], Tuple[str]]:
    """loads the test file and extracts all functions/derivatives"""
    data = open(file_path, "r").readlines()
    functions, derivatives = zip(*[line.strip().split("=") for line in data])
    return functions, derivatives


def score(true_derivative: str, predicted_derivative: str) -> int:
    """binary scoring function for model evaluation"""
    return int(true_derivative == predicted_derivative)

functions, true_derivatives = load_file("train.txt")

total_len = len(functions)
indices = list(range(total_len))
random.seed(42)
random.shuffle(indices)
train_size = math.floor(total_len * 0.8)
idx_train = indices[: train_size]
idx_val = indices[train_size:]

traindata = [functions[i] for i in idx_train]
trainlabel = [true_derivatives[i] for i in idx_train]
valdata = [functions[i] for i in idx_val]
vallabel = [true_derivatives[i] for i in idx_val]


def load_file(file_path: str) -> Tuple[Tuple[str], Tuple[str]]:
    """loads the test file and extracts all functions/derivatives"""
    data = open(file_path, "r").readlines()
    functions, derivatives = zip(*[line.strip().split("=") for line in data])
    return functions, derivatives


def score(true_derivative: str, predicted_derivative: str) -> int:
    """binary scoring function for model evaluation"""
    return int(true_derivative == predicted_derivative)


# --------- PLEASE FILL THIS IN --------- #
def predict(functions: str):
    return functions


# ----------------- END ----------------- #


def main(filepath: str = "test.txt"):
    """load, inference, and evaluate"""
    functions, true_derivatives = load_file("train.txt")
    print(set(functions))

    # predicted_derivatives = [predict(f) for f in functions]
    # scores = [score(td, pd) for td, pd in zip(true_derivatives, predicted_derivatives)]
    # print(np.mean(scores))


if __name__ == "__main__":
    main()
