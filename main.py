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

# My Classes
from model import *
from dataset import *
from tokenizer import *
from config import *


def load_file(file_path: str) -> Tuple[Tuple[str], Tuple[str]]:
    """loads the test file and extracts all functions/derivatives"""
    data = open(file_path, "r").readlines()
    functions, derivatives = zip(*[line.strip().split("=") for line in data])
    return functions, derivatives


def score(true_derivative: str, predicted_derivative: str) -> int:
    """binary scoring function for model evaluation"""
    return int(true_derivative == predicted_derivative)



def load_file(file_path: str) -> Tuple[Tuple[str], Tuple[str]]:
    """loads the test file and extracts all functions/derivatives"""
    data = open(file_path, "r").readlines()
    functions, derivatives = zip(*[line.strip().split("=") for line in data])
    return functions, derivatives


def score(true_derivative: str, predicted_derivative: str) -> int:
    """binary scoring function for model evaluation"""
    return int(true_derivative == predicted_derivative)


# --------- PLEASE FILL THIS IN --------- #
def predict(model, tokenizer, functions: str):
    tokenized_function = np.array([tokenizer.tokenize(ch) for ch in functions] + [EOS_TOKEN_IDX])
    encoder_input = [torch.from_numpy(tokenized_function)]
    encoder_input = nn.utils.rnn.pad_sequence(encoder_input, batch_first=True, padding_value=PAD_TOKEN_IDX)
    encoder_input = encoder_input.to(DEVICE)

    pred_logits = model(encoder_input, None, "test")
    prediction = torch.argmax(pred_logits, dim=-1)

    for i in range(MAX_SEQUENCE_LENGTH + 1):
        if i < MAX_SEQUENCE_LENGTH and prediction[0][i].item() == EOS_TOKEN_IDX:
            break
    pred_derivative = "".join([tokenizer.detokenize(idx) for idx in prediction[0][:i].cpu().tolist()])
    
    return pred_derivative


# ----------------- END ----------------- #


def main(filepath: str = "test.txt"):
    """load, inference, and evaluate"""
    functions, true_derivatives = load_file("train.txt")
    global DEVICE
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    functions = functions[:10000]
    true_derivatives = true_derivatives[:10000]
    # print(set(functions))

    tokenizer = DerivativeTokenizer()
    model = Seq2Seq(tokenizer.vocab_size(), EMBEDDING_SIZE, ENCODER_HIDDEN_SIZE, DECODER_HIDDEN_SIZE)
    model.load_state_dict(torch.load("best_derivative.pt"))
    model = model.to(DEVICE)
    model.eval()

    predicted_derivatives = [predict(model, tokenizer, f) for f in functions]
    scores = [score(td, pd) for td, pd in zip(true_derivatives, predicted_derivatives)]
    print(np.mean(scores))


if __name__ == "__main__":
    main()
