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
def predict(model, device, tokenizer, f: str):
    if len(f) >= MAX_SEQUENCE_LENGTH:
        tokenized_function = np.array([tokenizer.tokenize(ch) for ch in f] + [EOS_TOKEN_IDX])
    else:
        tokenized_function = np.array([tokenizer.tokenize(ch) for ch in f] + [EOS_TOKEN_IDX] + [PAD_TOKEN_IDX] * (MAX_SEQUENCE_LENGTH - len(f)))

    encoder_input = [torch.from_numpy(tokenized_function)]
    encoder_input = nn.utils.rnn.pad_sequence(encoder_input, batch_first=True, padding_value=PAD_TOKEN_IDX)
    encoder_input = encoder_input.to(device)
    # print(f"test encoder input: {encoder_input}")

    pred_logits = model(encoder_input, None, "test")
    prediction = torch.argmax(pred_logits, dim=-1)
    # print(f"test prediction: {prediction}")
    
    prediction = prediction.cpu().tolist()
    for i in range(MAX_SEQUENCE_LENGTH + 1):
        if i < MAX_SEQUENCE_LENGTH and prediction[0][i] == EOS_TOKEN_IDX:
            break
    pred_derivative = "".join([tokenizer.detokenize(idx) for idx in prediction[0][:i]])

    return pred_derivative


# ----------------- END ----------------- #


def main(filepath: str = "test.txt"):
    """load, inference, and evaluate"""
    functions, true_derivatives = load_file("train.txt")
    global DEVICE
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


    total_len = len(functions)
    indices = list(range(total_len))
    random.seed(52)
    random.shuffle(indices)
    # test_size = math.floor(total_len * 0.02)
    test_size = 5000
    test_idx = indices[: test_size]

    functions = [functions[i] for i in test_idx]
    true_derivatives = [true_derivatives[i] for i in test_idx]
    # print(set(functions))
    print(f"Total data sample: {len(functions)}")

    tokenizer = DerivativeTokenizer()
    model = Seq2Seq(tokenizer.vocab_size(), EMBEDDING_SIZE, ENCODER_HIDDEN_SIZE, DECODER_HIDDEN_SIZE)
    model.load_state_dict(torch.load("./checkpoints/best.pt"))
    model = model.to(DEVICE)
    model.eval()

    predicted_derivatives = [predict(model, DEVICE, tokenizer, f) for f in functions]
    # for td, pd in zip(true_derivatives, predicted_derivatives):
    #     if score(td, pd) < 1:
    #         print(f"true: {td} | pred: {pd}")
    scores = [score(td, pd) for td, pd in zip(true_derivatives, predicted_derivatives)]
    print(np.mean(scores))


if __name__ == "__main__":
    main()
