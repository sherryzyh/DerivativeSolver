from typing import Tuple, List
import numpy as np

import tqdm
import time

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# My Classes
from dataset import DerivativeDataset
from tokenizer import DerivativeTokenizer
from model import *
from config import *

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


def train_one_epoch(model, device, train_loader, loss_fn, optimizer, tokenizer):
    num_batches = len(train_loader)
    total_loss = 0.0
    model.train()
    losses = []
    for i, (encoder_input, decoder_input, label) in tqdm(enumerate(train_loader), total=num_batches):
        model.zero_grad()
        encoder_input, decoder_input, label = encoder_input.to(device), decoder_input.to(device), label.to(device)
        logits = model(encoder_input, decoder_input)
        logits = logits.reshape(-1, tokenizer.vocab_size())

        label = label.view(-1)
        loss = loss_fn(logits, label)
        total_loss += loss.item()

        losses.append(loss.item())

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / num_batches

    return avg_loss, losses

def evaluate(model, device, val_loader, tokenizer, loss_fn):
    num_batches = len(val_loader)
    correct = 0.0
    count = 0.0
    model.eval()
    with torch.no_grad():
      # val batchsize = 1
        for i, (encoder_input, true_derivative) in tqdm(enumerate(val_loader), total=num_batches):
            encoder_input = encoder_input.to(device)
            batchsize = encoder_input.size(0)
            pred_logits = model(encoder_input, None, "test") # list of pred seq_len <EOS>
            prediction = torch.argmax(pred_logits, dim=-1) # (batchsize)

            for j in range(batchsize):
                for i in range(MAX_SEQUENCE_LENGTH + 1):
                  if i < MAX_SEQUENCE_LENGTH and prediction[j][i].item() == EOS_TOKEN_IDX:
                    break
                pred_derivative = "".join([tokenizer.detokenize(idx) for idx in prediction[j][:i].cpu().tolist()])
                print(f"pred: {pred_derivative} | true: {true_derivative[j]}")
                correct += score(true_derivative[j], pred_derivative)
            count += batchsize
            # print(f"\nbatch {i}, correct {correct}, count {count}")
        
    
    acc = correct / count
    print(f"\nacc: {acc:.5f} | correct {correct}, count {count}")
    return acc


def train_val(model, device, train_loader, val_loader, n_epochs, loss_fn, tokenizer, optimizer, exp_record):
    model.to(device)
    starttime = time.time()
    for epoch in range(n_epochs):

        print("Training Epoch {} ... " .format(epoch))
        avg_loss, train_losses = train_one_epoch(model, device, train_loader, loss_fn, optimizer, tokenizer)
        exp_record["train_losses"].append(train_losses)
        traintime = time.time() - starttime
        print(f"\nTraining losses: {train_losses}")

        print("Testing Epoch {} ... " .format(epoch))
        val_acc = evaluate(model, device, val_loader, tokenizer, loss_fn)
        exp_record["val_acc"].append(val_acc)

        print("Epoch: {}, Avg train loss = {}, time = {:.2f} min, Val acc = {}, time = {:.2f} min"\
              .format(epoch, avg_loss, traintime/60, val_acc, (time.time()-starttime)/60))

def main():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    # experiment config
    loss_fn = nn.CrossEntropyLoss()

    tokenizer = DerivativeTokenizer()
    train_dataset = DerivativeDataset(traindata, trainlabel, tokenizer, isTrain=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                shuffle=True, num_workers=4, collate_fn=train_dataset.collate_fn_train)
    val_dataset = DerivativeDataset(valdata, vallabel, tokenizer, isTrain=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=4, collate_fn=val_dataset.collate_fn_val)

    model = Seq2Seq(tokenizer.vocab_size(), EMBEDDING_SIZE, ENCODER_HIDDEN_SIZE, DECODER_HIDDEN_SIZE)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    exp_record = {"train_losses": [], "val_acc": []}
    train_val(model, DEVICE, train_loader, val_loader, EPOCHES, loss_fn, tokenizer, optimizer, exp_record)

if __name__ == "__main__":
    main()