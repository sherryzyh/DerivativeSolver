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
from main import predict

def load_file(file_path: str) -> Tuple[Tuple[str], Tuple[str]]:
    """loads the test file and extracts all functions/derivatives"""
    data = open(file_path, "r").readlines()
    functions, derivatives = zip(*[line.strip().split("=") for line in data])
    return functions, derivatives


def score(true_derivative: str, predicted_derivative: str) -> int:
    """binary scoring function for model evaluation"""
    return int(true_derivative == predicted_derivative)


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


def evaluate(model, device, val_loader, tokenizer):
    num_batches = len(val_loader)
    correct_eval = 0.0
    count = 0.0
    model.eval()
    with torch.no_grad():
        for _, (encoder_input, functions, true_derivative) in tqdm(enumerate(val_loader), total=num_batches):
            encoder_input = encoder_input.to(device)
            batchsize = encoder_input.size(0)
            pred_logits = model(encoder_input, None, "test") # list of pred seq_len <EOS>
            prediction = torch.argmax(pred_logits, dim=-1) # (batchsize)
            for j in range(batchsize):
                for i in range(MAX_SEQUENCE_LENGTH + 1):
                  if i < MAX_SEQUENCE_LENGTH and prediction[j][i].item() == EOS_TOKEN_IDX:
                    break
                eval_derivative = "".join([tokenizer.detokenize(idx) for idx in prediction[j][:i].cpu().tolist()])
                
                correct_eval += score(true_derivative[j], eval_derivative)
            count += batchsize
        
    
    acc_eval = correct_eval / count
    print(f"\neval acc: {acc_eval:.5f} | correct {correct_eval}, count {count}")
    return acc_eval


def train_val(model, device, train_loader, val_loader, n_epochs, loss_fn, tokenizer, optimizer, exp_record):
    model.to(device)
    starttime = time.time()
    best_acc = 0.0
    for epoch in range(n_epochs):

        print("Training Epoch {} ... " .format(epoch))
        avg_loss, train_losses = train_one_epoch(model, device, train_loader, loss_fn, optimizer, tokenizer)
        exp_record["train_losses"].append(train_losses)
        traintime = time.time() - starttime
        # print(f"\nTraining losses: {train_losses}")

        print("Testing Epoch {} ... " .format(epoch))
        val_acc = evaluate(model, device, val_loader, tokenizer)
        exp_record["val_acc"].append(val_acc)

        print("Epoch: {}, Avg train loss = {}, time = {:.2f} min, Val acc = {}, time = {:.2f} min"\
              .format(epoch, avg_loss, traintime/60, val_acc, (time.time()-starttime)/60))

        torch.save(model.state_dict(), f"./checkpoints/model_epoch_{epoch}.pt")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"./checkpoints/best.pt")

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    functions, true_derivatives = load_file("train.txt")
    print(f"MAX INPUT: {max(map(len, functions))}")

    total_len = len(functions)
    indices = list(range(total_len))
    random.seed(42)
    random.shuffle(indices)

    # Training config
    tokenizer = DerivativeTokenizer()
    loss_fn = nn.CrossEntropyLoss()
    model = Seq2Seq(tokenizer.vocab_size(), EMBEDDING_SIZE, ENCODER_HIDDEN_SIZE, DECODER_HIDDEN_SIZE)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    # Train Val split
    train_size = math.floor(total_len * 0.8)
    idx_train = indices[: train_size]
    idx_val = indices[train_size:]

    traindata = [functions[i] for i in idx_train]
    trainlabel = [true_derivatives[i] for i in idx_train]
    valdata = [functions[i] for i in idx_val]
    vallabel = [true_derivatives[i] for i in idx_val]

    train_dataset = DerivativeDataset(traindata, trainlabel, tokenizer, isTrain=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                shuffle=True, num_workers=4, collate_fn=train_dataset.collate_fn_train)
    val_dataset = DerivativeDataset(valdata, vallabel, tokenizer, isTrain=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=4, collate_fn=val_dataset.collate_fn_val)


    # Start training
    exp_record = {"train_losses": [], "val_acc": []}

    train_val(model, DEVICE, train_loader, val_loader, EPOCHES, loss_fn, tokenizer, optimizer, exp_record)
    print(f"EXP RECORD: {exp_record['val_acc']}")

    # model.load_state_dict(torch.load("best.pt"))
    # model = model.to(DEVICE)
    # wholeset = DerivativeDataset(functions, true_derivatives, tokenizer, isTrain=False)
    # dataloader = DataLoader(wholeset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=wholeset.collate_fn_val)
    # test_acc = evaluate(model, DEVICE, dataloader, tokenizer)



if __name__ == "__main__":
    main()