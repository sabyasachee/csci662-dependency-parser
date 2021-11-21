# author - Sabyasachee

import sys
import pickle
import random
import numpy as np
import argparse
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt, use

from dependency_parser import DependencyParser

def train(device_id, batch_size, n_epochs, use_cube_activation):
    '''
    train the dependency parser model using the tensor files created from preparedata
    device_name is the name of the device where the training will be done
    '''
    print("loading the tensors and vocab")
    train = torch.load("train.pt")
    devel = torch.load("dev.pt")
    vocab = pickle.load(open("vocab.dict","rb"))

    train_dataset = TensorDataset(train)
    devel_dataset = TensorDataset(devel)

    word_vocab_size = len(vocab["word"])
    pos_tag_size = len(vocab["pos"])
    arc_tag_size = len(vocab["arc"])
    label_size = len(vocab["label"])

    device = torch.device(device_id) if device_id > -1 else torch.device("cpu")
    print(f"training on device {device}")

    print("seeding for reproducibility")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    print("creating the model")
    model = DependencyParser(word_vocab_size, pos_tag_size, arc_tag_size, label_size, use_cube_activation=use_cube_activation)
    model.to(device)

    print(f"batch_size = {batch_size}, n_epochs = {n_epochs}\n")

    optim = torch.optim.Adagrad(model.parameters(), lr=0.01, weight_decay=1e-8)
    loss_function = nn.CrossEntropyLoss()

    train_losses = []
    devel_losses = []

    print("training")
    for i in range(n_epochs):
        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        devel_loader = DataLoader(devel_dataset, batch_size=batch_size)
        
        n_batches = len(train_loader)
        if n_batches > 10:
            n_print_batches = n_batches//10
        else:
            n_print_batches = 1
        
        model.train()
        train_loss = 0
        print(f"\tepoch {i + 1}")
        
        for j, batch in enumerate(train_loader):
            optim.zero_grad()
            inp = batch[0].to(device)
            out = model(inp)
            
            true = batch[0][:,-1].to(device)
            loss = loss_function(out, true)
            
            loss.backward()
            optim.step()
            
            train_loss += loss.detach().cpu().item()
            if ((j + 1) % n_print_batches == 0) or (j == n_batches - 1):
                avg_train_loss = train_loss/(j + 1)
                print(f'\t\tbatch {j + 1:5d} train_loss = {avg_train_loss:.4f}')
                
        train_losses.append(train_loss/len(train_loader))
        
        model.eval()
        dev_loss = 0
        n_dev_batches = len(devel_loader)
        
        with torch.no_grad():
            for batch in devel_loader:
                inp = batch[0].to(device)
                out = model(inp)
                
                true = batch[0][:,-1].to(device)
                loss = loss_function(out, true)
                dev_loss += loss.detach().cpu().item()
                
        avg_dev_loss = dev_loss/n_dev_batches
        devel_losses.append(avg_dev_loss)
        print(f"\t\tdev loss = {avg_dev_loss:.4f}")

        if (i + 1) % 100 == 0:
            torch.save(model.state_dict(), f"train_cube-{use_cube_activation}_{batch_size}_{i + 1}.model")

    print()
    print("plotting training and dev loss")
    epochs = np.arange(n_epochs)
    plt.plot(epochs, train_losses, label="train")
    plt.plot(epochs, devel_losses, label="dev")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(f"loss_cube-{use_cube_activation}_{batch_size}_{n_epochs}.png")

    print("saving training and dev loss to text file")
    with open(f"loss_cube-{use_cube_activation}_{batch_size}_{n_epochs}.txt","w") as fw:
        fw.write("train_loss\tdev_loss\n")
        for train_loss, dev_loss in zip(train_losses, devel_losses):
            fw.write(f"{train_loss:.4f}\t{dev_loss:.4f}\n")

    print("saving model")
    torch.save(model.state_dict(), f"train_cube-{use_cube_activation}_{batch_size}_{n_epochs}.model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train parser")
    parser.add_argument("-b", type=int, default=1024, help="batch size")
    parser.add_argument("-e", type=int, default=1000, help="number of epochs")
    parser.add_argument("-c", action="store_true", help="set to use cube activation function")
    parser.add_argument("-d", type=int, default=-1, help="gpu device number, don't provide to use cpu")
    args = parser.parse_args()

    train(args.d, args.b, args.e, args.c)