import torch, math
from torch import nn
from torch.utils.data import DataLoader 
import matplotlib.pyplot as plt
import wandb

from kidney_dataset import KidneyDataset
from process_data import *
from model import NeuralNetwork
from train_cval_test import *


def feature_selection():
    # TODO: function to perform feature selection
    #       to find relevant genes


def main():
    # load dataset
    adata = load_data('./data/')
    train_adata, test_adata = train_test(adata, 0.8)

    dataset = KidneyDataset('./data/')

    train_set, test_set = random_split(dataset, [math.floor(len(dataset) * 0.8), len(dataset) - math.floor(len(dataset) * 0.8)])

    # not sure about batch sizes, need to figure this out more
    train_dl = DataLoader(train_set, batch_size=64)
    test_dl = DataLoader(test_set, batch_size=64)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    model = NeuralNetwork()
    model = model.to(device)
    wandb.watch(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    for i in range(4):
        print(f"Epoch {i+1}\n-------------------------------")
        train(train_dl, model, loss_fn, optimizer)

    test(test_dl, model)

if __name__ == '__main__':
    wandb.init(project='kidney-ml', entity='roblburris')
    config = wandb.config
    config.learning_rate = 0.01
    
    main()