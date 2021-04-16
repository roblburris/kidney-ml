import torch, math
from torch import nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from kidney_dataset import KidneyDataset

dataset = KidneyDataset('../data/mouse-adult/')

train_set, test_set = random_split(dataset, [math.floor(len(dataset) * 0.3), len(dataset) - math.floor(len(dataset) * 0.3)])

# not sure about batch sizes, need to figure this out more
train_dl = DataLoader(train_set, batch_size=64)
test_dl = DataLoader(test_set, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(19125, 3),
            nn.Softmax()
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)

model = NeuralNetwork()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

for i in range(4):
    print(f"Epoch {i+1}\n-------------------------------")
    train(train_dl, model, loss_fn, optimizer)
    test(test_dl, model)

    
