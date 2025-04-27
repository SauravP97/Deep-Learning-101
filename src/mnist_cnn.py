import torchvision.io as tio
from torch.utils.data import DataLoader 
from torchvision import datasets, transforms
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BATCH_SIZE = 64
EPOCHS = 10


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def load_dataset():
    # Define a transform to normalize the data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Load the training dataset
    train_dataset = datasets.MNIST(
        root='../dataset/data', train=True, download=False, transform=transform)

    # Load the testing dataset
    test_dataset = datasets.MNIST(
        root='../dataset/data', train=False, download=False, transform=transform)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    return train_loader, test_loader


def training():
    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    train_loader, test_loader = load_dataset()

    for epoch in range(EPOCHS):
        for id, image_data in enumerate(train_loader, 0):
            hand_image, target = image_data
            optimizer.zero_grad()
            logits = model(hand_image)

            loss = criterion(logits, target)

            loss.backward()
            optimizer.step()
        
        print(f'Loss at epoch {epoch}: {loss}')
    
    print("Test Loss: ", evaluate_test_loss(test_loader, model))


def evaluate_test_loss(test_loader, model):
    totals = 0
    correct = 0

    for i, image_row in enumerate(test_loader, 0):
        # zero the parameter gradients
        image_data, target = image_row
        logits = model(image_data)

        _, predicted = torch.max(logits, 1)

        if predicted[0] == target[0]:
            correct += 1
        totals += 1

    accuracy = (correct / totals) * 100.0
    return accuracy


training()
