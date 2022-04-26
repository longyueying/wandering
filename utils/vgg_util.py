from datasets.cifar import Cifar10Dataset
from models.vgg import vgg13
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch


def train():
    epochs = 12
    lr = 0.01
    batch_size = 64
    dir_cifar10 = "E:/Data/cifar/cifar-10-batches-py"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = Cifar10Dataset(dir_cifar10, train=True, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    model = vgg13()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr
    )
    for epoch in range(epochs):
        for i, (data, target) in enumerate(train_dataloader):
            pred = model(data)
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                loss, current = loss.item(), (i+1) * batch_size
                print(f"loss: {loss:>7f}  [{current:>5d}/{len(train_dataset):>5d}]")


if __name__ == "__main__":
    train()