from model import LeNet5
import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

batch_size = 32
lr = 1e-3
epochs = 40
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(train_loader,model,criterion,optimizer,epoch):
    train_loss = 0.0
    train_acc = 0.0
    train_len = 0.0
    total = 0.0
    model.train()
    for i,data in enumerate(train_loader):

        img, label = data
        img = img.to(device)
        label = label.to(device)
        output = model(img)

        loss = criterion(output,label)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        train_loss += loss.item()
        train_len += output.shape[0]

        #ACC
        pred = torch.argmax(output,1)
        num_correct = pred.eq(label.data.view_as(pred)).sum()
        train_acc += num_correct.item()
        total += label.size(0)

    print('TRAIN: Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
        epoch + 1, train_loss / (train_len), train_acc / (total)))

def valid(test_loader,model,criterion,epoch):
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    test_len = 0.0
    total = 0.0
    with torch.no_grad():

        for i, data in enumerate(test_loader):
            img, label = data

            img = img.to(device)
            label = label.to(device)
            output = model(img)

            loss = criterion(output, label)


            test_loss += loss.item()
            test_len += output.shape[0]

            # ACC
            pred = torch.argmax(output, 1)
            num_correct = pred.eq(label.data.view_as(pred)).sum()
            test_acc += num_correct.item()
            total += label.size(0)

    print('VALIED: Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
        epoch + 1, test_loss / (test_len), test_acc / (total)))



def main():
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=False)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(),download=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    model = LeNet5(1,10).to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = optim.SGD(model.parameters(),lr=lr)
    for epoch in range(epochs):

        train(train_loader,model,criterion,optimizer,epoch)

        valid(test_loader,model,criterion,epoch)

if __name__ =='__main__':
    main()