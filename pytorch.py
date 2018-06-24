import torch 
import torch.nn as nn
import torchvision.datasets as dsets
from skimage import transform
import torchvision.transforms as transforms
from torch.autograd import Variable
import pandas as pd;
import numpy as np;
from torch.utils.data import Dataset, DataLoader
import random;
import math;


num_epochs = 5;
batch_size = 100;
learning_rate = 0.001;


class FashionMNISTDataset(Dataset):
    '''Fashion MNIST Dataset'''
    def __init__(self, csv_file, transform=None,download=False):
        data = pd.read_csv("mnist_train.csv");
        self.X = np.array(data.iloc[:, 1:]).reshape(-1, 1, 28, 28)#.astype(float);
        self.Y = np.array(data.iloc[:, 0]);
        
        del data;
        self.transform = transform;
        
    def __len__(self):
        return len(self.X);
    
    def __getitem__(self, idx):
        item = self.X[idx];
        label = self.Y[idx];
        
        if self.transform:
            item = self.transform(item);
        return (item, label);
        
        
    def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()

convert("train-images-idx3-ubyte", "train-labels-idx1-ubyte","mnist_train.csv", 60000)
convert("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte","mnist_test.csv", 10000)


train_dataset = FashionMNISTDataset(csv_file="mnist_train.csv");
test_dataset = FashionMNISTDataset(csv_file="mnist_test.csv")


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True);
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True);


class CNNmodel(nn.Module):
    def __init__(self):
        super(CNNmodel, self).__init__()
        #convolutional layer 1
        self.cnn1=nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.relu1=nn.ReLU()
        #self.batchnorm1=nn.BatchNorm2d(16)
        
        #Maxpool 1
        self.maxpool1=nn.MaxPool2d(kernel_size=2)
        
         #convolutional layer 2
        self.cnn2=nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.relu2=nn.ReLU()
        #self.batchnorm2=nn.BatchNorm2d(32)
        
        #Maxpool 2
        self.maxpool2=nn.MaxPool2d(kernel_size=2)
        
        #self.dropout = nn.Dropout(p=0.5)
        
        #Linear Layer
        self.fc1=nn.Linear(32*7*7, 10)
        
    def forward(self, x):
        #Convolution 1
        out=self.cnn1(x)
        out=self.relu1(out)
        
        #Maxpool 1
        out=self.maxpool1(out)
        
        #Convolution 2
        out=self.cnn2(out)
        out=self.relu2(out)
        
        #Maxpool 2
        out=self.maxpool2(out)
        
        out=out.view(out.size(0),-1)
       # out = self.dropout(out)
        out=self.fc1(out)
        return out
        
       
#instance of the Convolution Network
model = CNNmodel();
#loss function and optimizer
criterion = nn.CrossEntropyLoss();
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate);   


#training the model
losses = [];
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.float())
        labels = Variable(labels)
        correct=0
        total=0
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        #outputs=model(images)
        _, predicted=torch.max(outputs.data,1)
        total+=labels.size(0)
        correct+= (predicted==labels).sum()
        accuracy=100*(float(correct)/float(total))
        losses.append(loss.data[0]);
        
        if (i+1) % 100 == 0:
            print ('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f, Accuracy=%f' 
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0], accuracy))


#testing the model
model.eval()
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.float())
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
print('Test Accuracy of the model on the 10000 test images: %.4f %%' % (100 * correct / total))