import torch
import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms
from torch.autograd import Variable
# import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

# create a function instead of a script
# adjust to fit our input

trainset = []
testset = []

batch_size = 100
n_iters = 5500
num_epochs = n_iters / ( len(trainset) / batch_size )
num_epochs = int(num_epochs)

learning_rate = 0.001

# data loader: combines a dataset and a sampler and provides single- or multi-process
# iterators over the dataset

# shuffle: True if the data is reshuffled at every epoch (Default: False)
train_loader = DataLoader(dataset = trainset, batch_size = batch_size, shuffle = True)

test_loader = DataLoader(dataset = testset, batch_size = batch_size, shuffle = True)

# defining the convolutional network
# two convolutionar layers and one fully connected layer

class conv_network(nn.Module):
    def __init__(self):
        super(conv_network, self).__init__()

        # Constraints for level 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.batch1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size = 2)   # defauld stride = kernel_size

        # Constraints for level 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.batch2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Defining the Linear layer
        self.fc = nn.Linear(32 * 7 * 7, 10)

    # defining the network flow
    def forward(self, x):
        # Conv 1
        out = self.conv1(x)
        out = self.batch1(out)
        out = self.relu1(out)

        # Max Pool 1
        out = self.pool1(out)

        # Conv 2
        out = self.conv2(out)
        out = self.batch2(out)
        out = self.relu2(out)

        # Max Pool 2
        out = self.pool2(out)

        out = out.view(out.size(0), -1)
        # Linear Layer
        out = self.fc(out)

        return out


model = conv_network()

print(model.parameters)

# defining loss function and optimizer

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

losses = []

for epoch in range(num_epochs):
    for i, (images, labels)in enumerate(train_loader, 0):
        images = Variable(images.float())
        labels = Variable(labels)

        # Forward, Backward, Optimizer
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.data[0])

        if (i + 1) % 100 == 0:
            print('Epoch : %d/%d, Iter : %d/%d,  '
                  'Loss: %.4f% (epoch + 1, num_epochs, i + 1, len(trainset) // batch_size, '
                  'loss.data[0])')

# Save the model parameters

torch.save(model.state_dict(), './model/conv.pt')

# Load the model

conv1 = conv_network()
conv1.load_state_dict(torch.load('./model/conv.pt'))

# Evaluate the model

correct = 0
total = 0

for images, labels in test_loader:
    images = Variable(images.float())

    output = conv1(images)

    _, predicted = torch.max(output.data, 1)

    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Test accuracy is: %.4 %%' % (100 * correct / total))

# Plot Losses

loss = losses[0::100]
plt.xlabel('Epoches')
plt.ylabel('Losses')
plt.title('')
plt.plot(loss)
plt.show()

