import numpy as np
import json
import time
import torch
import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms
from torch.autograd import Variable
# import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import dataset
from sklearn.model_selection import train_test_split
import json
import matplotlib.pyplot as plt


# convolutional network
# two convolutional layers and one fully connected layer
import input

# TODO:
#   CNN                                         [DONE]
#   Verification                                [DONE]
#   Fancy output, like a status or something    [    ]

cnn_status = {}
cnn_status['Parameters'] = {}
cnn_status['Results'] = {}

class conv_network(nn.Module):
    def __init__(self, number_of_classes=43):
        super(conv_network, self).__init__()

        # Constraints for level 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.batch1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)  # default stride = kernel_size

        # Constraints for level 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.batch2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Defining the Linear layer
        self.fc = nn.Linear(32 * 7 * 7, number_of_classes)

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


def train_model(trainset, n_iters, batch_size, learning_rate, number_of_classes, output_name):

    num_epochs = n_iters / (len(trainset) / batch_size)
    num_epochs = int(num_epochs)
    print("Total number of epochs:", num_epochs)
    cnn_status['Results']['Number_of_epochs'] = num_epochs

    # data loader: combines a dataset and a sampler and provides single- or multi-process
    # iterators over the dataset

    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)

    # define the conv network
    model = conv_network(number_of_classes)

    # defining loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses = []

    for epoch in range(num_epochs):
        print("Epoch: ", epoch)
        epoch_counter = 0
        for i, (images, labels) in enumerate(train_loader, 0):
            images = Variable(images.float())
            labels = Variable(labels)

            # Forward, Backward, Optimizer
            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels.squeeze(1).long())
            loss.backward()
            optimizer.step()

            losses.append(loss.data.item())

            epoch_counter = epoch_counter + 1

    # Save the model parameters
    torch.save(model.state_dict(), output_name)

    # Plot Losses
    loss = losses[0::epoch_counter]
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    plt.title('')
    plt.plot(loss)
    plt.savefig('./model/loss_plot.png')
    plt.show()

    cnn_status['Results']['Final_loss'] = losses[-1]


def evaluate_model(testset, batch_size, model_file):

    test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)

    # Load the model
    conv1 = conv_network()
    conv1.load_state_dict(torch.load(model_file))

    # Evaluate the model
    correct = 0
    total = 0

    # for images, labels in test_loader:
    for i, (images, labels) in enumerate(test_loader, 0):
        images = Variable(images.float())

        output = conv1(images)

        _, predicted = torch.max(output.data, 1)

        temp_labels = labels.squeeze(1)

        total += labels.size(0)
        correct += (predicted == temp_labels).sum().item()

    print("\nCorrectly classified", correct, "out of a total of", total, "images")
    print("Non-rounded test accuracy: ", 100.0*float(correct)/float(total))

    cnn_status['Results']['Test_accuracy'] = 100.0*float(correct)/float(total)


def preprocess_labels(labels):
    """
    Convert labels from type string to type torch.Tensor
    """

    new_labels = [int(label) for label in labels]
    new_labels = [torch.Tensor([label]) for label in new_labels]

    return new_labels


def preprocess_data(data):
    """
    Swap axes of the list of images to be used for CNN
    """

    X = np.asarray(data)

    X = np.swapaxes(X, 2, 3)
    X = np.swapaxes(X, 1, 2)

    return X


def input_and_preprocess(number_of_classes):
    """
    Load and preprocess data for use in CNN
    """
    data, labels = input.test_input(grayscale=False, image_range=number_of_classes)

    labels = preprocess_labels(labels)
    data = preprocess_data(data)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.20, random_state=42)

    train_data = []
    for i in range(len(X_train)):
        train_data.append([X_train[i], y_train[i]])

    test_data = []
    for i in range(len(X_test)):
        test_data.append([X_test[i], y_test[i]])

    return train_data, test_data


def cnn_pipeline():
    start_time = time.time()

    # input
    train_data, test_data = input_and_preprocess(number_of_classes=43)

    # starting parameters
    batch_size = 50
    n_iters = 4000
    learning_rate = 0.001
    number_of_classes = 43
    output_file = './model/conv.pt'

    cnn_status['Parameters']['Batch_size'] = batch_size
    cnn_status['Parameters']['Number_of_iterations'] = n_iters
    cnn_status['Parameters']['Learning_rate'] = learning_rate

    # train and evaluate
    train_model(train_data, n_iters, batch_size, learning_rate, number_of_classes, output_file)
    evaluate_model(test_data, batch_size, output_file)

    # print results and computation time
    end_time = time.time()
    cnn_status['Results']['Total_time(minutes)'] = (end_time - start_time)/60.0

    with open('./model/cnn_status.json', 'w') as json_file:
        json.dump(cnn_status, json_file, indent=1)


if __name__ == "__main__":
    cnn_pipeline()



