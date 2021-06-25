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
import data_reduction
import input

# TODO:
#   CNN                                         [DONE]
#   Verification                                [DONE]
#   Fancy output, like a status or something    [DONE]

cnn_status = {}
cnn_status['Parameters'] = {}
cnn_status['Results'] = {}


class conv_network(nn.Module):
    def __init__(self, number_of_classes=43):
        super(conv_network, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=100, kernel_size=(5,5), stride=1, padding=2)
        self.batch1 = nn.BatchNorm2d(100)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)  # default stride = kernel_size

        # Constraints for level 2
        self.conv2 = nn.Conv2d(in_channels=100, out_channels=150, kernel_size=(3,3), stride=1, padding=2)
        self.batch2 = nn.BatchNorm2d(150)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Constraints for level 3
        self.conv3 = nn.Conv2d(in_channels=150, out_channels=250, kernel_size=(3, 3), stride=1, padding=2)
        self.batch3 = nn.BatchNorm2d(250)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        # # Constraints for level 4
        # self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2)
        # self.batch4 = nn.BatchNorm2d(128)
        # self.relu4 = nn.LeakyReLU()
        # self.pool4 = nn.AvgPool2d(kernel_size=2)
        #
        # # Constraints for level 5
        # self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=2)
        # self.batch5 = nn.BatchNorm2d(256)
        # self.relu5 = nn.LeakyReLU()
        # self.pool5 = nn.AvgPool2d(kernel_size=2)

        # Defining the Linear layer

        # self.conv_drop = nn.Dropout2d()

        self.fc1 = nn.Linear(250 * 5 * 5, 350)
        self.fc2 = nn.Linear(350, number_of_classes)

    # defining the network flow
    def forward(self, x):
        # Conv 1
        out = self.conv1(x)
        out = self.batch1(out)
        out = self.relu1(out)

        # Max Pool 1
        out = self.pool1(out)

        # out = self.conv_drop(out)

        # Conv 2
        out = self.conv2(out)
        out = self.batch2(out)
        out = self.relu2(out)

        # Max Pool 2
        out = self.pool2(out)

        # out = self.conv_drop(out)

        # # Conv 3
        out = self.conv3(out)
        out = self.batch3(out)
        out = self.relu3(out)

        # Max Pool 3
        out = self.pool3(out)

        # out = self.conv_drop(out)

        # # Perform forward pass -------> Someone's idea!
        # x = self.bn1(F.max_pool2d(F.leaky_relu(self.conv1(x)),2))
        # x = self.conv_drop(x)
        # x = self.bn2(F.max_pool2d(F.leaky_relu(self.conv2(x)),2))
        # x = self.conv_drop(x)
        # x = self.bn3(F.max_pool2d(F.leaky_relu(self.conv3(x)),2))
        # x = self.conv_drop(x)
        # x = x.view(-1, 250*2*2)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)

        # # # Conv 4
        # out = self.conv4(out)
        # out = self.batch4(out)
        # out = self.relu4(out)
        #
        # # Max Pool 4
        # out = self.pool4(out)
        #
        # # # Conv 5
        # out = self.conv5(out)
        # out = self.batch5(out)
        # out = self.relu5(out)
        #
        # # Max Pool 5
        # out = self.pool5(out)

        # print("Afer pool6 size:", out.size())

        # print(out.size())
        out = out.view(out.size(0), -1)
        # Linear Layer
        out = self.fc1(out)
        out = self.fc2(out)

        return out


def train_model(trainset, n_iters, batch_size, learning_rate, number_of_classes, output_name, plot_output_name, number_of_epochs=None):

    if number_of_epochs:
        num_epochs = number_of_epochs
    else:
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
    plt.savefig(plot_output_name)
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

    # cnn_status['Results']['Test_accuracy'] = 100.0*float(correct)/float(total)

    return 100.0*float(correct)/float(total)


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


def separate_images(images, labels, data_reduction_indices):
    """
    Separate images and labels as per the data reduction
    """
    images_selected = []
    images_not_selected = []

    labels_selected = []
    labels_not_selected = []

    for i in range(len(images)):
        if i in data_reduction_indices:
            images_selected.append(images[i])
            labels_selected.append(labels[i])
        else:
            images_not_selected.append(images[i])
            labels_not_selected.append(labels[i])

    return images_selected, labels_selected, images_not_selected, labels_not_selected


def input_and_preprocess(number_of_classes, data_reduction_indices=None, augmentation_flag=False):
    """
    Load and preprocess data for use in CNN
    """
    data, labels = input.test_input(grayscale=False, image_range=number_of_classes, augmentation_flag=augmentation_flag)

    if data_reduction_indices:
        data, labels, additional_data, additional_labels = separate_images(data, labels, data_reduction_indices)

        additional_labels = preprocess_labels(additional_labels)
        additional_data = preprocess_data(additional_data)

    labels = preprocess_labels(labels)
    data = preprocess_data(data)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.001, random_state=42)

    train_data = []
    for i in range(len(X_train)):
        train_data.append([X_train[i], y_train[i]])

    test_data = []
    for i in range(len(X_test)):
        test_data.append([X_test[i], y_test[i]])
    if data_reduction_indices:
        for i in range(len(additional_data)):
            test_data.append([additional_data[i], additional_labels[i]])

    return train_data, test_data


def input_and_preprocess_alternate(data_reduction_filename='./model/reduced_indices_alternate.csv', augmentation_flag=False):
    train_images, train_labels, test_images, test_labels = data_reduction.data_reduction_alternate(augmentation_flag=augmentation_flag)

    y_train_before_rd = preprocess_labels(train_labels)
    y_test = preprocess_labels(test_labels)

    X_train_before_rd = preprocess_data(train_images)
    X_test = preprocess_data(test_images)

    # load data reduction indices
    data_reduction_indices = np.loadtxt(data_reduction_filename, delimiter=',')
    data_reduction_indices = [int(x) for x in data_reduction_indices]

    X_train, y_train, _, _ =\
        separate_images(X_train_before_rd, y_train_before_rd, data_reduction_indices)

    train_data = []
    for i in range(len(X_train)):
        train_data.append([X_train[i], y_train[i]])

    test_data = []
    for i in range(len(X_test)):
        test_data.append([X_test[i], y_test[i]])

    return train_data, test_data


def cnn_pipeline(status_filename,
                 model_output_name='./model/conv.pt',
                 plot_output_name='./model/loss_plot.png',
                 data_reduction_flag=False,
                 data_reduction_filename='./model/reduced_indices.csv',
                 augmentation_flag=False,
                 main_flag=True):
    start_time = time.time()

    # input
    # train_data, test_data = input_and_preprocess(number_of_classes=43)
    if main_flag:
        # data reduction or not
        if data_reduction_flag:
            data_reduction_indices = np.loadtxt(data_reduction_filename, delimiter=',')
            data_reduction_indices = [int(x) for x in data_reduction_indices]

            # input
            train_data, test_data = input_and_preprocess(number_of_classes=43,
                                                         data_reduction_indices=data_reduction_indices,
                                                         augmentation_flag=augmentation_flag)
        else:
            # input
            train_data, test_data = input_and_preprocess(number_of_classes=43, augmentation_flag=augmentation_flag)
    else:  # alternate_flag
        train_data, test_data = input_and_preprocess_alternate(augmentation_flag=augmentation_flag)

    # starting parameters
    batch_size = 8
    n_iters = 4000
    learning_rate = 0.001
    number_of_classes = 43
    output_file = model_output_name

    cnn_status['Parameters']['Data_reduction'] = data_reduction_flag
    cnn_status['Parameters']['Batch_size'] = batch_size
    cnn_status['Parameters']['Number_of_iterations'] = n_iters
    cnn_status['Parameters']['Learning_rate'] = learning_rate

    # train and evaluate
    train_time_1 = time.time()
    train_model(train_data, n_iters, batch_size, learning_rate, number_of_classes, output_file, plot_output_name, number_of_epochs=20)
    train_time_2 = time.time()
    acc = evaluate_model(test_data, batch_size, output_file)

    # print results and computation time
    end_time = time.time()
    cnn_status['Results']['Training_time(minutes)'] = (train_time_2 - train_time_1)/60.0
    cnn_status['Results']['Total_time(minutes)'] = (end_time - start_time)/60.0

    X_test, y_test = input.read_test_data(grayscale=False)

    y_test = preprocess_labels(y_test)
    X_test = preprocess_data(X_test)

    test_data = []
    for i in range(len(X_test)):
        test_data.append([X_test[i], y_test[i]])

    print("\nREAL TEST RESULTS:")
    if data_reduction_flag==False:
        print("No data reduction:")
    else:
        print("With data reduction:")

    accuracy = evaluate_model(test_data, batch_size, output_file)

    cnn_status['Results']['Accuracy_on_test_set'] = accuracy

    with open(status_filename, 'w') as json_file:
        json.dump(cnn_status, json_file, indent=2)

if __name__ == "__main__":
    status_file_1 = './model/cnn_status.json'
    output_file_1 = './model/conv.pt'
    plot_output_name_1 = './model/loss_plot.png'

    status_file_2 = './model/cnn_status_dr.json'
    output_file_2 = './model/conv_dr.pt'
    plot_output_name_2 = './model/loss_plot_dr.png'


    cnn_pipeline(status_filename=status_file_1,
                 model_output_name=output_file_1,
                 plot_output_name=plot_output_name_1,
                 data_reduction_flag=False,
                 augmentation_flag=True,
                 main_flag=True)

    cnn_pipeline(status_filename=status_file_2,
                 model_output_name=output_file_2,
                 plot_output_name=plot_output_name_2,
                 data_reduction_flag=True,
                 augmentation_flag=True,
                 main_flag=True)


