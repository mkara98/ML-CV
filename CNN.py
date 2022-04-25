import datetime
import glob
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# number of class that we will predict
num_classes = 8

# Transform method for images to tensor
transformOfImage = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])


# Create dataset for Hand classification csv file and images
class HandDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.list = []
        for im in glob.glob(root_dir):
            self.list.append(im)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.list[idx]
        image = cv.imread(img_name)
        gender = self.landmarks_frame.iloc[idx, 2]
        aspectOfHand = self.landmarks_frame.iloc[idx, 6]
        image = cv.resize(image, (96, 96))

        sample = {'image': image, 'info': (gender + ' ' + aspectOfHand)}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample


# Create Convolutional Neural Networks
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv0 = nn.Conv2d(3, 32, 3,
                               stride=3)  # 3 input image channel, 32 output channels, 3x3 square convolution, stride 3
        self.conv1 = nn.Conv2d(32, 96, 3)  # 32 input , 96 output , 3x3 square convolution
        self.max1 = nn.MaxPool2d(2)  # 2x2 filter and 2 stride
        self.norm1 = nn.BatchNorm2d(96)  # Normalization layer
        self.conv2 = nn.Conv2d(96, 96, 4)  # 96 input, 96 output , 4x4 square convolution
        self.conv3 = nn.Conv2d(96, 256, 3)  # 96 input , 256 output , 3x3 square convolution
        self.max2 = nn.MaxPool2d(2)  # 2x2 filter and 2 stride
        self.norm2 = nn.BatchNorm2d(256)  # Normalization layer
        self.fc1 = nn.Linear(256 * 5 * 5, 520)  # First fully connected layer, 5 * 5 image size, 520 output
        self.drop1 = nn.Dropout2d(p=0.7)  # Dropout layer with p is 0.7
        self.fc2 = nn.Linear(520, 128)  # Second fully connected, 520 input 128 output
        self.drop2 = nn.Dropout2d(p=0.5)  # Dropout layer with p is 0.5
        self.fc3 = nn.Linear(128, num_classes)  # Final layer which output is number of class

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.relu(self.conv0(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.max1(x))
        x = F.relu(self.norm1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.max2(x))
        x = F.relu(self.norm2(x))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

# Create dataset
# Image file is Hands and csv file is HandInfo.csv
hand_data = HandDataset(root_dir="Hands/*", csv_file="HandInfo.csv",
                        transform=transformOfImage)

# Learning rates for train
lr_rate = 0.001
batch_size = 64

# Split dataset for train, validation and test
train_set, val_set, test_set = torch.utils.data.random_split(hand_data, [6400, 2300, 2376])

# Create Dataloader for train, validation and test
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# Loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr_rate, momentum=0.9)

loss_values = []
vall_values = []
Train_acc = []
Val_acc = []


# Print current time
def time():
    currentDT = datetime.datetime.now()
    print(str(currentDT))


def test(Loader_for_test):
    return_matrix = torch.zeros(num_classes, num_classes)
    # Without calculation
    with torch.no_grad():
        # Evaluate model which we created before
        net.eval()
        val_loss = 0
        accuracy1 = 0

        for data in Loader_for_test:
            # Move to device
            inputs, labels = data['image'].to(device), data['info']
            # Data to tensor type
            le = preprocessing.LabelEncoder()
            targets = le.fit_transform(labels)
            targets = torch.as_tensor(targets).to(device)
            # Forward pass
            output = net.forward(inputs)
            # Calculate loss
            valloss = criterion(output, targets)
            val_loss += valloss.item() * inputs.size(0)
            output = torch.exp(output)
            top_p, top_class = output.topk(1, dim=1)
            equals = top_class == targets.view(*top_class.shape)
            # Calculate accuracy
            accuracy1 += torch.mean(equals.type(torch.FloatTensor)).item()
            valid_loss = val_loss / len(Loader_for_test.dataset)
            _, predicts = torch.max(output, 1)
            # Convert tensor to numpy for confusion matrix
            matrix = confusion_matrix(y_true=targets.cpu(), y_pred=predicts.cpu())
            if matrix.shape == return_matrix.shape:
                return_matrix += matrix

    return accuracy1 / len(Loader_for_test), valid_loss, return_matrix, predicts


def train(epochs):
    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0
        # Training the model
        net.train()

        for data in train_loader:
            # Move to device
            inputs, labels = data['image'].to(device), data['info']
            le = preprocessing.LabelEncoder()
            targets = le.fit_transform(labels)
            targets = torch.as_tensor(targets).to(device)
            # Clear optimizers
            optimizer.zero_grad()
            # Forward pass
            output = net(inputs)
            # Loss
            loss = criterion(output, targets)
            # Calculate gradients (backpropogation)
            loss.backward()
            # Adjust parameters based on gradients
            optimizer.step()
            # Add the loss to the training set's running loss
            train_loss += loss.item() * inputs.size(0)
            output = torch.exp(output)
            top_p, top_class = output.topk(1, dim=1)
            equals = top_class == targets.view(*top_class.shape)
            train_acc += torch.mean(equals.type(torch.FloatTensor)).item()
        # Print the progress of our training
        print("Train end")
        accuracy_return, return_valid_loss, matrix = test(val_loader)
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_acc / len(train_loader.dataset)
        # Append values to array for plot
        loss_values.append(train_loss)
        Train_acc.append(accuracy_return)
        vall_values.append(return_valid_loss)
        Val_acc.append(accuracy_return)
        print("Train Loss: {:.4f} Train Acc: {:.4f} Vall Loss: {:.4f} Vall Acc: {:.4f}".format(train_loss,
                                                                                               train_acc * 100,
                                                                                               return_valid_loss,
                                                                                               accuracy_return * 100))

"""
time()
# Number of epochs
train(epochs=50)
# Save model for 
torch.save(net, "Model")
time()

# Show plot change in loss value
plt.plot(loss_values, label=' Train Loss values')
f1 = plt.gcf()
plt.legend(loc='best')
plt.show()
path = "Train loss"
f1.savefig(path)
# End

# Show Loss and Accuracy
plt.plot(loss_values, label='Loss values')
plt.plot(vall_values, label="Validation Loss")
plt.plot(Train_acc, label='Train Accuracy')
plt.plot(Val_acc, label="Validation Accuracy")
f2 = plt.gcf()
plt.legend(loc='best')
plt.show()
path2 = "Accuracy and loss"
f2.savefig(path2)
# end
"""

"""
# Prepare model for test
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = torch.load('Model')
net.eval()
testList = []

time()

# Test dataset for 30 times
for i in range(1):
    accuracy, val, matrix, pre = test(test_loader)
    testList.append(accuracy)
    print("Test set accuracy is {}".format(accuracy * 100))
    print(pre.cpu().numpy().astype(str))
time()

sn.heatmap(matrix, annot=True, fmt="g")
plt.show()

# Plot for Test accuracy
plt.plot(testList, label='Test accuracy rate')
f3 = plt.gcf()
plt.legend(loc='best')
plt.show()
path3 = "Accuracy rate"
f3.savefig(path3)
# End
"""
