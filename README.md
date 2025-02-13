# Code Overview

# Task 1 Neural Network



# Task 2 Covolutional Neural Network

# Defining the CNN Class

The code defines a Convolutional Neural Network (CNN) class that inherits from nn.Module. The __init__ method initializes the layers of the network, and the forward method defines the forward pass of the network.

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()


# Convolutional Layers

The network contains three convolutional layers, each followed by a ReLU activation and a pooling layer:

#    conv1:

    Input channels: 3 (e.g., RGB image)
    
    Output channels: 64
    
    Kernel size: 3x3
    
    Stride: 1
    
    Padding: 0

#    conv2:

    Input channels: 64

    Output channels: 256

    Kernel size: 3x3

    Stride: 1

    Padding: 0

#    conv3:

    Input channels: 256

    Output channels: 128

    Kernel size: 3x3

    Stride: 1

    Padding: 0

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=0)

#  Pooling Layer

A MaxPooling layer is defined with a 2x2 kernel and a stride of 2:

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

# Fully Connected Layers

The fully connected layers (dense layers) are defined:

#    fc1:

    Input features: 128 * 30 * 30 (flattened output from the convolutional layers and pooling)

    Output features: 128

#    fc2:

    Input features: 128

    Output features: 64

#    fc3:

    Input features: 64

    Output features: 4 (e.g., 4 classes for classification)

        self.fc1 = nn.Linear(128 * 30 * 30, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)


 # Forward Pass

The forward method defines the forward pass of the network:

Pass the input x through the convolutional layers, followed by ReLU activation and pooling.

Flatten the tensor.

Pass the flattened tensor through the fully connected layers, with ReLU activation after the first two layers.

Return the output of the final fully connected layer.

        def forward(self, x):
          x = self.pool(F.relu(self.conv1(x)))
          x = self.pool(F.relu(self.conv2(x)))
          x = self.pool(F.relu(self.conv3(x)))
          x = x.view(x.size(0), -1)  # Flatten the tensor
          x = F.relu(self.fc1(x))
          x = F.relu(self.fc2(x))
          x = self.fc3(x)
          return x

# Loss Function and Optimizer

criterion: Defines the loss function as CrossEntropyLoss, which is suitable for classification problems.

optimizer: Defines the optimizer as Adam with a learning rate of 0.0001.

      criterion = nn.CrossEntropyLoss()
      optimizer = optim.Adam(model.parameters(), lr=0.0001)


