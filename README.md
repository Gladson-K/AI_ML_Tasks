

# Dataset Links for this Implementation

MNIST Dataset (Neural Network) : https://www.kaggle.com/datasets/hojjatk/mnist-dataset

Corn or Maize Dataset (Convolutional Neural Network) : https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset

# Code Overview

# Task 1 Neural Network

Sure! Let's break down this code step-by-step:

### Libraries and Functions

1. **Importing NumPy**:
    
    import numpy as np
    

2. **Sigmoid Function**:
    ```python
    def sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    ```
    - **Purpose**: The sigmoid function is a common activation function in neural networks. It maps any real-valued number to a value between 0 and 1.
    - **Clipping**: The input is clipped to avoid numerical overflow issues.

3. **Generating Weights**:
    ```python
    def generate_wt(x, y):
        return np.random.randn(x, y)
    ```
    - **Purpose**: This function generates a matrix of weights with dimensions `x` by `y` using a normal distribution.

4. **Generating Biases**:
    ```python
    def generate_bias(y, bias):
        if bias == 1:
            return np.ones(y)
        elif bias == 0:
            return np.zeros(y)
        else:
            return np.random.randn(y)
    ```
    - **Purpose**: This function generates a bias vector based on the provided `bias` value (1 for ones, 0 for zeros, else random values).

### Neural Network Class

5. **Class Initialization**:
    ```python
    class Neuralnetwork():
        def __init__(self, alpha, epochs):
            self.alpha = alpha
            self.epochs = epochs
    ```
    - **Purpose**: This initializes the neural network with the learning rate `alpha` and the number of training epochs.

6. **Feed Forward**:
    ```python
    def feed_forward(self, x, w1, b1, w2, b2, w3, b3):
        z1 = np.dot(x, w1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, w2) + b2
        a2 = sigmoid(z2)
        z3 = np.dot(a2, w3) + b3
        a3 = sigmoid(z3)
        return a3
    ```
    - **Purpose**: This function performs the forward pass through the neural network.

7. **Loss Function**:
    ```python
    def loss(self, out, Y):
        s = np.square(out - Y)
        return np.sum(s) / len(s)
    ```
    - **Purpose**: This calculates the loss using Mean Squared Error (MSE) between the predicted output and the actual output.

8. **Back Propagation**:
    ```python
    def back_propagation(self, x, y, w1, b1, w2, b2, w3, b3):
        z1 = np.dot(x, w1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, w2) + b2
        a2 = sigmoid(z2)
        z3 = np.dot(a2, w3) + b3
        a3 = sigmoid(z3)
        diff_3 = a3 - y
        diff_2 = np.multiply(np.dot(diff_3, w3.T), np.multiply(a2, 1 - a2))
        diff_1 = np.multiply(np.dot(diff_2, w2.T), np.multiply(a1, 1 - a1))
        w1_diff = np.dot(x.T.reshape(-1, 1), diff_1.reshape(1, -1))
        w2_diff = np.dot(a1.T.reshape(-1, 1), diff_2.reshape(1, -1))
        w3_diff = np.dot(a2.T.reshape(-1, 1), diff_3.reshape(1, -1))
        b1_diff = diff_1
        b2_diff = diff_2
        b3_diff = diff_3
        w1 -= self.alpha * w1_diff
        w2 -= self.alpha * w2_diff
        w3 -= self.alpha * w3_diff
        b1 -= self.alpha * b1_diff
        b2 -= self.alpha * b2_diff
        b3 -= self.alpha * b3_diff
        return w1, b1, w2, b2, w3, b3
    ```
    - **Purpose**: This function performs the backward pass, calculating gradients and updating weights and biases accordingly.

9. **Training**:
    ```python
    def train(self, X, Y, w1, b1, w2, b2, w3, b3):
        acc = []
        losss = []
        assert len(X) == len(Y), "Mismatch in number of samples between X and Y"
        for j in range(self.epochs):
            l = []
            for i in range(len(X)):
                out = self.feed_forward(X[i], w1, b1, w2, b2, w3, b3)
                l.append(self.loss(out, Y[i]))
                w1, b1, w2, b2, w3, b3 = self.back_propagation(X[i], Y[i], w1, b1, w2, b2, w3, b3)
            print(f"Epoch {j+1} ===== Accuracy: {(1 - (sum(l) / len(X))) * 100:.2f}% ==== Loss: {sum(l) / len(X):.4f}\n")
            acc.append((1 - (sum(l) / len(X))) * 100)
            losss.append(sum(l) / len(X))
        return acc, losss, w1, b1, w2, b2, w3, b3
    ```
    - **Purpose**: This function trains the neural network over a specified number of epochs, calculating accuracy and loss for each epoch.

10. **Prediction**:
    ```python
    def predict(self, X, w1, b1, w2, b2, w3, b3):
        predictions = []
        for i in range(len(X)):
            out = self.feed_forward(X[i], w1, b1, w2, b2, w3, b3)
            predictions.append(np.argmax(out))
        return np.array(predictions)
    ```
    - **Purpose**: This function makes predictions on new data by performing a forward pass through the network.

### Model Initialization and Parameters

- **Initialize the Neural Network**:
  ```python
    model_nn = Neuralnetwork(alpha=0.0005, epochs=20)
  ```
    - **Purpose**: Creates a neural network model with a learning rate of 0.0005 and 20 epochs.

- **Generate Weights**:
    ```python
    w1 = generate_wt(784, 64)
    w2 = generate_wt(64, 32)
    w3 = generate_wt(32, 10)
    ```
    - **Purpose**: Creates weight matrices for each layer of the network with the specified dimensions.

- **Generate Biases**:
    ```python
    print("Enter bias term 1 for one 0 for zero and 2 for random bias value")
    b1 = generate_bias(64, 0)
    b2 = generate_bias(32, 0)
    b3 = generate_bias(10, 0)
    ```
    - **Purpose**: Creates bias vectors for each layer based on user input. In this case, all biases are set to zero.


# Task 2 Covolutional Neural Network

# Defining the CNN Class

The code defines a Convolutional Neural Network (CNN) class that inherits from nn.Module. The __init__ method initializes the layers of the network, and the forward method defines the forward pass of the network.
```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
```

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
```python
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=0)
```
#  Pooling Layer

A MaxPooling layer is defined with a 2x2 kernel and a stride of 2:
```python
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
```
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
```python
        self.fc1 = nn.Linear(128 * 30 * 30, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)
```

 # Forward Pass

The forward method defines the forward pass of the network:

Pass the input x through the convolutional layers, followed by ReLU activation and pooling.

Flatten the tensor.

Pass the flattened tensor through the fully connected layers, with ReLU activation after the first two layers.

Return the output of the final fully connected layer.
```python
        def forward(self, x):
          x = self.pool(F.relu(self.conv1(x)))
          x = self.pool(F.relu(self.conv2(x)))
          x = self.pool(F.relu(self.conv3(x)))
          x = x.view(x.size(0), -1)  # Flatten the tensor
          x = F.relu(self.fc1(x))
          x = F.relu(self.fc2(x))
          x = self.fc3(x)
          return x
```
# Loss Function and Optimizer

criterion: Defines the loss function as CrossEntropyLoss, which is suitable for classification problems.

optimizer: Defines the optimizer as Adam with a learning rate of 0.0001.
```python
      criterion = nn.CrossEntropyLoss()
      optimizer = optim.Adam(model.parameters(), lr=0.0001)
```

