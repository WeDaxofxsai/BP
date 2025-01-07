import numpy as np
from data import *
import matplotlib.pyplot as plt


class BPNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.3):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.losses = []

        # Initialize weights and biases
        self.weights_input_hidden = np.random.uniform(
            -0.5, 0.5, (self.input_size, self.hidden_size)
        )
        self.bias_hidden = np.zeros(self.hidden_size)
        self.weights_hidden_output = np.random.uniform(
            -0.5, 0.5, (self.hidden_size, self.output_size)
        )
        self.bias_output = np.zeros(self.output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, x):
        # Forward propagation
        self.input = x
        self.hidden_input = (
            np.dot(self.input, self.weights_input_hidden) + self.bias_hidden
        )
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.final_input = (
            np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        )
        self.final_output = self.sigmoid(self.final_input)
        return self.final_output

    def backward(self, target):
        # Backward propagation
        output_error = target - self.final_output
        output_delta = output_error * self.sigmoid_derivative(self.final_output)

        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output += self.learning_rate * np.dot(
            self.hidden_output[:, None], output_delta[None, :]
        )
        self.bias_output += self.learning_rate * output_delta
        self.weights_input_hidden += self.learning_rate * np.dot(
            self.input[:, None], hidden_delta[None, :]
        )
        self.bias_hidden += self.learning_rate * hidden_delta

    def train(self, training_data, labels, epochs=10000):
        for epoch in range(epochs):
            for x, target in zip(training_data, labels):
                # print(target)
                self.forward(x)
                self.backward(target)
            loss = np.mean((labels - self.predict(training_data)) ** 2)
            self.losses.append(loss)
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, x):
        outputs = np.array([self.forward(sample) for sample in x])
        return outputs


# Prepare the training data (9x7 grid flattened into 63 inputs)
def prepare_training_data():

    training_d = []

    for i in range(10):
        # training_d.append(tranning_data[i])
        temp = []
        for a in tranning_data[i]:
            temp.extend(a)
        training_d.append(temp)

    labels = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ]

    return np.array(training_d), np.array(labels)


def get_test_data():
    res = []
    for i in test_data_2:
        temp = []
        for a in i:
            temp.extend(a)
        res.append(temp)
    return np.array(res)


# 添加画图函数
def plot_loss(losses):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(losses)), losses, label="Training Loss")
    plt.title("Training Loss Over Epochs", fontsize=16)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Create the network
    input_size = 63  # 9x7 grid
    hidden_size = 6
    output_size = 10  # Digits 0-9
    learning_rate = 0.3

    nn = BPNeuralNetwork(input_size, hidden_size, output_size, learning_rate)

    # Load training data
    training_data, labels = prepare_training_data()

    # Train the network
    nn.train(training_data, labels, epochs=600)

    # Test the network (replace with actual test data)

    test_d = get_test_data()  # Using training data as a placeholder
    predictions = nn.predict(test_d)
    print("Predictions:", predictions)
    for p in predictions:
        print(np.argmax(p))
    plot_loss(nn.losses)
