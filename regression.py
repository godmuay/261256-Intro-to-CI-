import numpy as np
import matplotlib.pyplot as plt
import copy

class NN:
    def __init__(self, layer, learning_rate=0.1, momentum_rate=0.9, activation_function='sigmoid'):
        self.V = []
        self.layer = layer
        self.momentum_rate = momentum_rate
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.w, self.delta_w, self.b, self.delta_bias, self.local_gradient = self.init_inform(layer)
        self.keep_error = []  # Initialize keep_error to keep track of errors

    def activation(self, x):
        if self.activation_function == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.activation_function == "relu":
            return np.where(x > 0, x, 0.0)
        elif self.activation_function == "tanh":
            return np.tanh(x)
        elif self.activation_function == "linear":
            return x
    
    def activation_diff(self, x):
        if self.activation_function == "sigmoid":
            return x * (1 - x)
        elif self.activation_function == "relu":
            return np.where(x > 0, 1.0, 0.0)
        elif self.activation_function == "tanh":
            return 1 - x**2
        elif self.activation_function == "linear":
            return np.ones_like(x)

    def init_inform(self, layer):
        weights = []
        delta_weights = []
        biases = []
        delta_biases = []
        local_gradientes = [np.zeros(layer[0])]
        for i in range(1, len(layer)):
            weights.append(np.random.rand(layer[i], layer[i-1]))
            delta_weights.append(np.zeros((layer[i], layer[i-1])))
            biases.append(np.random.rand(layer[i]))
            delta_biases.append(np.zeros(layer[i]))
            local_gradientes.append(np.zeros(layer[i]))
        return weights, delta_weights, biases, delta_biases, local_gradientes
    
    def feed_forward(self, input):
        self.V = [input]
        for i in range(len(self.layer) - 1):
            self.V.append(self.activation((self.w[i] @ self.V[i]) + self.b[i]))

    def back_propagation(self, design_output):
        for i, j in enumerate(reversed(range(1, len(self.layer)))):
            if i == 0:
                error = np.array(design_output - self.V[j])
                self.local_gradient[j] = error * self.activation_diff(self.V[j])
            else:
                self.local_gradient[j] = self.activation_diff(self.V[j]) * (self.w[j].T @ self.local_gradient[j+1])
            self.delta_w[j-1] = (self.momentum_rate * self.delta_w[j-1]) + np.outer(self.learning_rate * self.local_gradient[j], self.V[j-1])
            self.delta_bias[j-1] = (self.momentum_rate * self.delta_bias[j-1]) + self.learning_rate * self.local_gradient[j]
            self.w[j-1] += self.delta_w[j-1]
            self.b[j-1] += self.delta_bias[j-1]
        return np.sum(error**2) / 2

    def train(self, input, design_output, Epoch=10000, L_error=0.001):
        N = 0
        er = 10000
        while N < Epoch and er > L_error:
            actual_output = []
            er = 0
            for i in range(len(input)):
                self.feed_forward(input[i])
                actual_output.append(self.V[-1])
                er += self.back_propagation(design_output[i])
            er /= len(input)
            self.keep_error.append(er)
            N += 1
            print(f"Epoch = {N} | AV_Error = {er}")

    def test(self, input, design_output, type="regression"):
        actual_output = []
        for i in input:
            self.feed_forward(i)
            actual_output.append(self.V[-1])
    
        if type == "classification":
            correct_predictions = 0
            for i in range(len(actual_output)):
                pred_class = 0 if actual_output[i][0] > actual_output[i][1] else 1
                true_class = 0 if design_output[i][0] > design_output[i][1] else 1
                if pred_class == true_class:
                    correct_predictions += 1
            accuracy = (correct_predictions / len(actual_output)) * 100
            print(f"Accuracy = {accuracy}%")
            
            # Update actual_output and design_output for plotting
            actual_output = [0 if output[0] > output[1] else 1 for output in actual_output]
            design_output = [0 if output[0] > output[1] else 1 for output in design_output]

            plt.figure(figsize=(12, 6))  # Create a specific figure size for combined plots

            # Plot MSE vs Epoch
            plt.subplot(2, 1, 1)
            plt.plot(self.keep_error, color='skyblue')
            plt.title('MSE vs. Epoch of TrainSet')
            plt.xlabel('Epoch')
            plt.ylabel('MSE')

            # Plot Actual Output vs Design Output
            plt.subplot(2, 1, 2)
            plt.plot(actual_output, label='Actual Output', color='skyblue', marker='o')
            plt.plot(design_output, label='Design Output', color='navy', marker='x')
            plt.plot([100 if abs(actual_output[i] - design_output[i]) > 100 else abs(actual_output[i] - design_output[i]) for i in range(len(actual_output))], label='Error', color='red', linestyle='--')
            plt.xlabel('Sample')
            plt.ylabel('Output')
            plt.title(f'Actual Output vs. Design Output of TestSet\nAccuracy = {accuracy:.2f}%')
            plt.legend()

            plt.tight_layout()  # Adjust subplot positions to fit the figure area
            plt.show()  # Display all plots together

        else:
            actual_output = [element[0] for element in actual_output]
    
            plt.figure(figsize=(12, 6))  # Create a specific figure size for combined plots
            
            # Plot MSE vs Epoch
            plt.subplot(2, 1, 1)
            plt.plot(self.keep_error, color='skyblue')
            plt.title('MSE vs. Epoch of TrainSet')
            plt.xlabel('Epoch')
            plt.ylabel('MSE')

            # Plot Actual Output vs Design Output
            plt.subplot(2, 1, 2)
            plt.plot(actual_output, label='Actual Output', color='skyblue', marker='o')
            plt.plot(design_output, label='Design Output', color='navy', marker='x')
            plt.plot([100 if abs(actual_output[i] - design_output[i]) > 100 else abs(actual_output[i] - design_output[i]) for i in range(len(actual_output))], label='Error', color='red', linestyle='--')
            plt.xlabel('Sample')
            plt.ylabel('Output')
            plt.title('Actual Output vs. Design Output of TestSet')
            plt.legend()

            plt.tight_layout()  # Adjust subplot positions to fit the figure area
            plt.show()  # Display all plots together

def Read_Data1(filename='C:/Users/66882/OneDrive - Chiang Mai University/Desktop/ci/Flood_dataset.txt'):
    data = []
    input = []
    design_output = []
    with open(filename) as f:
        for line in f.readlines()[2:]:
            data.append([float(element[:-1]) for element in line.split()])
    data = np.array(data)
    np.random.shuffle(data)
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    epsilon = 1e-8  # A very small value to avoid division by zero
    data = (data - min_vals) / (max_vals - min_vals + epsilon)
    for i in data:
        input.append(i[:-1])
        design_output.append(np.array(i[-1]))
    return input, design_output

def k_fold_varidation(data, k=10):
    test = []
    train = []
    for i in range(0, len(data), int(len(data) / k)):
        test.append(data[i:i + int(len(data) / k)])
        train.append(data[:i] + data[i + int(len(data) / k):])
    return train, test

if __name__ == "__main__":
    # Parameters
    k = 5  # Set k-fold validation
    layer = [8, 16, 1]  # Define the neural network layer structure for regression
    learning_rate = 0.3
    momentum_rate = 0.9
    Max_Epoch = 1000
    AV_error = 0.001
    activation_function = 'sigmoid'  # Activation function choice
    data_type = "regression"  # Data type set to regression

    print(f"k-fold-validation = {k}")
    print(f"learning rate = {learning_rate}")
    print(f"momentum rate = {momentum_rate}")
    print(f"Max Epoch = {Max_Epoch}")
    print(f"AV error = {AV_error}")
    print(f"data type = {data_type}")

    # Perform k-fold validation
    input, design_output = Read_Data1()
    input_train, input_test = k_fold_varidation(input, k)
    design_output_train, design_output_test = k_fold_varidation(design_output, k)

    # Initialize model
    nn = NN(layer, learning_rate, momentum_rate, activation_function)

    # Test the model using cross-validation
    for i in range(len(input_train)):
        nn_copy = copy.deepcopy(nn)
        nn_copy.train(input_train[i], design_output_train[i], Epoch=Max_Epoch, L_error=AV_error)
        nn_copy.test(input_test[i], design_output_test[i], type=data_type)
