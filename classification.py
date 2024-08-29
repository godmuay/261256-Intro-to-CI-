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
        for i in range(1, len(layer), 1):
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
        for i, j in enumerate(reversed(range(1, len(self.layer), 1))):
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
        keep_error = []
        keep_accuracy = []
        er = 10000
        while N < Epoch and er > L_error:
            actual_output = []
            er = 0
            correct_predictions = 0
            for i in range(len(input)):
                self.feed_forward(input[i])
                actual_output.append(self.V[-1])
                er += self.back_propagation(design_output[i])
                predicted = 0 if self.V[-1][0] > self.V[-1][1] else 1
                correct_predictions += int(predicted == np.argmax(design_output[i]))
            er /= len(input)
            accuracy = correct_predictions / len(input) * 100
            keep_error.append(er)
            keep_accuracy.append(accuracy)
            N += 1
            print(f"Epoch = {N} | AV_Error = {er} | Accuracy = {accuracy}%")
        return keep_error, keep_accuracy

    def test(self, input, design_output):
        actual_output = []
        predicted_output = []
        for i in range(len(input)):
            self.feed_forward(input[i])
            predicted = np.argmax(self.V[-1])
            predicted_output.append(predicted)
            actual_output.append(np.argmax(design_output[i]))
        return predicted_output, actual_output

def Read_Data2(filename='C:/Users/66882/OneDrive - Chiang Mai University/Desktop/ci/cross.txt'):
    data = []
    input = []
    design_output = []
    with open(filename) as f:
        a = f.readlines()
        for line in range(1, len(a), 3):
            z = np.array([float(element) for element in a[line][:-1].split()])
            zz = np.array([float(element) for element in a[line + 1].split()])
            data.append(np.append(z, zz))
    data = np.array(data)
    np.random.shuffle(data)
    for i in data:
        input.append(i[:-2])
        design_output.append(i[-2:])
    return input, design_output

def k_fold_varidation(data, k=10):
    test = []
    train = []
    for i in range(0, len(data), int(len(data) / k)):
        test.append(data[i:i + int(len(data) / k)])
        train.append(data[:i] + data[i + int(len(data) / k):])
    return train, test

def create_confusion_matrix(actual, predicted, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for a, p in zip(actual, predicted):
        cm[a][p] += 1
    return cm

def plot_confusion_matrix(cm, labels, cmap='Greens'):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)

    # Set the color of text to black
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, f'{cm[i, j]}', horizontalalignment="center",
                 color="black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

if __name__ == "__main__":
    k = 3  # Number of folds for cross-validation
    layer = [2, 16, 2]  # Layer configuration for classification
    learning_rate = 0.3
    momentum_rate = 0.8
    Max_Epoch = 500
    AV_error = 0.001
    activation_function = 'sigmoid'  # Activation function for the network

    # Load data
    input, design_output = Read_Data2()

    # Perform k-fold cross-validation
    input_train, input_test = k_fold_varidation(input, k)
    design_output_train, design_output_test = k_fold_varidation(design_output, k)

    num_classes = 2  # Number of classes for classification

    # Create and train the model
    nn = NN(layer, learning_rate, momentum_rate, activation_function) 

    all_accuracies = []
    combined_cm = np.zeros((num_classes, num_classes), dtype=int)

    for i in range(len(input_train)):
        nn_copy = copy.deepcopy(nn)
        keep_error, keep_accuracy = nn_copy.train(input_train[i], design_output_train[i], Epoch=Max_Epoch, L_error=AV_error)
        predicted_output, actual_output = nn_copy.test(input_test[i], design_output_test[i])
        
        # Compute and accumulate confusion matrix
        cm = create_confusion_matrix(actual_output, predicted_output, num_classes)
        combined_cm += cm

        # Collect accuracy for averaging
        all_accuracies.extend(keep_accuracy)

    # Calculate average accuracy
    avg_accuracy = np.mean(all_accuracies)
    print(f'Average Accuracy: {avg_accuracy:.2f}%')

    # Plot combined confusion matrix
    plt.figure(figsize=(8, 6))
    plot_confusion_matrix(combined_cm, labels=['Class 0', 'Class 1'])
    plt.title('Combined Confusion Matrix')

    plt.show()
