import sys
from matplotlib import pyplot
import numpy as np
import pdb
# from PIL import Image

from mnist import MNIST

class NeuralNet:

    def __init__(self, num_hidden_layers, units_per_layer):
        self.num_hidden_layers = num_hidden_layers
        self.units_per_layer = units_per_layer
        self.weights_layers = []
        self.features_layers = []
        self.bias_layers = []
        self.preactivation_features_layers = []
        self.activations = []
        self.act_derivs = []
        self.features = None
        self.labels = None
        self.graph_flag = False
        self.test_features = None
        self.test_labels = None

    def set_graph_errors(self, test_features, test_labels):
        self.graph_flag = True
        self.test_features = test_features
        self.test_labels = test_labels

    def train(self, features, labels, learning_rate, iterations):
        self.features = features
        self.labels = labels
        num_parameters, num_instances = features.shape
        #pdb.set_trace()

        # theta = np.zeros((num_parameters, 1))
        cost_function = []

        training_error = []
        testing_error = []

        num_iter = 0

        # initialization
        random_weights = np.vectorize(lambda x: np.random.normal(0, 0.1))
        self.weights_layers = [random_weights(np.empty((num_parameters, units_per_layer)))]
        self.features_layers = [features]
        self.bias_layers = [np.zeros((units_per_layer, 1))]
        self.preactivation_features_layers = []
        self.activations = []
        self.act_derivs = []
        for i in range(num_hidden_layers):
            self.weights_layers.append(random_weights(np.empty((units_per_layer, units_per_layer))))
            self.features_layers.append(np.empty(1))
            self.bias_layers.append(np.zeros((units_per_layer, 1)))
            self.preactivation_features_layers.append(np.empty(1))
            # self.activations.append(np.tanh)
            self.activations.append(np.vectorize(lambda x: x if x > 0 else 0)) # ReLU
            # self.activations.append(np.vectorize(sigmoid))
            # self.act_derivs.append(np.vectorize(lambda x: 1 - np.tanh(x) ** 2))
            self.act_derivs.append(np.vectorize(lambda x: 1 if x > 0 else 0)) # ReLU
            # self.act_derivs.append(np.vectorize(lambda x: sigmoid(x) * (1 - sigmoid(x))))

        self.features_layers.append(np.empty(1))
        self.preactivation_features_layers.append(np.empty(1))
        self.activations.append(np.vectorize(sigmoid))
        self.act_derivs.append(np.vectorize(lambda x: sigmoid(x) * (1 - sigmoid(x))))

        # gradient descent
        while num_iter < iterations:
            if self.graph_flag:
                testing_error.append(evaluate(self.predict(self.test_features), self.test_labels, True))

            predictions = self.forwardprop()

            if self.graph_flag:
                training_error.append(evaluate(predictions, self.labels, True))

            #cost = -1 * (np.log(predictions).dot(labels) + (np.log(np.ones((1, num_instances)) - predictions).dot(np.ones((num_instances, 1)) - labels))) / num_instances
            #cost = cost[0, 0]
            cost = self.calculate_cost(predictions, labels, num_instances)
            print("Iteration:", num_iter, "Cost:", cost)
            cost_function.append(cost)
            #print("Iterations:", num_iter)

            db_layers, dW_layers = self.backprop()

            # update weights
            for i, weights_diff in enumerate(dW_layers):
                layer = -1 - i
                self.weights_layers[layer] = self.weights_layers[layer] - weights_diff.T * learning_rate
                #print("Weights", self.weights_layers[layer].shape)
                self.bias_layers[layer] = self.bias_layers[layer] - db_layers[i] * learning_rate
                #print("Biases", self.bias_layers[layer].shape)
                # print(dW_layers[-1])
                # print(np.max(weights_layers[0]), np.min(weights_layers[0]))

            # gradient = np.matmul(x_matrix.T, predictions - y_vector) / num_instances
            # diff = alpha * gradient

            # theta -= diff
            num_iter += 1

        pyplot.plot(cost_function)
        pyplot.savefig("cost_function.png")
        pyplot.close()

        if self.graph_flag:
            pyplot.plot(training_error[len(training_error) // 20:], label="Training Error")
            pyplot.plot(testing_error[len(testing_error) // 20:], label="Testing Error")
            pyplot.legend()
            pyplot.savefig("train_test_error.png")
            pyplot.close()

    def forwardprop(self):
        # features has one more layer than all others, because its first layer is the input data
        for layer, weights in enumerate(self.weights_layers):
            #pdb.set_trace()
            # print("Calculating layer:", layer)
            self.preactivation_features_layers[layer] = np.matmul(weights.T, self.features_layers[layer]) + self.bias_layers[layer]
            #print("Z", self.preactivation_features_layers[layer].shape)
            self.features_layers[layer + 1] = self.activations[layer](self.preactivation_features_layers[layer])

        # print(features_layers)
        # print(features_layers[-1].shape)
        return np.ones((1, self.features_layers[-1].shape[0])).dot(self.features_layers[-1]) / self.features_layers[-1].shape[0]
        # return np.array(np.argmax(features_layers[-1], axis=0)).reshape((1, features_layers[-1].shape[1]))

    def backprop(self):
        db_layers = []  # in reverse order,
        dW_layers = []  # from output to input

        # output layer
        # dz = np.multiply(self.features_layers[-1] - self.labels.T,
                         # self.act_derivs[-1](self.preactivation_features_layers[-1]))  # broadcasting
        dz = self.features_layers[-1] - self.labels.T # broadcasting
        m, n = dz.shape
        dW = np.matmul(dz, self.features_layers[-2].T) / self.features_layers[-2].shape[1]
        db = dz.dot(np.ones((n, 1))) / n
        db_layers.append(db)
        dW_layers.append(dW)
        # print(np.max(dW))
        # print(np.min(dW))

        # hidden layers
        for i in range(len(self.weights_layers) - 1):
            layer = -1 - i
            dz = np.multiply(np.matmul(self.weights_layers[layer].T, dz),
                             self.act_derivs[layer - 1](self.preactivation_features_layers[layer - 1]))
            #print("dz", dz.shape)
            m, n = dz.shape  # do these change?
            dW = np.matmul(dz, self.features_layers[layer - 2].T) / self.features_layers[layer - 2].shape[1]
            #print("dW", dW.shape)
            # print(np.max(dW))
            # print(np.min(dW))

            db = dz.dot(np.ones((n, 1))) / n
            db_layers.append(db)
            dW_layers.append(dW)

        return db_layers, dW_layers

    def calculate_cost(self, predictions, labels, num_instances):
        return (-1 * (np.log(predictions).dot(labels) + (np.log(1 - predictions).dot(1 - labels))) / num_instances)[0, 0]
        # return np.linalg.norm(labels - predictions.T) ** 2 / num_instances / 2

    def predict(self, features):
        self.features_layers[0] = features
        predictions = self.forwardprop()
        self.features_layers[0] = self.features
        return predictions


def get_args(argv):
    args = {}
    while len(argv) > 1:
        if argv[0][0] == "-":
            args[argv[0][1:]] = argv[1]
        argv = argv[2:]
    return args

def read_data():
    data = MNIST("../")

    train_images, train_labels = data.load_training()
    test_images, test_labels = data.load_testing()
    return train_images, train_labels, test_images, test_labels

def parse_data(images, labels, size, id_num):
    step = len(images) // size
    reduced_images = np.empty((784, size))
    reduced_labels = np.empty((size, 1))

    image_num = 0

    # reducing, scaling, and reshaping done together
    while image_num < size:
        for i in range(len(images[image_num])):
            reduced_images[i, image_num] = images[image_num][i] / 255
        reduced_labels[image_num] = 1 if labels[image_num] == id_num else 0
        # reduced_images[image_num, 28*28] = 1
        image_num += 1

    return reduced_images, reduced_labels

def sigmoid(z):
    return 1.0 / (1 + np.exp(-1 * z))

def evaluate(predictions, labels, graph_errors=False):
    a = predictions.flatten().tolist()
    b = labels.flatten().tolist()
    a = list(map(round, a))
    wrong = 0
    false_pos = 0
    false_neg = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            wrong += 1
            if a[i] == 1 and b[i] == 0:
                false_pos += 1
            else:
                false_neg += 1
    error = wrong / len(a)
    false_pos /= len(a)
    false_neg /= len(a)
    if graph_errors:
        return error
    else:
        print("Error rate:", error)
        print("False positive rate:", false_pos)
        print("False negative rate:", false_neg)

# input processing
args = get_args(sys.argv)

train_images, train_labels, test_images, test_labels = read_data()
id_num = int(args["id"]) if "id" in args else 5  # my id
size = int(args["size"]) if "size" in args else 60000
alpha = float(args["alpha"]) if "alpha" in args else 1
iterations = int(args["iterations"]) if "iterations" in args else 1000
graph_errors_flag = (args["graph_errors"] in {"True", "true", "T", "t", "1"}) if "graph_errors" in args else False
num_hidden_layers = int(args["hidden_layers"]) if "hidden_layers" in args else 1
units_per_layer = int(args["units"]) if "units" in args else 5

train_images, train_labels = parse_data(train_images, train_labels, size, id_num)

test_images, test_labels = parse_data(test_images, test_labels, len(test_images), id_num)


# Create NN
two_layer = NeuralNet(num_hidden_layers, units_per_layer)

print("Beginning training...")
if graph_errors_flag:
    two_layer.set_graph_errors(test_images, test_labels)
two_layer.train(train_images, train_labels, alpha, iterations)

if graph_errors_flag:
    print("Graphs saved.")
    graph_errors = False

discretize = np.vectorize(lambda x: 1 if x > 0.5 else 0)

train_predictions = discretize(two_layer.forwardprop())

print("Training error:")
evaluate(train_predictions, train_labels)

test_predictions = discretize(two_layer.predict(test_images))

print("Testing error:")
evaluate(test_predictions, test_labels)


# test examples
#
# id_nums = []
# others = {}
#
# test_example_images, test_example_labels = MNIST("mnist").load_testing()
# num_images = 0
#
# example_images = []
#
# for image in test_example_images:
#     id_nums.append(image)
#     example_images.append(image)
#     num_images += 1
#     if num_images >= 30:
#         break
#
# n_example_images = []
# for image in example_images:
#     n_example_images.append(np.empty((784, 1)))
#     for i in range(784):
#         n_example_images[-1][i, 0] = image[i]
#
# for image in n_example_images:
#     print(two_layer.predict(image))
#
# count = 0
# for image in n_example_images:
#     Image.fromarray(image.reshape((28, 28))).convert("RGB").save("number" + str(count) + ".jpg", "JPEG")
#     count += 1
