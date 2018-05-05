import numpy as np
import time
import pdb
import matplotlib.pyplot as plt 
from mnist import MNIST

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

class NeuronLayer:
    # activation is the activation function for this layer
    # prev_layer is layer connecting into this one
    # n is the number of neurons in previous layer
    # m is the number of neurons in this layer
    def __init__(self, activation_type, m, n):
        self.connections = n
        self.width = m
        self.type = activation_type
        self.reset()

    def __getitem__(self, key):
        return self.weights[key]

    def calculate(self, x):
        #print("Calculate")
        #print("X is", x)
        #print("Weights are", self.weights) # This is a 3rd order tensor
        Z = np.matmul(self.weights, x) + self.biases
        #print("Z value is", Z)
        return [self._calculate(z) for z in Z]

    def calculate_derivative(self, x):
        Z = np.matmul(self.weights, x) + self.biases
        return [self._calculate_derivative(z) for z in Z]

    def reset(self):
        self.weights = [[np.random.normal(scale=1.5) for x in range(self.connections)] 
                for y in range(self.width)]
        #print("On reset", self.weights)
        self.biases = np.zeros(self.width)

    def _calculate(self, z):
        if self.type == "sigmoid":
            return sigmoid(z)
        elif self.type == "tanh":
            return np.tanh(z)
        elif self.type == "relu":
            return max(0, z)

    def _calculate_derivative(self, z):
        if self.type == "sigmoid":
            return sigmoid(z) * (1 - sigmoid(z))
        elif self.type == "tanh":
            return 1 - np.tanh(z)**2
        elif self.type == "relu":
            return 1 if z > 0 else 0

    def add_weights(self, delta):
        #print("Adding weights before", self.weights)
        #print(delta)
        self.weights = np.add(self.weights, delta)
        #print("Adding weights after", self.weights)

    def add_biases(self, delta):
        self.biases = np.add(self.biases, delta)

    def get_weights(self):
        return self.weights

    def get_biases(self):
        return self.biases

class NeuralNetwork:
    def __init__(self):
        self.network = []

    def reset(self):
        for layer in self.network:
            layer.reset()

    def add_layer(self, layer):
        # we need the number of connections to match the number of neurons in 
        # the previous layer
        if len(self.network) == 0 or layer.connections == self.network[-1].width:
            self.network.append(layer)
        else:
            raise Exception("New layer does not match previous layer")

    def pop_layer(self):
        return self.network.pop()

    def predict(self, instance):
        X = instance
        i = 0
        for layer in self.network:
            #print("LAYER NUMBER", i, "IS", X)
            X = layer.calculate(X)
            i +=1
        return X

    def cost(self, instances, labels):
        #print("Starting COST")
        #print("Instances", instances)
        return .5 * (1.0 / len(instances)) * np.linalg.norm(np.subtract(labels, [self.predict(x) for x in instances]))

    def backpropagate(self, instances, labels, alpha):
        weight_changes = [[[0 for y in x] for x in layer] for layer in self.network]
        bias_changes = [[0 for x in range(layer.width)] for layer in self.network]
        # this might be inefficient
        for instance, label in zip(instances, labels):
            As = [instance]
            delta = []
            for layer in self.network:
                As.append(layer.calculate(As[-1]))

            # choosing array format for easy portability
            delta.append(np.multiply(np.subtract(self.predict(instance), label),
                layer.calculate_derivative(As[-2])))

            #print("Prediction", self.predict(instance), "label", label, "delta", delta[-1])
            for x in range(1, len(self.network))[::-1]:
                # print(x, "layer from back, delta is", delta[-1])
                layer = self.network[x]
                lhs = np.dot(np.transpose(layer.get_weights()), delta[-1])
                #print("As", As[x])
                bias_changes[x] = np.add(bias_changes[x], delta[-1]) # bias change is the error
                temp = np.add(weight_changes[x], np.matmul(np.transpose([delta[-1]]), [As[x]]))
                weight_changes[x] = temp
                #pdb.set_trace()
                derivs = self.network[x-1].calculate_derivative(np.transpose(As[x-1]))
                delta.append(np.multiply(lhs, derivs))
                #pdb.set_trace()

            #pdb.set_trace()
            weight_changes[0] = np.add(weight_changes[0], np.matmul(np.transpose([delta[-1]]), [As[0]]))
            bias_changes[0] = np.add(bias_changes[0], delta[-1])

            #pdb.set_trace()

        #print(bias_changes[-1][0])

        for x in range(len(weight_changes)):
            weight_changes[x] = np.dot(-alpha / len(instances), weight_changes[x])
            bias_changes[x] = np.dot(-alpha / len(instances), bias_changes[x])

        #print(bias_changes)

        

        # print("Weight change of final layer is", weight_changes[1]) 
        for x in range(len(self.network)):
            self.network[x].add_weights(weight_changes[x])
            self.network[x].add_biases(bias_changes[x])

def toBinary(z):
    if z >= 0.5:
        return 1
    else:
        return 0

def normalize_images(images):
    for x in range(len(images)):
        images[x] = np.multiply(1.0/255, images[x])
    return images

def normalize_labels(labels):
    newLabels = []
    for label in labels:
        if label == 5:
            newLabels.append([1])
        else:
            newLabels.append([0])
    return newLabels

def test_alphas(images, labels, iters, tImages = [], tLabels = []):
    network = NeuralNetwork()
    network.add_layer(NeuronLayer("tanh", 5, 784))
    network.add_layer(NeuronLayer("sigmoid", 1, 5))
    alphas = [.1, 1, 5, 10, 50]
    alphaPlots = []
    for alpha in alphas:
        network.reset()
        print("Testing alpha =", alpha)
        #costs = [network.cost(images, labels)]
        costs = [network.cost(images, labels)]
        for x in range(iters):
            network.backpropagate(images, labels, alpha)
            costs.append(network.cost(images, labels))
            #print("Iteration", x+1, "of", iters, "| Cost is", costs[-1])
        alphaPlot, = plt.plot(costs, label="alpha=" + str(alpha))
        alphaPlots.append(alphaPlot)
        test_correct(network, tImages, tLabels, True)

        print(costs)

    plt.legend(alphaPlots, list(map(lambda x: "alpha=" + str(x), alphas)))
    plt.xlabel("iteration number")
    plt.ylabel("cost")
    plt.savefig("nn_cost_alpha_comparison.jpg")
    plt.close()

def test_activations(images, labels, iters, alpha, tImages = [], tLabels = []):
    network = NeuralNetwork()
    networkB = NeuralNetwork()
    networkB.add_layer(NeuronLayer("relu", 5, 784))
    networkB.add_layer(NeuronLayer("sigmoid", 1, 5))
    network.add_layer(NeuronLayer("tanh", 5, 784))
    network.add_layer(NeuronLayer("sigmoid", 1, 5))

    costs = [network.cost(images, labels)]
    costsB = [networkB.cost(images, labels)]

    start = time.time()
    for x in range(iters):
        network.backpropagate(images, labels, alpha)
        costs.append(network.cost(images, labels))
    print("Tanh took", time.time() - start, "to run")

    start = time.time()
    for x in range(iters):
        networkB.backpropagate(images, labels, alpha / 3)
        costsB.append(networkB.cost(images, labels))
        #print("Iteration", x+1, "of", iters, "| Cost is", costs[-1])
    print("Relu took", time.time() - start, "to run")

    aPlot, = plt.plot(costs, label="Tanh")
    bPlot, = plt.plot(costsB, label="Relu")

    test_correct(network, tImages, tLabels, True)
    test_correct(networkB, tImages, tLabels, True)

    plt.legend([aPlot, bPlot], ["Tanh", "Relu"])
    plt.xlabel("iteration number")
    plt.ylabel("cost")
    plt.savefig("nn_activation.jpg")
    plt.close()

def test_layers(images, labels, iters, alpha, tImages = [], tLabels = []):
    networks = [NeuralNetwork() for x in range(3)]
    for x in range(3):
        networks[x].add_layer(NeuronLayer("relu", 10, 784))
        for y in range(x):
            networks[x].add_layer(NeuronLayer("relu", 10, 10))
        networks[x].add_layer(NeuronLayer("relu", 1, 10))

    costs = [networks[x].cost(images, labels) for x in range(3)]

    for network in networks:
        start = time.time()
        for x in range(iters):
            network.backpropagate(images, labels, alpha)
            costs.append(network.cost(images, labels))
        print("Layer with", len(network.network) - 1, "layers took", time.time() - start, "to run")

    layer_plots = [plt.plot(x)[0] for x in costs]

    for x in range(3):
        print("Network with", x + 1, "hidden layers")
        test_correct(networks[x], tImages, tLabels, True)

    plt.legend(layer_plots, ["1 hidden layer", "2 hidden layers", "3 hidden layers"])
    plt.xlabel("iteration number")
    plt.ylabel("cost")
    plt.savefig("nn_layers.jpg")
    plt.close()

def test_correct(network, images, labels, _print=False, origLabels=None):
    correct = 0.0
    false_pos = 0
    false_neg = 0
    for x in range(len(images)):
        if toBinary(network.predict(images[x])[0]) == labels[x][0]:
            correct += 1
        elif labels[x][0] == 0:
            false_pos += 1
        else:
            false_neg += 1
    if _print:
        print("Test run on", len(images), "instances.")
        print("Correct prediction on", correct, "instances.")
        print("Correct rate is", correct / len(images))
        print("False negatives:", false_neg)
        print("False positives:", false_pos)

    return correct / len(images)

def test_num_hidden_units(images, labels, tImages, tLabels, alpha, iters):
    units = [5,7,9,10]
    unitPlots = []
    
    for unit in units:

        costs = [0 for x in range(iters)]

        network = NeuralNetwork()
        avg_correct = 0.0
        for y in range(10):
            network = NeuralNetwork()
            network.add_layer(NeuronLayer("tanh", unit, 784))
            network.add_layer(NeuronLayer("sigmoid", 1, unit))
            print("Unit", unit, "y", y)
            for x in range(iters):
                network.backpropagate(images, labels, alpha)
                costs[x] += network.cost(images, labels)
                #print("Iteration", x+1, "of", iters, "| Cost is", costs[-1])
            correct = test_correct(network, tImages, tLabels, False)
            avg_correct += correct
        
        avg_correct *= .1

        print(unit, "units", avg_correct, "correct")

        costs = np.dot(.25, costs)
        unitPlot, = plt.plot(costs, label=str(unit) + " hidden units")
        unitPlots.append(unitPlot)
        #print("With", unit, "hidden units, our test error is",
            #test_correct(network, tImages, tLabels))

    plt.legend(unitPlots, list(map(lambda x: str(x) + " hidden units", units)))
    plt.xlabel("iteration number")
    plt.ylabel("cost")
    plt.savefig("nn_cost_units_comparison.jpg")
    plt.close()

def main():
    network = NeuralNetwork()
    network.add_layer(NeuronLayer("tanh", 9, 784))
    network.add_layer(NeuronLayer("sigmoid", 1, 9))
    #network.add_layer(NeuronLayer("sigmoid", 1, 784))
    alpha = 5
    iters = 300
    mndata = MNIST('./')
    images, labels = mndata.load_training()
    tImages, tLabels = mndata.load_testing()

    images = images[:6000]
    labels = labels[:6000]

    oldLabels = labels

    images = normalize_images(images)
    labels = normalize_labels(labels)
    tImages = normalize_images(tImages)
    tLabels = normalize_labels(tLabels)

    test_errors = []
    training_errors = []

    #test_alphas(images, labels, iters, tImages, tLabels)
    #test_num_hidden_units(images, labels, tImages, tLabels, alpha, iters)
    #test_activations(images, labels, iters, alpha, tImages, tLabels)
    #test_layers(images, labels, iters, alpha, tImages, tLabels)

    training_errors.append(1 - test_correct(network, images, labels))
    test_errors.append(1 - test_correct(network, tImages, tLabels))

    costs = [network.cost(images, labels)]
    for x in range(iters):
        print("Iteration", x, "of", iters, "Cost is", network.cost(images, labels))
        network.backpropagate(images, labels, alpha)
        training_errors.append(1 - test_correct(network, images, labels))
        test_errors.append(1 - test_correct(network, tImages, tLabels))
        print("Test error is", test_errors[-1])
        costs.append(network.cost(images, labels))
    
    trainPlot, = plt.plot(training_errors, label="Training data error rate")
    testPlot, = plt.plot(test_errors, label="Test data error rate")

    plt.legend([trainPlot, testPlot], ["Training data error rate", "Test data error rate"])
    plt.xlabel("iteration number")
    plt.ylabel("cost")
    plt.savefig("nn_errors_comparison.jpg")
    plt.close()
             
    plt.plot(costs)
    plt.savefig("neural_network_cost.jpg")
    plt.close()
    
    test_correct(network, tImages, tLabels, True)
    exampleIndices = [2, 1, 15, 9, 8, 7, 6, 18, 23]
    for x in exampleIndices:
        print(x, toBinary(network.predict(tImages[x])[0]))
    
    #test_num_hidden_units(images, labels, tImages, tLabels, alpha, iters)
    #test_alphas(images, labels, iters)

if __name__ == "__main__":
    main()
