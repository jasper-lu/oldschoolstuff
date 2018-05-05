import numpy as np
import time
import pdb
import matplotlib.pyplot as plt 

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
