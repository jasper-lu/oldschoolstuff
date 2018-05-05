from mnist import MNIST
import sys
import math
import numpy as np
import matplotlib.pyplot as plt 

def predict(weights, instance):
    z = np.dot(weights, instance)
    if z < 0:
        return 1 - 1.0 / (1 + math.exp(z))
    return 1.0 / (1 + math.exp(-z))

def labelToBinary(label, mask):
    if label == mask:
        return 1
    return 0

def genPlot(array):
    plt.plot(array)
    plt.xlabel("iteration number")
    plt.ylabel("cost")
    plt.savefig("logistic_cost.jpg")
    plt.close()

"""
def gradient_descent(instances, labels, alpha, iters, trueValue):
    weights = [0 for x in range(0, len(instances[0]) + 1)]
    costs = []

    instances = np.insert(instances, 0, 1, axis=1)

    N = len(instances)
    for i in range(iters):
        change = [0 for x in range(0, len(weights))]
        cost = 0
        for x in range(len(instances)):
            instance = [1] + instances[x]
            prediction = predict(weights, instance)
            label = labelToBinary(labels[x], trueValue)
            diff = prediction - label
            change = np.add(change, np.dot(diff, instance))  
            # vectorize this
            cost -= (label * math.log(prediction)) + ((1 - label) * math.log(1 - prediction))
            
        weights -= np.dot(alpha, change) 
        costs.append(cost / N)
        print("Cost of iteration", i + 1, "is", cost / N)

    genPlot(costs)
    return weights
"""

def gradient_descent(instances, labels, alpha, iters, trueValue):
    weights = [0 for x in range(0, len(instances[0]) + 1)]
    costs = []

    # pad with a column of 1s
    instances = np.insert(instances, 0, 1, axis=1)

    N = len(instances)
    for i in range(iters):
        change = [0 for x in range(0, len(weights))]
        cost = 0
        for x in range(len(instances)):
            instance = [1] + instances[x]
            prediction = predict(weights, instance)
            label = labelToBinary(labels[x], trueValue)
            diff = prediction - label
            change = np.add(change, np.dot(diff, instance))  
            # vectorize this
            cost -= (label * math.log(prediction)) + ((1 - label) * math.log(1 - prediction))
            
        weights -= np.dot(alpha, change) 
        costs.append(cost / N)
        print("Cost of iteration", i + 1, "is", cost / N)

    genPlot(costs)
    return weights

alpha = (len(sys.argv) > 1 and float(sys.argv[1])) or 0.000001
iter = (len(sys.argv) > 2 and int(sys.argv[2])) or 50

mndata = MNIST('./')
images, labels = mndata.load_training()

for image in images:
    for x in range(len(image)):
        image[x] = image[x] / 255.0

classifier = gradient_descent(images, labels, alpha, iter, 5)

tImages, tLabels = mndata.load_testing()
for image in tImages:
    for x in range(len(image)):
        image[x] = image[x] / 255.0

correct = 0
for x in range(len(tImages)):
    if predict(classifier, tImages[x]) == labelToBinary(tLabels[x]):
        correct += 1

print("Test error is:", (correct * 1.0 / len(tImages)))

