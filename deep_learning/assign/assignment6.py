import numpy as np
import time
import pdb
import matplotlib.pyplot as plt 
from relu_net import NeuralNet
from mnist import MNIST

toBinary = np.vectorize(lambda x: 1 if x >= 0.5 else 0)

def normalize_images(images):
    arr = []
    for x in range(len(images)):
        arr.append(np.array(images[x]).T.dot(1.0/255))
    return np.array(arr).T

def normalize_labels(labels):
    newLabels = []
    for label in labels:
        if label == 5:
            newLabels.append([1])
        else:
            newLabels.append([0])
    return np.array(newLabels).T

def test_correct(network, images, labels, _print=False, origLabels=None):
    correct = 0.0
    false_pos = 0
    false_neg = 0
    predictions = toBinary(network.predict(images))

    for i in range(predictions.shape[1]):
        if predictions[0][i] == labels[0][i]:
            correct += 1
        elif labels[0][i] == 0:
            false_pos += 1
        else:
            false_neg += 1
    if _print:
        print("Test run on", len(images), "instances.")
        print("Correct prediction on", correct, "instances.")
        print("Correct rate is", correct / images.shape[1])
        print("False negatives:", false_neg)
        print("False positives:", false_pos)

    return correct / images.shape[1]

def timeFunc(f):
    before = time.time()
    ret = f()
    print("Took", time.time() - before, "to run.")
    return ret

def main():
    """
    net = NeuralNet(784)
    net.add_layer("relu", 5)
    net.add_layer("relu", 5)
    net.add_layer("sigmoid", 1)
    """
    net = NeuralNet.build([784,10,10,1],"sigmoid")
    #network.add_layer(NeuronLayer("sigmoid", 1, 784))
    alpha = 1
    iters = 500

    mndata = MNIST('./mnist/')
    images, labels = mndata.load_training()
    tImages, tLabels = mndata.load_testing()

    images = images[:600]
    labels = labels[:600]

    oldLabels = labels

    images = normalize_images(images)
    labels = normalize_labels(labels)
    tImages = normalize_images(tImages)
    tLabels = normalize_labels(tLabels)

    #test_alphas(images, labels, iters, tImages, tLabels)
    #test_num_hidden_units(images, labels, tImages, tLabels, alpha, iters)
    #test_activations(images, labels, iters, alpha, tImages, tLabels)
    #test_layers(images, labels, iters, alpha, tImages, tLabels)

    costs, training, test = net.train(images, labels, iters, alpha, silent=False, verbose=True,
        errors=False, tX=tImages, tY=tLabels, batchSize=1)
    training = [1 - x for x in training]
    test = [1 - x for x in test]
    #test_correct(net, tImages, tLabels, True)

    trainPlot = plt.plot(training)[0]
    testPlot = plt.plot(test)[0]
    plt.legend([trainPlot, testPlot], ["Training data error rate", "Test data error rate"])
    plt.xlabel("iteration number")
    plt.ylabel("cost")
    plt.savefig("nn_errors_comparison.jpg")
    plt.close()

    plt.plot(costs)
    plt.xlabel("iteration number")
    plt.ylabel("cost")
    plt.savefig("neural_network_cost.jpg")
    plt.close()

    test_correct(net,tImages,tLabels,True)
    
    """
    pdb.set_trace()
    test_correct(network, tImages, tLabels, True)
    exampleIndices = [2, 1, 15, 9, 8, 7, 6, 18, 23]
    for x in exampleIndices:
        print(x, network.predict(tImages[x])[0], True)
    """
    
    #test_num_hidden_units(images, labels, tImages, tLabels, alpha, iters)
    #test_alphas(images, labels, iters)

if __name__ == "__main__":
    main()
