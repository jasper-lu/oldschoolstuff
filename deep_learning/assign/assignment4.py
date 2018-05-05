import numpy as np
import time
import pdb
import matplotlib.pyplot as plt 
from joblib import Parallel, delayed
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

def wrapped_net_train(net, X, Y, iters, alpha, tX=[], tY=[], silent=False, errors=False, verbose=False):
    before = time.time()
    costs = net.train(X, Y, iters, alpha, errors, verbose, silent, tX, tY) 
    print("Training took", time.time() - before, "to run.")
    return (net, costs)

def test_alphas(images, labels, iters, tImages = [], tLabels = []):
    alphas = [.1, .5, 1, 2.5, 10]
    networks = [NeuralNet.build([784,5,5,1], "relu") for x in range(5)]

    rets = Parallel(3)(delayed(wrapped_net_train)(net, images, labels, iters, a, tX=tImages, tY=tLabels, silent=True) 
            for net, a in zip(networks, alphas))

    networks = [x[0] for x in rets]
    alphaCosts = [x[1] for x in rets]

    #alphaCosts = [net.train(images, labels, iters, a) for net, a in zip(networks, alphas)]
    alphaPlots = [plt.plot(x[0])[0] for x in alphaCosts] 

    for net, a in zip(networks, alphas):
        print("Final test correctness for alpha =", a, "is", net.test(tImages, tLabels))

    plt.legend(alphaPlots, list(map(lambda x: "alpha=" + str(x), alphas)))
    plt.xlabel("iteration number")
    plt.ylabel("cost")
    plt.savefig("nn_cost_alpha_comparison.jpg")
    plt.close()

def test_activations(images, labels, iters, alpha, tImages = [], tLabels = []):
    acts = ["relu", "sigmoid", "tanh"]
    networks = [NeuralNet.build([784,5,5,5,1],x) for x in acts]

    rets = Parallel(3)(delayed(wrapped_net_train)(net,images,labels,iters,alpha)
        for net in networks)
    
    #rets = [net.train(images, labels, iters, alpha) for net in networks]

    networks = [x[0] for x in rets]
    actCosts = [x[1] for x in rets]

    for net, act in zip(networks, acts):
        print("Final test correctness for activation", act, "is", net.test(tImages, tLabels))

    actPlots = [plt.plot(x[0])[0] for x in actCosts]

    plt.legend(actPlots, acts)
    plt.xlabel("iteration number")
    plt.ylabel("cost")
    plt.savefig("nn_cost_act_comparison.jpg")
    plt.close()

def test_layers(images, labels, iters, alpha, tImages = [], tLabels = []):
    networks = [NeuralNet.build([784,5,5,1],"relu"), NeuralNet.build([784,5,5,5,1], "relu"),
        NeuralNet.build([784,5,5,5,5,1], "relu")]

    rets = Parallel(3)(delayed(wrapped_net_train)(net,images,labels,iters,alpha,silent=True)
        for net in networks)

    networks = [x[0] for x in rets]
    layerCosts = [x[1] for x in rets]

    for i in range(3):
        print("Final test correctness for", i+2, "hidden layers is", networks[i].test(tImages, tLabels))

    layerPlots = [plt.plot(x[0])[0] for x in layerCosts]

    plt.legend(layerPlots, ["2 hidden layer", "3 hidden layers", "4 hidden layers"])
    plt.xlabel("iteration number")
    plt.ylabel("cost")
    plt.savefig("nn_layers.jpg")
    plt.close()

def test_num_hidden_units(images, labels, tImages, tLabels, alpha, iters):
    units = [5,7,9,10]

    networks = [NeuralNet.build([784,x,x,1],"relu") for x in units]
    rets = Parallel(3)(delayed(wrapped_net_train)(net,images,labels,iters,alpha,silent=True)
        for net in networks)

    networks = [x[0] for x in rets]
    unitCosts = [x[1] for x in rets]

    for unit, net in zip(units, networks):
        print("Final test correctness for", unit, "hidden units per layer is", net.test(tImages, tLabels))

    unitPlots = [plt.plot(x[0])[0] for x in unitCosts]

    plt.legend(unitPlots, [str(x) + " hidden units" for x in units])
    plt.xlabel("iteration number")
    plt.ylabel("cost")
    plt.savefig("nn_cost_units_comparison.jpg")
    plt.close()

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
    net = NeuralNet.build([784,10,10,1],"relu")
    #network.add_layer(NeuronLayer("sigmoid", 1, 784))
    alpha = 1
    iters = 500

    mndata = MNIST('./mnist/')
    images, labels = mndata.load_training()
    tImages, tLabels = mndata.load_testing()

    images = images[:6000]
    labels = labels[:6000]

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
        errors=True, tX=tImages, tY=tLabels, batchSize=600)
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
