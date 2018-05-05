import pdb
import matplotlib.pyplot as plt 
import numpy as np

toBinary = np.vectorize(lambda x: 1 if x >= 0.5 else 0)

def test_correct(network, images, labels, _print=False, origLabels=None):
    correct = 0.0
    false_pos = 0
    false_neg = 0
    toBinary = np.vectorize(lambda x: 1 if x >= 0.5 else 0)
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

def plot_decision_boundary(model, X, Y):
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = np.ravel(toBinary(model(np.c_[xx.ravel(), yy.ravel()].T)))
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=Y, cmap=plt.cm.Spectral)
    plt.show()

def plot_cost(cost):
    plt.plot(cost)
    plt.xlabel("Iteration number")
    plt.ylabel("Cross-entropy cost")
    plt.title("Cost over time")
    plt.show()
    plt.close()

