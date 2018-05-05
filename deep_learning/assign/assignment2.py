from mnist import MNIST
import pdb
import sys
import math
import numpy as np
import matplotlib.pyplot as plt 

def sigmoid(z):
    if z < 0:
        return 1 - sigmoid(-z)
    return 1.0 / (1 + math.exp(-z))

def error(h, y):
    return y * math.log(h) + (1 - y) * math.log(1-h)

def predict(weights, instance):
    z = np.dot(weights, [1] + instance)
    val = sigmoid(z)
    if val >= .5:
        return 1
    return 0

def labelToBinary(label, mask):
    if label == mask:
        return 1
    return 0

def binaryToEnglish(x):
    if x == 1:
        return "Yes"
    else:
        return "No"

def genPlot(array):
    plt.plot(array)
    plt.xlabel("iteration number")
    plt.ylabel("cost")
    plt.savefig("logistic_cost.jpg")
    plt.close()

def gradient_descent(instances, labels, alpha, iters, trueValue):
    weights = [0 for x in range(0, len(instances[0]) + 1)]
    costs = []

    # pad with a column of 1s
    instances = np.insert(instances, 0, 1, axis=1)
    labels = list(map(lambda x: labelToBinary(x, trueValue), labels))

    N = len(instances)
    for i in range(iters):
        cost = 0
        
        predictions = list(map(sigmoid, np.matmul(instances, weights)))
        diffs = np.subtract(predictions, labels)

        changes = np.dot(np.transpose(diffs), instances)

        cost -= sum(map(error, predictions, labels))

        weights -= np.dot(alpha, changes) 
        costs.append(cost)
        print("Cost of iteration", i + 1, "of", iters, "is", cost / N)

    genPlot(costs)
    return weights

alpha = (len(sys.argv) > 1 and float(sys.argv[1])) or 0.000005
iter = (len(sys.argv) > 2 and int(sys.argv[2])) or 3000

mndata = MNIST('./')
images, labels = mndata.load_training()
images = images[:6000]
labels = labels[:6000]

for image in images:
    for x in range(len(image)):
        image[x] = image[x] / 255.0

tImages, tLabels = mndata.load_testing()

for image in tImages:
    for x in range(len(image)):
        image[x] = image[x] / 255.0

classifier = gradient_descent(images, labels, alpha, iter, 5)

correct = 0
for x in range(len(images)):
    binary = 0
    if predict(classifier, images[x]) == labelToBinary(labels[x], 5):
        correct += 1
print("Training error is:", (1 - (correct * 1.0 / len(images))))

correct = 0
for x in range(len(tImages)):
    binary = 0
    if predict(classifier, tImages[x]) == labelToBinary(tLabels[x], 5):
        correct += 1

print("Test error is:", (1 - (correct * 1.0 / len(tImages))))

for x in range(30):
    image = tImages[x]
    plt.clf()
    plt.imshow(np.reshape(image, (28, 28)), cmap='gray')
    plt.savefig("image_" + str(x) + ".jpg")
    plt.close()
    prediction = predict(classifier, image)
    correct = labelToBinary(tLabels[x], 5)
    print("Prediction for image", x, "is", prediction)
    print("Is the image a 5?", binaryToEnglish(correct))
    print("Is the prediction correct?", "Yes" if prediction == correct else "No")
    

