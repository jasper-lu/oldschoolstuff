import numpy as np
import matplotlib.pyplot as plt 

FOOD_TRUCK_FILE = "food_truck_data"
HOUSING_PRICE_FILE = "housing_price_data"

def scale(instances):
    avgs = [0 for x in range(len(instances[0]))]
    for x in instances:
        for y in range(len(x)):
            avgs[y] = avgs[y] + x[y]
        
    avgs = [x / len(instances) for x in avgs]

    sdevs = [0 for x in range(len(instances[0]))]

    for instance in instances:
        for x in range(len(instance)):
            sdevs[x] += (instance[x] - avgs[x]) ** 2
        
    sdevs = [x / len(instances) for x in sdevs]
    sdevs = [x ** .5 for x in sdevs]

    new_instances = [[0 for x in range(len(instances[y]))] for y in range(len(instances))]

    for x in range(len(instances)):
        for y in range(len(instances[x])):
            new_instances[x][y] = (instances[x][y] - avgs[y]) / sdevs[y]
        
    print("avgs:", avgs)
    print("sdevs:", sdevs)
    return new_instances

# Only for 2-d plots
def plot_model(x_coords, y_coords, weights, file_name):
    plt.scatter(x_coords, y_coords)

    xs = []
    ys = []
    for x in range(0, int(max(x_coords)[0]* 11)):
        xs.append(x * .1)
        ys.append(np.dot(weights, [1, x *.1]))
    plt.plot(xs, ys)
    plt.savefig(file_name + "_model.jpg")
    plt.close()

def learn(file_name, alpha, iterations, scale_features=False, plot=False):
    data = open(file_name + ".txt")
    instances = []
    labels = []

    for line in data:
        arr = list(map(float, line.split(',')))
        a = arr[0:-1]
        b = arr[-1]
        instances.append(a)
        labels.append(b)

    if scale_features:
        instances = scale(instances)

    weights = gradient_descent(instances, labels, alpha, iterations, file_name)

    if plot:
        plot_model(instances, labels, weights, file_name)

    return weights

def gradient_descent(instances, labels, alpha, iters, file_name):
    weights = [0 for x in range(0, len(instances[0]) + 1)]
    errors = []
    N = len(instances)
    for i in range(iters):
        error = 0
        change = [0 for x in range(0, len(weights))]
        for x in range(0, len(instances)):
            instance = [1] + instances[x]
            prediction = np.dot(instance, weights)
            diff = prediction - labels[x]
            error = error + (diff) ** 2
            for y in range(0, len(weights)):
                change[y] += (1/N) * diff * instance[y]

        error = error * (1/ (2 * N))

        errors.append(error)

        for x in range(0, len(weights)):
            weights[x] = weights[x] - (alpha * change[x])

    plt.plot(errors)
    plt.xlabel("iteration number")
    plt.ylabel("cost")
    plt.savefig(file_name + "_error.jpg")
    plt.close()

    return weights

print(learn(FOOD_TRUCK_FILE, 0.0005, 100, False, True))
print(learn(HOUSING_PRICE_FILE, 0.001, 4000, True, False))
