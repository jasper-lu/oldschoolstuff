import matplotlib.pyplot as plt 
import time
import pdb
import numpy as np
from sklearn.utils import shuffle

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

# let's assume instances are now passed in as N (instances) x M (features) matrices
activations = {}
activations["relu"] = np.vectorize(lambda x: max(x, 0))
activations["sigmoid"] = np.vectorize(sigmoid)
activations["tanh"] = np.vectorize(lambda x: np.tanh(x))

relu = np.vectorize(lambda x: max(x, 0))

deriv_activations = {}
deriv_activations["relu"] = np.vectorize(lambda x: 1 if x > 0 else 0)
deriv_activations["sigmoid"] = np.vectorize(lambda x: sigmoid(x) * (1 - sigmoid(x)))
deriv_activations["tanh"] = np.vectorize(lambda x: 1 - np.tanh(x)**2)

class Layer:
    def __init__(self, activation, m, n):
        self.activation = activation
        self.activation_func = activations[activation]
        self.activation_deriv_func = deriv_activations[activation]
        self.W = np.random.normal(0, 0.1, (m, n))
        self.b = np.zeros((m, 1))
        self.width = m
        self.connections = n

    def z(self, x): # x should be passed in here as a column vector
        return np.dot(self.W, x) + self.b

    def createDropMask(self, dropProb):
        if dropProb == 0:
            return None
        return np.reshape([np.random.binomial(1, dropProb, size=self.width)], (self.width, 1))

    def calculate(self, x, dropMask=None):
        Z = self.z(x)
        ret = activations[self.activation](Z)
        if not dropMask is None:
            ret = np.multiply(ret, dropMask)
        return ret

    def derivative(self, x):
        return deriv_activations[self.activation](self.z(x))

class NeuralNet:
    def __init__(self, n):
        self.network = []
        self.input_features = n

    @staticmethod
    def build(architecture, activation):
        net = NeuralNet(architecture[0])
        for x in range(1, len(architecture) - 1):
            net.add_layer(activation, architecture[x])
        net.add_layer("sigmoid", architecture[-1])
        return net

    def cost(self, X, Y, regParam = 0):
        a = X
        n, m = Y.shape
        for layer in self.network:
            a = layer.calculate(a)
        left = np.vectorize(lambda x: np.log(x))
        right = np.vectorize(lambda x: np.log(1-x))
        C = (-1./m) * (np.multiply(Y, left(a)) + np.multiply(1-Y, right(a))).dot(np.ones((m,1)))
        if regParam > 0:
            C += (regParam / (2 * m)) * self.sumSquareWeights()
        return C[0][0] 

    def sumSquareWeights(self):
        ret = 0
        for layer in self.network:
            for row in layer.W:
                for col in row:
                    ret += col ** 2
        return ret

    def temp(self):
        print("Hi")
        return 23
    
    def quadratic_cost(self, X, Y):
        a = X
        n, m = Y.shape
        for layer in self.network:
            a = layer.calculate(a)
        square = np.vectorize(lambda x: x ** 2)
        C = (1./m) * square(a-Y).dot(np.ones((m,1)))
        return C[0][0]
       
    def test(self, X, Y):
        toBinary = np.vectorize(lambda x: 1 if x >= .5 else 0)
        compare = np.vectorize(lambda a, b: 1 if a==b else 0)
        predictions = toBinary(self.predict(X))
        n = Y.shape[1]
        #pdb.set_trace()
        return compare(predictions, Y).dot(np.ones((n,1)))[0][0] * 1.0 / n

    def createDropMask(self, dropProb):
        dropW = [layer.createDropMask(dropProb) for layer in self.network[:-1]]
        dropW.append(None)
        return dropW

    def createBatches(self, X, Y, batchSize):
        if batchSize == 0:
            return [X], [Y]
        XX, YY = shuffle(X.T, Y.T)
        batches = []
        labels = []
        for i in range(0, X.shape[1], batchSize):
            batches.append(XX[i:i+batchSize].T)
            labels.append(YY[i:i+batchSize].T)
        
        return batches, labels

    def backProp(self, X, Y, dropProb):
        dropW = self.createDropMask(dropProb)
        dW = [np.empty(layer.W.shape) for layer in self.network]
        db = [np.empty(layer.b.shape) for layer in self.network]

        m, n = X.shape

        before = time.time()
        a = [X]

        A = X
        Z = []
        for layer, dropL in zip(self.network, dropW):
            a.append(layer.calculate(a[-1], dropL))
        """
        for w, b, act, dropL in zip(weights, biases, self.activations(), dropW):
            z = np.dot(w, A) + b
            Z.append(z)
            A = act(z)
            a.append(A)
        """

        before = time.time()

        delta = [np.empty((layer.width, len(Y))) for layer in self.network]
        delta[-1] = a[-1] - Y # this is an n-dim array

        #pdb.set_trace()
        dW[-1] = np.matmul(delta[-1], a[-2].T)
        db[-1] = delta[-1].dot(np.ones((n, 1)))

        for i in range(1, len(self.network)):
            l = -1 - i
            delta[l] = np.multiply(self.network[l+1].W.T.dot(delta[l+1]), self.network[l].derivative(a[l-1]))
            dW[l] = np.matmul(delta[l], a[l-1].T)
            db[l] = delta[l].dot(np.ones((n, 1))) 
                #pdb.set_trace()
            if not dropW[l] is None:
                dW[l] = np.multiply(dropW[l], dW[l]) / dropProb
                db[l] = np.multiply(dropW[l], db[l]) / dropProb

        return dW, db

    def train(self, X, Y, iters, alpha, regParam=0, errors=False, verbose=False, 
        silent=False, dropProb=0, batchSize=0, momentum=0, tX = [], tY = [], printGap=10):

        vdW = [np.zeros(layer.W.shape) for layer in self.network]
        vdb = [np.zeros(layer.b.shape) for layer in self.network]

        m, n = X.shape

        c = alpha / n # multiplication constant

        # a[0] should now be the inputs
        # a[-1] should now be the outputs
        # if we have one hidden layer, this is an array 3 long
        costs = [0 for x in range(iters+1)]
        costs[0] = self.cost(X,Y)

        ret = [costs]
        if errors:
            ret.append([0 for x in range(iters + 1)])
            ret.append([0 for x in range(iters + 1)])
            ret[1][0] = self.test(X,Y)
            ret[2][0] = self.test(tX,tY)

        weights = self.weights()
        biases = self.biases()
        # THIS IS THE BACKPROPAGATION CODE

        for iter in range(iters):
            beforeIter = time.time()
            if not silent:
                #pdb.set_trace()
                if iter % printGap == 0:
                    print("Iteration", iter, "of", iters)
                    print("Cost is", costs[iter])
            if errors or verbose:
                trainingCorrect = self.test(X,Y)
                testCorrect = self.test(tX,tY)
                #pdb.set_trace()
                if verbose and iter % printGap == 0:
                    print("Training correct is:", trainingCorrect)
                    print("Test correct is:", testCorrect)
                if errors:
                    ret[1][iter+1]=trainingCorrect
                    ret[2][iter+1]=testCorrect

            batches, labels = self.createBatches(X, Y, batchSize)
            #print("Took", time.time() - beforeIter, "to create batches and get through blocks")

            batchNum = 0
            for x, y in zip(batches, labels):
                before = time.time()
                tempDW, tempDb = self.backProp(x, y, dropProb)
                #print("Took", time.time() - before, "to backprop")
                before = time.time()
                #print(batchNum)
                batchNum+=1
                #pdb.set_trace()
                # Account for momentums. dW = tempDW if momentum=0
                vdW = np.multiply(vdW, momentum) - tempDW
                vdb = np.multiply(vdb, momentum) - tempDb
                momentumTime = time.time()
                #print("Took", momentumTime-before, "to acccount for momentum")

                for l in range(len(self.network)):
                    self.network[l].W = self.network[l].W * (1 - regParam*c) + c * vdW[l]
                    self.network[l].b += c * vdb[l]

                #print("Took", time.time() - momentumTime, "to update weights (end of update for minibatch)")
            
            #print("Took", time.time()-beforeIter, "to update")
            beforeCost = time.time()

            costs[iter+1]=self.cost(X,Y,regParam)
            #print("Took", time.time() - beforeCost, "to get cost")

        if not silent:
            print("Iteration", iters, "of", iters)
            print("Cost is", costs[iters])
        if errors or verbose:
            trainingCorrect = self.test(X,Y)
            testCorrect = self.test(tX, tY)
            if verbose:
                print("Training correct is:", trainingCorrect)
                print("Test correct is:", testCorrect)
            if errors:
                ret[1][iters] = trainingCorrect
                ret[2][iters] = testCorrect

        return ret

    def weights(self):
        return [layer.W for layer in self.network]

    def biases(self):
        return [layer.b for layer in self.network]

    def activations(self):
        return [layer.activation_func for layer in self.network]

    def activation_derivs(self):
        return [layer.activation_deriv_func for layer in self.network]

    def add_layer(self, activation, m):
        connections =  self.input_features if len(self.network) == 0 else self.network[-1].width
        self.network.append(Layer(activation,m,connections))

    def predict(self, x):
        a = x
        for layer in self.network:
            a = layer.calculate(a)
        return a


"""
net = NeuralNet()
net.add_layer("relu", 10)
net.add_layer("sigmoid", 1)
net.network[0].W = np.array([[.5,.4],[.4,.5]])
net.network[1].W = np.array([[.3,.2]])
instances = np.array([[0,0,1,1],[0,1,0,1]])
xors = np.array([[0,1,1,0]])
#costs = net.train(instances, xors, 1000, 3)
costs = net.train(np.array([[0, 0, 1, 1],[0, 1, 0, 1]]), np.array([[0,1,1,0]]), 500, 1)
plt.plot(costs)
plt.savefig("neuralcost.jpg")

#pdb.set_trace()
print(net.predict(instances), xors)
print(net.predict(np.array([[1],[1]])))
"""
