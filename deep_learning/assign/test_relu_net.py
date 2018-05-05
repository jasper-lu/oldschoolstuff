from relu_net import Layer, NeuralNet
import unittest
import numpy as np

class TestNeuralNet(unittest.TestCase):
    def test_runs(self):
        net = NeuralNet(2)
        net.add_layer("relu", 4)

    def test_predict(self):
        net = NeuralNet(2)
        net.add_layer("relu", 2)
        net.add_layer("relu", 2)
        net.add_layer("relu", 1)

        net.network[0].W = np.ones((2,2))
        net.network[1].W = np.ones((2,2))
        net.network[2].W = np.ones((1,2))
        self.assertEqual(net.predict(np.array([[1,1]]).T)[0][0], 8)

    def test_backprop(self):
        net = NeuralNet(2)
        net.add_layer("relu", 2)
        net.add_layer("relu", 3)

        net.network[0].W = np.ones((2,2))
        net.network[1].W = np.ones((2,2))

        self.assertEqual(net.backpropagate(np.array([[1,2],[1,2]]),np.array([[3,3],[3,3]])), [])

class TestLayer(unittest.TestCase):
    def test_init(self):
        layer = Layer("relu", 3, 4)
        self.assertEqual(layer.b.shape, (3,1))
        self.assertEqual(layer.W.shape, (3,4))

    def test_calc(self):
        layer = Layer("relu", 3, 4)

        example = np.array([[0,0,0,0]]).T
        output = np.array([[100,100,100]]).T

        self.assertEqual(example.shape, (4,1))
        self.assertEqual(layer.z(example).shape, (3, 1))
        layer.b += np.array([[100,100,100]]).T

        np.testing.assert_array_equal(layer.calculate(example), output)

        np.testing.assert_array_equal(layer.derivative(example), output * .01)
        
        layer2 = Layer("relu", 2, 3)
        self.assertEqual(layer2.calculate(layer.calculate(example)).shape, (2, 1))

if __name__ == '__main__':
    unittest.main()
