from mnist import MNIST

N = [0 for x in range(10)]
Ntest = [0 for x in range(10)]

mndata = MNIST('./')
images, labels = mndata.load_training()

timages, tlabels = mndata.load_testing()

for label in labels:
    N[label] += 1

for label in tlabels:
    Ntest[label] += 1

print(N)
print(Ntest)
