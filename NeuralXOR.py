# test 1 - xor test
from NeuralNetwork import *
import numpy as np

np.random.seed(10)

trainingInputs = np.array([[0,1,0],[1,0,0],[1,1,0],[0,0,0],[0,1,1],[1,0,1],[0,0,1],[1,1,1]])
trainingOutputs = np.array([[1,1,0,0,1,1,0,0]]).T

layer1 = neuron_layer(10,3)
layer2 = neuron_layer(1,10)

myNetwork = NeuralNetwork(layer1,layer2)

n = 10000

myNetwork.printSynapticWeights()
myNetwork.train(trainingInputs,trainingOutputs,n)
print("\n"*20)
print("Trained",n,"times")
myNetwork.printSynapticWeights()

newInput = np.array(([1,1,1]))

print("\n"*2)
print("Expected outputs: ")
print(trainingOutputs)
print("Actual outputs:")
print(myNetwork.think(trainingInputs)[1])
#input()
