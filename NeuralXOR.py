# test 1 - xor test
from NeuralNetwork import *
import numpy as np

np.random.seed(10)

trainingInputs = np.array([[0,1,0],[1,0,0],[1,1,0],[0,0,0],[0,1,1],[1,0,1],[0,0,1],[1,1,1]])
trainingOutputs = np.array([[1,1,0,0,1,1,0,0]]).T


myNetwork = NeuralNetwork([3,20,10,1])

n = 100000
t = 0.01

myNetwork.printSynapticWeights()
#myNetwork.train(trainingInputs[:4],trainingOutputs[:4],n)
n = myNetwork.trainByAccuracy(trainingInputs[:4],trainingOutputs[:4],t)
#myNetwork.train(trainingInputs,trainingOutputs,n)
print("\n"*20)
print("Trained",n,"times")
myNetwork.printSynapticWeights()

newInput = np.array(([1,1,1]))

print("\n"*2)
print("Expected outputs: ")
print(trainingOutputs[:4])
print("Actual outputs:")
print(myNetwork.get(trainingInputs[:4]))
print("Test")
print(myNetwork.test(trainingInputs[4:],trainingOutputs[4:]))
input()
