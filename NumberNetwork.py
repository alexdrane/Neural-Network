# test 4 - number recognition from large dataset
from NeuralNetwork import *
import numpy as np

#np.random.seed(10)

trainingInputs = np.array()
trainingOutputs = np.array()


myNetwork = NeuralNetwork([9,30,30,20,2])

n = 100000
t = 0.001

myNetwork.printSynapticWeights()
#myNetwork.train(trainingInputs[:4],trainingOutputs[:4],n)
n = myNetwork.trainByAccuracy(trainingInputs[:12],trainingOutputs[:12],t)
#myNetwork.train(trainingInputs,trainingOutputs,n)
print("\n"*20)
print("Trained",n,"times")
myNetwork.printSynapticWeights()

newInput = np.array(([1,1,1]))

print("\n"*2)
print("Expected outputs: ")
print(trainingOutputs[:12])
print("Actual outputs:")
print(myNetwork.get(trainingInputs[:12]))
print("Test")
print(myNetwork.test(trainingInputs,trainingOutputs))
myNetwork.write("DoubleXORnetwork.txt")
input()
