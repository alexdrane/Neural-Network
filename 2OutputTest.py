# test 2 - xor test double output
from NeuralNetwork import *
import numpy as np

#np.random.seed(10)

trainingInputs = np.array([[0,1,0,0],[1,0,0,0],[1,1,0,0],[0,0,0,0],
[0,1,1,0],[1,0,1,0],[0,0,1,0],[1,1,1,0],
[0,1,0,1],[1,0,0,1],[1,1,0,1],[0,0,0,1],
[0,1,1,1],[1,0,1,1],[0,0,1,1],[1,1,1,1]])
trainingOutputs = np.array([[1,0],[1,0],[0,0],[0,0],[1,0],[1,0],[0,0],[0,0],
[1,1],[1,1],[0,1],[0,1],[1,1],[1,1],[0,1],[0,1]])


myNetwork = NeuralNetwork([4,20,20,2])

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
print(myNetwork.test(trainingInputs[12:],trainingOutputs[12:]))
myNetwork.write("DoubleXORnetwork.txt")
input()
