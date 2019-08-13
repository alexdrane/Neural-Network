# test 3 - checked or striped classifier
from NeuralNetwork import *
import numpy as np
import json

#np.random.seed(10)

trainingInputs = np.array([[0,0,0,
                            1,1,1,
                            0,0,0],
                            [1,0,1,
                            1,0,1,
                            1,0,1],
                            [1,1,1,
                            0,0,0,
                            1,1,1],
                            [0,1,0,
                            0,1,0,
                            0,1,0],
                            [1,0,1,
                            0,1,0,
                            1,0,1],
                            [0,1,0,
                            1,0,1,
                            0,1,0],
                            [1,1,0,
                            1,1,0,
                            0,0,0],
                            [0,1,1,
                            0,1,1,
                            0,0,0],
                            [0,0,0,
                            1,1,0,
                            1,1,0],
                            [0,0,0,
                            0,1,1,
                            0,1,1],
                            [0,0,1,
                            0,0,1,
                            1,1,1],
                            [1,0,0,
                            1,0,0,
                            1,1,1],
                            [1,1,1,
                            0,0,1,
                            0,0,1],
                            [1,1,1,
                            1,0,0,
                            1,0,0]])
trainingOutputs = np.array([[0,0],
                            [0,1],
                            [0,0],
                            [0,1],
                            [1,0],
                            [1,0],
                            [1,1],
                            [1,1],
                            [1,1],
                            [1,1],
                            [1,1],
                            [1,1],
                            [1,1],
                            [1,1],])


myNetwork = NeuralNetwork([9,30,30,20,2])

n = 100000
t = 0.0001

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
net = {'layers' : []}
for layer in myNetwork.layers:
    net['layers'].append(layer.synapticWeights.tolist())
data = {'network' : net}
with open('JSONdata.txt', 'w') as outfile:
    json.dump(data, outfile)
input()
