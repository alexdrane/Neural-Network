from NeuralNetwork import *
import numpy as np
import json

myNetwork = NeuralNetwork([9,30])

myNetwork.loadFromJson('JSONdata.txt')


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


print("Expected outputs: ")
print(trainingOutputs[:12])
print("Actual outputs:")
print(myNetwork.get(trainingInputs[:12]))
input()
