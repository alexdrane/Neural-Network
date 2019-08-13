#add to dataset

# 10x10 grid
from NeuralNetwork import *
import numpy as np
import pygame
from pygame import *
import json, sys
from faceNetwork import *

WIDTH = 600
HEIGHT  = 600

try:
    with open("dataset.txt") as json_file:
        data = json.load(json_file)
except:
    data = {'datas': []}


pygame.init()
DISPLAY = pygame.display.set_mode((WIDTH,HEIGHT))
myNetwork = NeuralNetwork([100,100,50,30,1])
myNetwork.loadFromJson("faceNet.txt")

grid = {'data': [], 'val' : 0}
for i in range(10):
    grid['data'].append([0,0,0,0,0,0,0,0,0,0])

myfont = pygame.font.SysFont('Comic Sans MS', 40)

trainingData = []
trainingOutputs = []
for d in data['datas']:
    #print(data)
    trainingOutputs.append([d['val']])
    n = []
    for row in d['data']:
        for v in row:
            n.append(v)
    trainingData.append(n)

trainingData = np.array(trainingData)
trainingOutputs = np.array(trainingOutputs)

while True:
    DISPLAY.fill(0)
    myNetwork.trainByAccuracy(trainingData,trainingOutputs,10)
    for event in pygame.event.get():
        if event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                net = {'layers' : []}
                for layer in myNetwork.layers:
                    net['layers'].append(layer.synapticWeights.tolist())
                nt = {'network' : net}
                with open('faceNet.txt', 'w') as outfile:
                    json.dump(nt, outfile)
                with open('dataset.txt', 'w') as outfile:
                    json.dump(data, outfile)
                pygame.quit()
                run()
                sys.exit()
            if event.key == K_SPACE:
                if grid['val'] == 0:
                    grid['val'] = 1
                else:
                    grid['val'] = 0
            if event.key == K_s:
                data['datas'].append(grid)
            if event.key == K_c:
                grid = {'data': [], 'val' : 0}
                for i in range(10):
                    grid['data'].append([0,0,0,0,0,0,0,0,0,0])
        if event.type == MOUSEBUTTONDOWN:
            mx,my = pygame.mouse.get_pos()
            sx = int(mx/(WIDTH/10))
            sy = int(my/(HEIGHT/10))
            if event.button == 1:
                grid['data'][sx][sy] = 1
            if event.button == 3:
                grid['data'][sx][sy] = 0


    for i in range(len(grid['data'])):
        for j in range(10):
            v = grid['data'][i][j]*255
            pygame.draw.rect(DISPLAY,(v,v,v),(int(i*WIDTH/10),int(j*HEIGHT/10),int(WIDTH/10),int(HEIGHT/10)))
    textsurface = myfont.render(str(grid['val']), False, (255, 0, 0))
    DISPLAY.blit(textsurface,(0,0))
    g = []
    for row in grid['data']:
        for v in row:
            g.append(v)
    g = np.array(g)
    textsurface = myfont.render(str(myNetwork.get(g)), False, (0,255, 0))
    DISPLAY.blit(textsurface,(400,0))

    pygame.display.update()
