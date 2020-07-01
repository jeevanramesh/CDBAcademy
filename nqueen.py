import random
import numpy as np
from numpy.random import choice
import pandas as pd

class NQueen():
    
    mutationRate = 0.2
    totalPopulation = 100
    crossOver = 0.5
    nQueen = 8
    loop_breaker = 1000
    seq_list = [x for x in range(nQueen)] 
    populationData = []
    populationMatrix = []
    fitnessData = []
    secure_random = random.SystemRandom()
    populationData=[]
    fitnessData = []
    probabilityDist = []

    def error_diagonal(self, mat):
        if (mat.shape[0] < 2 or mat.shape[1] < 2):
            # print("LESS than 2x2.")
            return 0
        num_error = mat.diagonal().sum() - 1
        return num_error
    
    def diagonal_error(self, matrix):
        total_num_error = 0
        temp = np.zeros(shape=(self.nQueen+2, self.nQueen+2))
        temp[1:self.nQueen+1, 1:self.nQueen+1] = matrix
        row_indices, col_indices = np.where(matrix == 1)
        row_indices = row_indices + 1
        col_indices = col_indices + 1
        total = 0
        for element_idx in range(self.nQueen):
            x = row_indices[element_idx]
            y = col_indices[element_idx]
            mat_bottom_right = temp[x:, y:]
            total = total + self.error_diagonal(mat_bottom_right)
            mat_bottom_left = temp[x:, y:0:-1]
            total = total + self.error_diagonal(mat_bottom_left)
            mat_top_right = temp[x:0:-1, y:]
            total = total + self.error_diagonal(mat_top_right)
            mat_top_left = temp[x:0:-1, y:0:-1]
            total = total + self.error_diagonal(mat_top_left)
        total_num_error = total_num_error + total / 2
        return total_num_error

    def fitness(self, randomData):
        matrix = np.uint8(np.zeros(shape=(8, 8)))
        i = 0
        for j in randomData:
            matrix[i][j] = 1
            i += 1
        column_fitness_error= len(randomData)-len(set(randomData))
        diagonal_fitness_error= self.diagonal_error(matrix)
        return (diagonal_fitness_error+column_fitness_error)

    def initialPopulation(self):
        for outloop in range(self.totalPopulation):
            randomData = []
            while True:  
                fitnessScore = 0
                for inloop in range(self.nQueen):
                    selectedData = self.secure_random.choice(self.seq_list)
                    randomData.append(selectedData)
                if len(set(randomData)) == self.nQueen:
                    break
                else:
                    randomData = []
            matrix = np.zeros(shape=(8, 8))
            i = 0
            for j in randomData:
                matrix[i][j] = 1
                i += 1
            self.populationData.append(randomData)
            fitnessScore=self.fitness(randomData)
            self.fitnessData.append(fitnessScore)
        probabilityDist = []
        for outloop in range(self.totalPopulation):
            probabilityDist.append(1/(self.fitnessData[outloop]+1))
        probDataFrame = pd.DataFrame({'String':self.populationData, 'FitnessScore':self.fitnessData,'Probability':probabilityDist})
        probDataFrame = probDataFrame.sort_values(['Probability'],ascending=False)
        probDataFrame = probDataFrame.reset_index(drop=True)
        return probDataFrame

    def crossoverMutation(self, probDataFrame):
        generate=[]
        for i in range(len(probDataFrame)):
            for j in range(int(probDataFrame.iloc[i][2]*100)):
                generate.append(probDataFrame.iloc[i][0])

        self.populationData=[]
        self.fitnessData = []
        probabilityDist = []
        for outloop in range(self.totalPopulation):
            while True:
                p1=self.secure_random.sample(generate,1)[0][0:int(self.crossOver*self.nQueen)]
                p2=self.secure_random.sample(generate,1)[0][int(self.crossOver*self.nQueen):]
                child=p1+p2
                if len(set(child)) == self.nQueen:
                    break
            for i in range(self.nQueen):
                m=np.random.random()
                if m < self.mutationRate:
                    swap=self.secure_random.choice(self.seq_list)
                    swapdata=child[swap]
                    child[swap]=child[i]
                    child[i]=swapdata
            self.populationData.append(child)
            fit = self.fitness(child)
            self.fitnessData.append(fit)
            if fit == 0:
                self.loop_breaker = 0
            probabilityDist.append(1/(self.fitnessData[outloop]+1))
        probDataFrame = pd.DataFrame({'String':self.populationData, 'FitnessScore':self.fitnessData,'Probability':probabilityDist})
        probDataFrame = probDataFrame.sort_values(['Probability'],ascending=False)
        return probDataFrame
    
    def mainFunc(self):
        probDataFrame=self.initialPopulation()
        while self.loop_breaker > 0:
            probDataFrame=self.crossoverMutation(probDataFrame)
            self.loop_breaker -= 1
        print(probDataFrame)

nqueen=NQueen()
nqueen.mainFunc()
