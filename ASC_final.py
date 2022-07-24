#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on 07 mar. 2020

@author: FranDesktop
@title: 'ASC_Multiobjective_Aggregation'
'''

import os
import math
import random
import matplotlib.pyplot as plt

class ASC_Multiobjective_Aggregation(object):
    
    def __init__(self, numGenerations, numIndividuals, individualDimension, neighborhoodPercentage, mutationPercentage, problem, absPathFile):
                
        self.__problem = problem
        self.__plot = True
        self.__writeResults = False
        self.__absPathFile = ""
        self.__absPathFile = absPathFile
        
        """ Init Constraints: """
        self.__numGenerations = numGenerations
        self.__currentGeneration = 1
        self.__numIndividuals = numIndividuals
        self.__numEvaluations = numGenerations*numIndividuals
        self.__neighborhoodPercentage = neighborhoodPercentage
        
        if(problem == "zdt3"):
            self.__lowerLimit = 0
            self.__upperLimit = 1
        elif(problem == "cf6"):
            self.__lowerLimit_x1 = 0
            self.__lowerLimit_xn = -2
            self.__upperLimit_x1 = 1
            self.__upperLimit_xn = 2
        
        if(problem == "zdt3"):
            if(individualDimension != 30):
                raise Exception("n must be 30 for zdt3")
            else:
                self.__individualDimension = individualDimension
        elif(problem == "cf6"):
            if(individualDimension == 4 or individualDimension == 16):
                self.__individualDimension = individualDimension
            else:
                raise Exception("n must be 4 or 16 for cf6")
        
        self.__numNearestNeighbors = round(numIndividuals*neighborhoodPercentage)
        self.__mutationPercentage = mutationPercentage
        self.__lambdaVectors = list()
        self.__lls_k_nearest_Neighbors = dict()
        """ End Constraints """
        
        self.__currentPopulation = list()
        self.__subProblems = dict()
        self.__z_current = [None for _ in range(0, 2)]
        self.__best_f1_value = None
        self.__best_f2_value = None
        
    def find_Pareto_Front(self):
        
        """ Inicializo vectores lambda que representa cada subproblema. """
        self.__lambdaVectors = self.initializeLambda()
        
        """ para cada lambda, calculo sus k-lambda's mas cercanos con sus respectivas distancias. """
        self.__lls_k_nearest_Neighbors = self.compute_lls_k_nearest_Neighbors()
        
        """ Inicializo poblacion de individuos, uno para cada subproblema. """
        self.__currentPopulation = self.initializePopulation()
        
        """ Asocio a cada individuo generado un subproblema. """
        self.association()
                
        """ Evaluo población inicial. """
        self.compute_fitness(input_dicc=self.__subProblems)
        
        """ Calculo referencia 'Z' para la población inicial. """
        self.compute_z_Reference()
        
        """ Creamos el archivo con los resultados y escribimos para la 1ª generacion """
        if(self.__writeResults):
            self.writeGenerationResults()
        
        """ Bucle de generaciones, descuento una unidad pues la anterior era la primera gen. """
        for _ in range(0, self.__numGenerations-1):
            
            self.__currentGeneration += 1
            
            """ Reproduccion: genero N hijos. Poblacion = 2*N """
            """ Asocio cada hijo al subproblema a partir del cual tome el individuo original y 3 vecinos aleatorios para generarlo. """
            childrens = self.reproduction()

            """ Evaluo a los hijos generados. """
            self.compute_fitness(input_dicc=childrens)
            
            """ Nos quedamos con los menores 'Z's de entre padres e hijos. """
            self.compute_z_Reference()
            
            """ Actualizamos el individuo de cada subproblema y cooperamos. """
            self.updateNeighborProblems(childrens)
            
            """ Llegado a este instante, he actualizado la vecindad y, por ende, la población: Guardamos el frente """            
            if(self.__writeResults):
                self.writeGenerationResults()
        
        """ Devuelvo el 'frente' encontrado. """
        if(self.__plot):
            self.plotFrontFound()
            
    def writeGenerationResults(self):
        f = open(self.__absPathFile,"a+")
        for k, v in self.__subProblems.items():
            f1 = v['fitness']['f1']
            f2 = v['fitness']['f2']
            f.write(str(f1)+"    "+str(f2)+"    "+"0\n")
        f.close()
        
    def plotFrontFound(self):
        
        bestParetoFront = self.getBestParetoFront()
        f1_bestParetoFront, f2_bestParetoFront = zip(*bestParetoFront)
        
        #plt.xlim(0, 1)
        #plt.ylim(-0.85, 1.25)

        plt.scatter(f1_bestParetoFront,f2_bestParetoFront,s=1,color='black')

        solutions = [(self.__subProblems[lambdaVector]["fitness"]["f1"], self.__subProblems[lambdaVector]["fitness"]["f2"]) for lambdaVector in self.__lambdaVectors]
        f1,f2 = zip(*solutions)
        
        plt.scatter(f1,f2,s=5,color='yellow')
        plt.show()
    
    def getBestParetoFront(self):
        if(self.__problem == "zdt3"):
            return getFront(path='PF.dat')
        elif(self.__problem == "cf6"):
            if(self.__individualDimension == 4):
                return getFront(path='cf6_4d_PF.dat')
            elif(self.__individualDimension == 16):
                return getFront(path='cf6_16d_PF.dat')
            else:
                raise Exception("getBestParetoFront")
        else:
            raise Exception("getBestParetoFront")
    
    def updateNeighborProblems(self, childrens):
        
        dicc1 = {i:tupla for i, tupla in enumerate(list(self.__lls_k_nearest_Neighbors.keys()))}
        dicc2 = {i:k_nearest_neighbors for i, k_nearest_neighbors in enumerate(list(self.__lls_k_nearest_Neighbors.values()))}
                    
        for idx in list(dicc1.keys()):
            
            lambdaVector = dicc1[idx]
            k_nearest_neighbors = dicc2[idx]
            
            children = childrens[lambdaVector]["individual"]
            f1_children = childrens[lambdaVector]["fitness"]["f1"]
            f2_children = childrens[lambdaVector]["fitness"]["f2"]
                            
            for (lambdaNeighbor, distance) in k_nearest_neighbors:

                Tchebycheff_children = self.Tchebycheff(input_lambdaVector=lambdaNeighbor, f1=f1_children, f2=f2_children)
                    
                currentIndividual = self.__subProblems[lambdaNeighbor]["individual"]
                
                f1_currentIndividual = self.__subProblems[lambdaNeighbor]["fitness"]["f1"]
                f2_currentIndividual = self.__subProblems[lambdaNeighbor]["fitness"]["f2"]
                
                Tchebycheff_currentIndividual = self.Tchebycheff(input_lambdaVector=lambdaNeighbor, f1=f1_currentIndividual, f2=f2_currentIndividual)
                                    
                if(Tchebycheff_children <= Tchebycheff_currentIndividual):
                    
                    self.__subProblems[lambdaNeighbor]["individual"] = children
                    self.__subProblems[lambdaNeighbor]["fitness"]["f1"] = f1_children
                    self.__subProblems[lambdaNeighbor]["fitness"]["f2"] = f2_children            
                    
        """ Non-hashable -> hashable """
        currentPopulation = set([tuple(dicc["individual"]) for lambdaVector, dicc in self.__subProblems.items()])
        self.__currentPopulation = [list(tupla) for tupla in currentPopulation]
        
    def Tchebycheff(self, input_lambdaVector, f1, f2):
        max_value = float("-inf")
        
        z1 = self.__z_current[0]
        lambda1 = input_lambdaVector[0]
        
        aggregation1 = lambda1*math.fabs(f1-z1)
        
        if(aggregation1 > max_value):
            max_value = aggregation1
        
        z2 = self.__z_current[1]
        lambda2 = input_lambdaVector[1]
        
        aggregation2 = lambda2*math.fabs(f2-z2)
        
        if(aggregation2 > max_value):
            max_value = aggregation2
        
        return max_value
        
    def compute_z_Reference(self):
        self.__z_current[0] = self.__best_f1_value
        self.__z_current[1] = self.__best_f2_value
    
    def association(self):
        for lambdaVector, individual in zip(self.__lambdaVectors, self.__currentPopulation):
            self.__subProblems[lambdaVector] = {"individual": individual}
    
    def compute_lls_k_nearest_Neighbors(self):
        dicc = dict()
        for lambdaVector in self.__lambdaVectors:
            ls = self.getNearestNeighbors(inputLambda=lambdaVector)
            dicc[lambdaVector] = ls
        return dicc
    
    def getNearestNeighbors(self, inputLambda):
        inputLambdaVectors = self.__lambdaVectors
        ls = self.getDistanceToEachLambda(inputLambda, inputLambdaVectors)
        orderedLs = sorted(ls, key=lambda tupla : tupla[1], reverse = False)
        return orderedLs[0:self.__numNearestNeighbors]
        
    def getDistanceToEachLambda(self, inputLambda, inputLambdaVectors):
        ls = list()
        for lambdaVector in inputLambdaVectors:
            d = math.sqrt(math.pow((lambdaVector[0]-inputLambda[0]), 2)+math.pow((lambdaVector[1]-inputLambda[1]), 2))
            ls.append((lambdaVector, d))
        return ls  
    
    def reproduction(self):
        lls_k_nearest_Neighbors = self.__lls_k_nearest_Neighbors
        children_to_subProblem = dict()

        for lambdaVector_subProblem, k_nearest_Neighbors in lls_k_nearest_Neighbors.items():
            currentIndividual = self.__subProblems[lambdaVector_subProblem]["individual"]
            random_lambdaVectors = random.sample(k_nearest_Neighbors, 3)
            children = self.crossover_diferential_evolution(currentIndividual, random_lambdaVectors)
            children_to_subProblem[lambdaVector_subProblem] = {"individual": children}
    
        return children_to_subProblem

    def crossover_diferential_evolution(self, currentIndividual, random_lambdaVectors):
        xR1 = random_lambdaVectors[0][0]
        xR2 = random_lambdaVectors[1][0]
        xR3 = random_lambdaVectors[2][0]
        
        individual_xR1 = self.__subProblems[xR1]["individual"]
        individual_xR2 = self.__subProblems[xR2]["individual"]
        individual_xR3 = self.__subProblems[xR3]["individual"]
    
        F = 0.5
        
        difference = [(x - y) for (x, y) in zip(individual_xR2, individual_xR3)]
        product = [(F*x) for x in difference]
        children = [(x + y) for (x, y) in zip(individual_xR1, product)]
        
        CR = 0.5
        
        for dimension in range(0, len(children)):
            
            rnd = random.random()
            rnd_dim = random.choice([i for i in range(0, len(children))])
        
            if((rnd <= CR) or (dimension - rnd_dim == 0)):
                pass
            elif((rnd > CR) or (math.fabs(dimension - rnd_dim) > 0)):
                children[dimension] = currentIndividual[dimension]
            
        if(self.__mutationPercentage > 0):
            self.mutation(input_children=children)
    
        if(self.__problem == "zdt3"):
            for dimension in range(0, len(children)):
                if(children[dimension] < self.__lowerLimit):
                    children[dimension] = self.__lowerLimit
                elif(children[dimension] > self.__upperLimit):
                    children[dimension] = self.__upperLimit
        elif(self.__problem == "cf6"):
            if(children[0] < self.__lowerLimit_x1):
                children[0] = self.__lowerLimit_x1
            elif(children[0] > self.__upperLimit_x1):
                children[0] = self.__upperLimit_x1
                
            for dimension in range(1, len(children)):
                if(children[dimension] < self.__lowerLimit_xn):
                    children[dimension] = self.__lowerLimit_xn
                elif(children[dimension] > self.__upperLimit_xn):
                    children[dimension] = self.__upperLimit_xn
        
        return children
    
    def mutation(self, input_children):
        
        a = math.floor(self.__mutationPercentage*self.__individualDimension)
        b = self.__individualDimension - a #Complementario
        
        booleanVector = b*[False] + a*[True]
        random.shuffle(booleanVector)
        
        SIG = 20
        
        for dimension in range(0, self.__individualDimension):
            b = random.choice(booleanVector)
            if(b):
                if(self.__problem == "zdt3"):
                    sd = (self.__upperLimit - self.__lowerLimit)/SIG
                elif(self.__problem == "cf6"):
                    if(dimension == 0):
                        sd = (self.__upperLimit_x1 - self.__lowerLimit_x1)/SIG
                    elif(dimension >= 1 and dimension <= self.__individualDimension - 1):
                        sd = (self.__upperLimit_xn - self.__lowerLimit_xn)/SIG
                input_children[dimension] = input_children[dimension] + random.gauss(0, sd)
    
    def initializeLambda(self):
                
        lambdaList = list()
            
        paso = 1/(self.__numIndividuals-1)
        
        e1 = 0.0
        e2 = 1.0
        
        for i in range(0, self.__numIndividuals):
            
            lambdaVector = (e1 + i*paso, e2 - i*paso)
            lambdaList.append(lambdaVector)
                
        return lambdaList
    
    def initializePopulation(self):
        population = list()
        for _ in range(0, self.__numIndividuals):
            individual = list()
            for dimension in range(0, self.__individualDimension):
                if(self.__problem == "zdt3"):
                    individual.append(random.uniform(self.__lowerLimit, self.__upperLimit))
                elif(self.__problem == "cf6"):
                    if(dimension == 0):
                        individual.append(random.uniform(self.__lowerLimit_x1, self.__upperLimit_x1))
                    elif(dimension >= 1 and dimension <= self.__individualDimension - 1):
                        individual.append(random.uniform(self.__lowerLimit_xn, self.__upperLimit_xn))
            population.append(individual)
        return population
    
    def compute_fitness(self, input_dicc):
        
        for lambdaVector, dicc in input_dicc.items():
            
            individual = dicc["individual"]
            
            (f1, f2) = self.individual_fitness_evaluation(individual)
            self.__numEvaluations = self.__numEvaluations - 1
            print(self.__numEvaluations)
            
            if(self.__best_f1_value is None):
                self.__best_f1_value = f1
            elif(f1 < self.__best_f1_value):
                self.__best_f1_value = f1
            else:
                pass
            
            if(self.__best_f2_value is None):
                self.__best_f2_value = f2
            elif(f2 < self.__best_f2_value):
                self.__best_f2_value = f2
            else:
                pass
                
            fitnessPerObjective = dict()
            
            fitnessPerObjective["f1"] = f1
            fitnessPerObjective["f2"] = f2
            
            input_dicc[lambdaVector]["fitness"] = fitnessPerObjective
        
    
    def individual_fitness_evaluation(self, individual):
        if(self.__problem == "zdt3"):
            f1_value = self.f1(individual)
            f2_value = self.f2(individual)
            return (f1_value, f2_value)
        elif(self.__problem == "cf6"):
            cf6_f1_value = self.cf6_f1(individual)
            cf6_f2_value = self.cf6_f2(individual)
            return (cf6_f1_value, cf6_f2_value)
        else:
            raise Exception("individual_fitness_evaluation")
    
    def f1(self, individual):
        return individual[0]
    
    def f2(self, individual):
        g_value = self.g(individual)
        return (g_value*self.h(self.f1(individual), g_value))
    
    def g(self, individual):
        suma = 0
        for idx in range(1, len(individual)):
            suma = suma + individual[idx]
        
        division = 9/(len(individual)-1)
        return (1 + (division)*suma)
    
    def h(self, f1, g):
        return (1 - math.sqrt(f1/g) - (f1/g)*math.sin(10*math.pi*f1))
    
    def cf6_f1(self, individual):
        J1 = self.cf6_j1_set()
        J2 = self.cf6_j2_set()
        suma = 0
        for j in J1:
            suma = suma + math.pow(self.cf6_y(j, J1, J2, individual), 2)
        f = individual[0] + suma
        
        tupla1 = self.restriction_1(individual)
        penalization1 = math.fabs(tupla1[0])
        w1 = tupla1[1] #1 o 0, peso como función de activacion de la aplicacion de la penalizacion
                
        tupla2 = self.restriction_2(individual)
        penalization2 = math.fabs(tupla2[0])
        w2 = tupla2[1] #1 o 0, peso como función de activacion de la aplicacion de la penalizacion
        
        f_penalized = f + w1*penalization1 + w2*penalization2
        
        return f_penalized
    
    def cf6_f2(self, individual):
        J1 = self.cf6_j1_set()
        J2 = self.cf6_j2_set()
        suma = 0
        for j in J2:
            suma = suma + math.pow(self.cf6_y(j, J1, J2, individual), 2)
        f = math.pow((1-individual[0]), 2) + suma
        
        tupla1 = self.restriction_1(individual)
        penalization1 = math.fabs(tupla1[0])
        w1 = tupla1[1] #1 o 0, peso como función de activacion de la aplicacion de la penalizacion
        
        tupla2 = self.restriction_2(individual)
        penalization2 = math.fabs(tupla2[0])
        w2 = tupla2[1] #1 o 0, peso como función de activacion de la aplicacion de la penalizacion
        
        f_penalized = f + w1*penalization1 + w2*penalization2
        
        return f_penalized
    
    def cf6_j1_set(self):
        res = list()
        n = self.__individualDimension
        for i in range(1, n+1):
            if((i%2 != 0) and (2<=i) and (i<=n)):
                res.append(i)
        return res
    
    def cf6_j2_set(self):
        res = list()
        n = self.__individualDimension
        for i in range(1, n+1):
            if((i%2 == 0) and (2<=i) and (i<=n)):
                res.append(i)
        return res
    
    def cf6_y(self, j, J1, J2, individual):
        if(j in J1):
            return individual[j-1] - 0.8*individual[0]*math.cos(6*math.pi*individual[0] + ((j*math.pi)/self.__individualDimension))
        elif(j in J2):
            return individual[j-1] - 0.8*individual[0]*math.sin(6*math.pi*individual[0] + ((j*math.pi)/self.__individualDimension))
        else:
            raise Exception("cf6_y bug")
    
    def restriction_1(self, individual):
        n = self.__individualDimension
        x = individual
        pi = math.pi
        g1 = x[1] - 0.8*x[0]*math.sin(6*pi*x[0] + (2*pi/n)) - self.sgn(0.5*(1-x[0])-math.pow((1-x[0]),2))*math.sqrt(math.fabs(0.5*(1-x[0])-math.pow((1-x[0]),2)))
        return (g1, g1 < 0)
    
    def restriction_2(self, individual):
        n = self.__individualDimension
        x = individual
        pi = math.pi
        g2 = x[3] - 0.8*x[0]*math.sin(6*pi*x[0] + (4*pi/n)) - self.sgn(0.25*math.sqrt(1-x[0])-0.5*(1-x[0]))*math.sqrt(math.fabs(0.25*math.sqrt(1-x[0])-0.5*(1-x[0])))
        return (g2, g2 < 0)
    
    def sgn(self, x):
        if(x>0):
            return 1
        elif(x == 0):
            return 0
        else:
            return -1    

def HV_final_each_seed(dir_path_1, dir_path_2):
    
    array1 = os.listdir(dir_path_1)
    array2 = os.listdir(dir_path_2)
    
    f = open('./HV.out',"w+")
    f.close()
    for (filePath1, filePath2) in zip(array1, array2):
        
        seed_filePath1 = filePath1.split("_")[-1].replace(".out","")
        seed_filePath2 = filePath2.split("_")[-1].replace("seed","").replace(".out","")
        print(seed_filePath1)
        print(seed_filePath2)
        
        if(seed_filePath1 != seed_filePath2):
            raise Exception("seed's must be equal")
        
        all_points = HV_all_points(filePath1=dir_path_1+filePath1, filePath2=dir_path_2+filePath2)
        
        f = open('HV.out',"a+")
        f.write(str(seed_filePath1)+"    "+str(all_points[0])+"    "+str(all_points[1])+"    "+str(all_points[2])+"    "+str(all_points[3])+"    "+str(all_points[4])+"    "+str(all_points[5])+"\n")
        f.close()

def HV_all_points(filePath1, filePath2):
    worst_f1_value_1stALG, worst_f2_value_1stALG = reference_hv_point(filePath=filePath1)
    worst_f1_value_2ndALG, worst_f2_value_2ndALG = reference_hv_point(filePath=filePath2)
    
    worst_f1 = max(worst_f1_value_1stALG, worst_f1_value_2ndALG)
    worst_f2 = max(worst_f2_value_1stALG, worst_f2_value_2ndALG)
    
    return [worst_f1_value_1stALG, worst_f1_value_2ndALG, worst_f1, worst_f2_value_1stALG, worst_f2_value_2ndALG, worst_f2]

def reference_hv_point(filePath):
    f = open(filePath, 'r')
    
    worst_f1_value = float('-inf')
    worst_f2_value = float('-inf')
    
    Lines = f.readlines()
    for line in Lines:
        strTupla = (line.strip()).split()
        
        f1 = float(strTupla[0])
        f2 = float(strTupla[1])
        
        if(f1 > worst_f1_value):
            worst_f1_value = f1
        else:
            pass
        
        if(f2 > worst_f2_value):
            worst_f2_value = f2
        else:
            pass
    
    return ([worst_f1_value, worst_f2_value])
    
def compute_executions(seed_init, seed_end, numIndividuals, numGenerations, individualDimension, neighborhoodPercentage, mutationPercentage, problem):
    for value_seed in range(seed_init, seed_end+1):
        random.seed(value_seed)
        
        path = './myP'+str(numIndividuals)+'G'+str(numGenerations)+'/'
        nameFile = "P_"+str(numIndividuals)+"_G_"+str(numGenerations)+"_DIM_"+str(individualDimension)+"_Prob.Neigh_"+str(neighborhoodPercentage)+"_Prob.Mutat_"+str(mutationPercentage)+"_seed_"+str(value_seed)+".out"
        absPathFile = path+nameFile
        
        f = open(absPathFile,"w+")
        f.close()
        
        ASC_Multiobjective_Aggregation_instance = ASC_Multiobjective_Aggregation(numGenerations, numIndividuals, individualDimension, neighborhoodPercentage, mutationPercentage, problem=problem, absPathFile=absPathFile)
        ASC_Multiobjective_Aggregation_instance.find_Pareto_Front()

def agg_vs_nsgaii_capture_results(my_dir, problem, dimension):
    
    dir = my_dir
    arr = os.listdir(dir)
    for path in arr:
        
        dicc = dict()
        data = path.replace("_", ",").split(",")
        for (i,j) in zip(data[0::2], data[1::2]):
            dicc[i] = j
                
        value_seed = int(dicc['seed'].replace(".out",""))
        random.seed(value_seed)
        numIndividuals = int(dicc['P'])
        numGenerations = int(dicc['G'])
        individualDimension = int(dicc['DIM'])
        neighborhoodPercentage = float(dicc['Prob.Neigh'])
        mutationPercentage = float(dicc['Prob.Mutat'])
        
        gen = numGenerations
        endLine = gen*numIndividuals - 1 # 9999
        startLine = endLine - numIndividuals + 1 # 9999 - 100 + 1 = 9960
        
        file = open(dir+path, 'r')
        Lines = file.readlines()
        agg_front = list()
        for line in Lines[startLine:endLine+1]:
            strTupla = (line.strip()).split()
            floatTupla = (float(strTupla[0]),float(strTupla[1]))
            agg_front.append(floatTupla)
        file.close()
        
        nsgaii_dir = './P'+str(numIndividuals)+'G'+str(numGenerations)+'/'
        try:
            file = open(nsgaii_dir+'allpopm_seed'+str(value_seed)+".out", 'r')
        except:
            file = open(nsgaii_dir+'all_popm_seed'+str(value_seed)+".out", 'r')
        Lines = file.readlines()
        nsgaii_front = list()
        for line in Lines[startLine:endLine+1]:
            strTupla = (line.strip()).split()
            floatTupla = (float(strTupla[0]),float(strTupla[1]))
            nsgaii_front.append(floatTupla)
        file.close()
        
        if(problem == "zdt3"):
            bestParetoFront = getFront(path="PF.dat")
        elif(problem == "cf6" and dimension == 4):
            bestParetoFront = getFront(path="cf6_4d_PF.dat")
        elif(problem == "cf6" and dimension == 16):
            bestParetoFront = getFront(path="cf6_16d_PF.dat")

        f1_ideal,f2_ideal = zip(*bestParetoFront)
        plt.scatter(f1_ideal,f2_ideal,s=1,color='black')

        f1_agg, f2_agg = zip(*agg_front)
        f1_nsgaii, f2_nsgaii = zip(*nsgaii_front)
        
        plt.scatter(f1_agg, f2_agg, s=1,color='green')
        plt.scatter(f1_nsgaii, f2_nsgaii, s=1,color='red')
        
        nameFile = "P_"+str(numIndividuals)+"_G_"+str(numGenerations)+"_DIM_"+str(individualDimension)+"_Prob.Neigh_"+str(neighborhoodPercentage)+"_Prob.Mutat_"+str(mutationPercentage)+"_seed_"+str(value_seed)
        fig = './results_'+nameFile+'.png'
        plt.savefig(fig)
        
        plt.clf()
        plt.cla()
        plt.close()

def get_IGD(my_dir, problem, dimension):
    
    if(problem == "zdt3"):
        pathBestParetoFront = "PF.dat"
    elif(problem == "cf6" and dimension == 4):
        pathBestParetoFront = "cf6_4d_PF.dat"
    elif(problem == "cf6" and dimension == 16):
        pathBestParetoFront = "cf6_16d_PF.dat"
    
    dir = my_dir
    arr = os.listdir(dir)
    for path in arr:
        
        dicc = dict()
        data = path.replace("_", ",").split(",")
        for (i,j) in zip(data[0::2], data[1::2]):
            dicc[i] = j
                
        value_seed = int(dicc['seed'].replace(".out",""))
        random.seed(value_seed)
        numIndividuals = int(dicc['P'])
        numGenerations = int(dicc['G'])
        individualDimension = int(dicc['DIM'])
        neighborhoodPercentage = float(dicc['Prob.Neigh'])
        mutationPercentage = float(dicc['Prob.Mutat'])
        
        f = open("./IGD_seed_"+str(value_seed)+".out",'a+')

        ls_IGD_myFront = list()
        ls_IGD_professorFront = list()
        
        """ Calculamos los IGD's para cada generación """
        for gen in range(1, numGenerations+1):
            
            endLine = gen*numIndividuals - 1 #99
            startLine = endLine - numIndividuals + 1 # 99 - 100 + 1 = -1 + 1 = 0
            
            file = open(dir+path, 'r')
            Lines = file.readlines()
            myFront = list()
            for line in Lines[startLine:endLine+1]:
                strTupla = (line.strip()).split()
                floatTupla = (float(strTupla[0]),float(strTupla[1]))
                myFront.append(floatTupla)
            file.close()
            
            IGD_value_myFront = IGD(myFront=myFront, pathBestFront=pathBestParetoFront)
            tupla1 = (gen, IGD_value_myFront)
            ls_IGD_myFront.append(tupla1)
            
            professor_dir = './P'+str(numIndividuals)+'G'+str(numGenerations)+'/'
            try:
                file = open(professor_dir+'allpopm_seed'+str(value_seed)+".out", 'r')
            except:
                file = open(professor_dir+'all_popm_seed'+str(value_seed)+".out", 'r')
            
            Lines = file.readlines()
            professorFront = list()
            for line in Lines[startLine:endLine+1]:
                strTupla = (line.strip()).split()
                floatTupla = (float(strTupla[0]),float(strTupla[1]))
                professorFront.append(floatTupla)
            file.close()
            
            IGD_value_professorFront = IGD(myFront=professorFront, pathBestFront=pathBestParetoFront)
            tupla2 = (gen, IGD_value_professorFront)
            ls_IGD_professorFront.append(tupla2)
            
            f.write(str(gen)+"    "+str(tupla1[1])+"    "+str(tupla2[1])+"\n")
        f.close()
            
        gen,IGD_myFront = zip(*ls_IGD_myFront)
        plt.scatter(gen,IGD_myFront,color='blue')
        
        gen,IGD_professorFront = zip(*ls_IGD_professorFront)
        plt.scatter(gen,IGD_professorFront,color='red')

        fig = './results_value_seed_'+str(value_seed)+'.png'
        plt.savefig(fig)
        
        plt.clf()
        plt.cla()
        plt.close()

def IGD(myFront, pathBestFront):
        
    def min_distance(tupla, myFront):
        f1_bestParetoFront = tupla[0]
        f2_bestParetoFront = tupla[1]
        minima_distancia = float('inf')
        for (f1, f2) in myFront:
            d = math.sqrt(math.pow((f1-f1_bestParetoFront), 2)+math.pow((f2-f2_bestParetoFront), 2))
            if(d<minima_distancia):
                minima_distancia = d
        return minima_distancia
    
    bestParetoFront = getFront(path=pathBestFront)
    suma = 0
    for tupla in bestParetoFront:
        suma = suma + min_distance(tupla, myFront)
    return suma/len(bestParetoFront)

def getFront(path):
    file = open(path, 'r')
    Lines = file.readlines()
    front = list()
    for line in Lines:
        strTupla = (line.strip()).split()
        floatTupla = (float(strTupla[0]),float(strTupla[1]))
        front.append(floatTupla)
    return front

if __name__ == '__main__':
    
    #compute_executions(seed_init=1, seed_end=10, numIndividuals=80, numGenerations=50, individualDimension=16, neighborhoodPercentage=0.15, mutationPercentage=2/16, problem="cf6")
    #agg_vs_nsgaii_capture_results(my_dir='./myP80G50/', problem="cf6", dimension=16)
    #HV_final_each_seed(dir_path_1="./myP80G50/", dir_path_2="./P80G50/")
    #get_IGD(my_dir='./myP100G100/', problem="zdt3", dimension=30)
    
    random.seed(1)
    
    numIndividuals=100
    numGenerations=100
    individualDimension=30
    neighborhoodPercentage=0.15
    mutationPercentage=1/individualDimension
    problem="zdt3"
    
    ASC_Multiobjective_Aggregation_instance = ASC_Multiobjective_Aggregation(numGenerations, numIndividuals, individualDimension, neighborhoodPercentage, mutationPercentage, problem=problem, absPathFile="")
    ASC_Multiobjective_Aggregation_instance.find_Pareto_Front()
    