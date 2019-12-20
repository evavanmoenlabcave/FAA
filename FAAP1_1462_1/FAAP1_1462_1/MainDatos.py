# -*- coding: utf-8 -*-
"""

@author: profesores faa
"""
from abc import ABCMeta, abstractmethod
import random

from functools import reduce
import math
import operator

from Datos import Datos
from Clasificador import ClasificadorNaiveBayes
from EstrategiaParticionado import ValidacionCruzada,ValidacionSimple
import numpy as np

from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

dataset=Datos('ConjuntosDatos/german.data')
#dataset=Datos('ConjuntosDatos/tic-tac-toe.data')
print("----Pruebas P1")
print (dataset.nombreAtributos)
print (dataset.tipoAtributos)
print (dataset.nominalAtributos)
print (dataset.datos)
print("----Pruebas P2")
d1 = Datos('ConjuntosDatos/german.data')
d2 = Datos('ConjuntosDatos/tic-tac-toe.data')

NB = ClasificadorNaiveBayes()


print ("VALIDACION SIMPLE (german.data): ")
v_simple = ValidacionSimple('ValidacionSimple',3,0.8)
p = v_simple.creaParticiones(d1.datos)
for k in p:
	print ("Indices Test: ")
	print (k.indicesTest)
	print ("Indices Train: ")
	print (k.indicesTrain)
	print ("----")
print (v_simple.nombreEstrategia)

print ("-----------------------")
print ("VALIDACION CRUZADA (german.data): ")
v_cross = ValidacionCruzada('ValidacionCruzada',5)
p = v_cross.creaParticiones(d1.datos)
for k in p:
	print ("Indices Test: ")
	print (k.indicesTest)
	print ("Indices Train: ")
	print (k.indicesTrain)
	print ("----")
print (v_cross.nombreEstrategia)

print ("-----------------------")
errorS, p = NB.validacion(v_simple,d1,NB,seed=None,laPlace=True)

print("\nError NB simple: ")
print(errorS[0])

errorC, p = NB.validacion(v_cross,d1,NB,seed=None,laPlace=True)
print("\nError NB cruzada: ")
print(errorC)
print("\nMedia errores NB cruzada: ")
print(np.mean(errorC))


