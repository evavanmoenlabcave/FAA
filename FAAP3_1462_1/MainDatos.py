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
import Clasificador as c
from EstrategiaParticionado import ValidacionCruzada,ValidacionSimple
import numpy as np

from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


#dataset=Datos('ConjuntosDatos/tic-tac-toe.data')


d_ex1 = Datos('online_shoppers.data', normalizar=True)
#k=50
estrategia = ValidacionSimple(0.7)
c8 = c.ClasificadorVecinosProximos(5, weights='distance')
error8 = c8.validacion(estrategia, d_ex1, c8, 45)
print(error8)
#c = c.ClasificadorVecinosProximos(k, normalizar=False, weights='distance')
#error1 = c.validacion(estrategia, d_ex1, c, seed=45)
#plt.show()
#print("Fichero wdbc.data: ")
#print("Media: ")
#print(np.mean(error1))
#print("Desviacion tipica: ")
#print(np.std(error1))

k=50
#particion.creaParticiones(d_ex1.datos, 45)
#train = particion.particiones[-1].indicesTrain
#test = particion.particiones[-1].indicesTest
#datostrain = d_ex1.extraeDatos(train)
#datostest = d_ex1.extraeDatos(test)

##WDBC
#d_ex1 = Datos('wdbc.data')
#d_ex2 = Datos('online_shoppers.data')
#estrategia = ValidacionSimple(0.6)
#
#
#c13_online = c.ClasificadorNaiveBayes()
#error13 = c13_online.validacionRoc(estrategia, d_ex2, c13_online, 'Naive Bayes', seed=45)
#c14_online = c.ClasificadorNaiveBayes(correccionLaplace=True)
#error14 = c14_online.validacionRoc(estrategia, d_ex2, c14_online, 'Naive Bayes Laplace', seed=45)
#plt.show()
#
##c1 = c.ClasificadorVecinosProximos(10, normalizar=True, weights='uniform')
##error1 = c1.validacion(estrategia, d_ex1, c1, seed=45)
##print(error1)
##c2 = c.ClasificadorVecinosProximos(3, normalizar=True, weights='distance')
##error2 = c1.validacionRoc(estrategia, d_ex1, c2, 'knn_k=3', 45)
#"""
#c3 = c.ClasificadorVecinosProximos(5, normalizar=True, weights='distance')
#error3 = c3.validacionRoc(estrategia, d_ex1, c3, 'knn_k=5', 45)
#c4 = c.ClasificadorVecinosProximos(11, normalizar=True, weights='distance')
#error4 = c4.validacionRoc(estrategia, d_ex1, c4, 'knn_k=11', 45)
#c5 = c.ClasificadorVecinosProximos(21, normalizar=True, weights='distance')
#error5 = c5.validacionRoc(estrategia, d_ex1, c5, 'knn_k=21', 45)
#c6 = c.ClasificadorVecinosProximos(51, normalizar=True, weights='distance')
#error6 = c6.validacionRoc(estrategia, d_ex1, c6, 'knn_k=51', 45)
#c7 = c.ClasificadorRegresionLogistica(nEpocas=100, constante=1)
#error7 = c7.validacionRoc(estrategia, d_ex1, c7, 'regresion nEp=100, c=1', 45)
#c8 = c.ClasificadorRegresionLogistica(nEpocas=1000, constante=1)
#error8 = c8.validacionRoc(estrategia, d_ex1, c8, 'regresion nEp=1000, c=1', 45)
#c9 = c.ClasificadorRegresionLogistica(nEpocas=100, constante=0.1)
#error9 = c9.validacionRoc(estrategia, d_ex1, c9, 'regresion nEp=100, c=0.1', 45)
#c10 = c.ClasificadorRegresionLogistica(nEpocas=1000, constante=0.1)
#error10 = c10.validacionRoc(estrategia, d_ex1, c10, 'regresion nEp=1000, c=0.1', 45)
#c11 = c.ClasificadorRegresionLogistica(nEpocas=100, constante=0.01)
#error11 = c11.validacionRoc(estrategia, d_ex1, c11, 'regresion nEp=100, c=0.01', 45)
#c12 = c.ClasificadorRegresionLogistica(nEpocas=1000, constante=0.01)
#error12 = c12.validacionRoc(estrategia, d_ex1, c12, 'regresion nEp=1000, c=0.01', 45)
#c13 = c.ClasificadorNaiveBayes()
#error13 = c13.validacionRoc(estrategia, d_ex1, c13, 'Naive Bayes', 45)
#c14 = c.ClasificadorNaiveBayes(correccionLaplace=True)
#error14 = c14.validacionRoc(estrategia, d_ex1, c14, 'Naive Bayes Laplace', 45)
#"""
#plt.show()
#"""
###WDBC
#d_ex1 = Datos('online_shoppers.data')
#estrategia = ValidacionSimple(0.6)
#c1 = c.ClasificadorVecinosProximos(1, normalizar=True, weights='distance')
#error1 = c1.validacionRoc(estrategia, d_ex1, c1, 'knn_k=1', 45)
#c2 = c.ClasificadorVecinosProximos(3, normalizar=True, weights='distance')
#error2 = c1.validacionRoc(estrategia, d_ex1, c2, 'knn_k=3', 45)
#c3 = c.ClasificadorVecinosProximos(5, normalizar=True, weights='distance')
#error3 = c3.validacionRoc(estrategia, d_ex1, c3, 'knn_k=5', 45)
#c4 = c.ClasificadorVecinosProximos(11, normalizar=True, weights='distance')
#error4 = c4.validacionRoc(estrategia, d_ex1, c4, 'knn_k=11', 45)
#c5 = c.ClasificadorVecinosProximos(21, normalizar=True, weights='distance')
#error5 = c5.validacionRoc(estrategia, d_ex1, c5, 'knn_k=21', 45)
#c6 = c.ClasificadorVecinosProximos(51, normalizar=True, weights='distance')
#error6 = c6.validacionRoc(estrategia, d_ex1, c6, 'knn_k=51', 45)
#c7 = c.ClasificadorRegresionLogistica(nEpocas=100, constante=1)
#error7 = c7.validacionRoc(estrategia, d_ex1, c7, 'regresion nEp=100, c=1', 45, oneHot=True)
#c8 = c.ClasificadorRegresionLogistica(nEpocas=1000, constante=1)
#error8 = c8.validacionRoc(estrategia, d_ex1, c8, 'regresion nEp=1000, c=1', 45, oneHot=True)
#c9 = c.ClasificadorRegresionLogistica(nEpocas=100, constante=0.1)
#error9 = c9.validacionRoc(estrategia, d_ex1, c9, 'regresion nEp=100, c=0.1', 45, oneHot=True)
#c10 = c.ClasificadorRegresionLogistica(nEpocas=1000, constante=0.1)
#error10 = c10.validacionRoc(estrategia, d_ex1, c10, 'regresion nEp=1000, c=0.1', 45, oneHot=True)
#c11 = c.ClasificadorRegresionLogistica(nEpocas=100, constante=0.01)
#error11 = c11.validacionRoc(estrategia, d_ex1, c11, 'regresion nEp=100, c=0.01', 45, oneHot=True)
#c12 = c.ClasificadorRegresionLogistica(nEpocas=1000, constante=0.01)
#error12 = c12.validacionRoc(estrategia, d_ex1, c12, 'regresion nEp=1000, c=0.01', 45, oneHot=True)
#c13 = c.ClasificadorNaiveBayes()
#error13 = c13.validacionRoc(estrategia, d_ex1, c13, 'Naive Bayes', 45, oneHot=True)
#c14 = c.ClasificadorNaiveBayes(correccionLaplace=True)
#error14 = c14.validacionRoc(estrategia, d_ex1, c14, 'Naive Bayes Laplace', 45, oneHot=True)
#plt.show()
#"""
#
#
#
##c.entrenamiento(datostrain,d_ex1.nominalAtributos, d_ex1.diccionarios)
##pred = c.clasifica(datostest, d_ex1.nominalAtributos, d_ex1.diccionarios)
##print(pred)
##error1 = c.validacion(estrategia, d_ex1, c, 45)
##print("Fichero wdbc.data: ")
##print("Media: ")
##print(error1)
##print(np.mean(error1))
##print("Desviacion tipica: ")
##print(np.std(error1))
#
##
##
##particion = ValidacionSimple(0.6)
##particion.creaParticiones(d3.datos, 45)
##
##train = particion.particiones[-1].indicesTrain
##test = particion.particiones[-1].indicesTest
##datostrain = d3.extraeDatos(train)
##datostest = d3.extraeDatos(test)
##
##
###d3.normal(d3.extraeDatos(e.particiones[0].indicesTrain))
###d3.normalizar()
##
##print('\n')
##
##knn = c.ClasificadorVecinosProximos(1, weights='uniform', normalizar = True)
###knn.entrenamiento(datostrain,d3.nominalAtributos, d3.diccionarios)
###pred = knn.clasifica(datostest, d3.nominalAtributos, d3.diccionarios)
##error = knn.validacion(particion, d3, knn)
### knn = c.ClasificadorRegresionLogistica()
### knn.entrenamiento(datostrain,d3.nominalAtributos, d3.diccionarios)
### pred = knn.clasifica(datostest, d3.nominalAtributos, d3.diccionarios)
###error = knn.validacion(particion, d3, knn)
###print(d3.diccionarios)
##print("error")
##print (error)
###print(pred)
###print(len(pred))
##
#
#
#
