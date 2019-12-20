#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABCMeta,abstractmethod
import numpy as np
from Datos import Datos
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
"""
Created on Mon Oct 14 19:04:27 2019

@author: giogiasvalley
"""
dataset = Datos('ConjuntosDatos/german.data')
v_simple = ShuffleSplit(len(dataset.datos), test_size=.25, random_state=0)
v_cruzada = 2

f = open("sk_particionado.txt","w")

#VSimple + laplace

atributos = preprocessing.OneHotEncoder(categorical_features=dataset.nominalAtributos[:-1],sparse=False)
X = atributos.fit_transform(dataset.datos[:,:-1])
Y = dataset.datos[:,-1]

clasificador = MultinomialNB(alpha=1, class_prior=None, fit_prior=True)
clasificador.fit(X,Y)

v_s = cross_val_score(clasificador, X, Y, cv=v_simple)

f.write("MSE de NB usando VSimple con laPlace: " + str((1-np.mean((v_s)))) + "con varianza: "+ str((np.var((v_s)))) + "\n")

#VCruzada + laplace
atributos = preprocessing.OneHotEncoder(categorical_features=dataset.nominalAtributos[:-1],sparse=False)
X = atributos.fit_transform(dataset.datos[:,:-1])
Y = dataset.datos[:,-1]

clasificador = MultinomialNB(alpha=1, class_prior=None, fit_prior=True)
clasificador.fit(X,Y)

v_c = cross_val_score(clasificador, X, Y, cv=v_cruzada)

f.write("MSE de NB usando VCruzada con laPlace: " + str((1-np.mean((v_c)))) + "con varianza: "+ str((np.var((v_c)))) + "\n")


#VSimple - laplace
atributos = preprocessing.OneHotEncoder(categorical_features=dataset.nominalAtributos[:-1],sparse=False)
X = atributos.fit_transform(dataset.datos[:,:-1])
Y = dataset.datos[:,-1]

clasificador = MultinomialNB(alpha=0, class_prior=None, fit_prior=True)
clasificador.fit(X,Y)

v_s = cross_val_score(clasificador, X, Y, cv=v_simple)

f.write("MSE de NB usando VSimple sin laPlace: " + str((1-np.mean((v_s)))) + "con varianza: "+ str((np.var((v_s)))) + "\n")

#VCruzada - laplace
atributos = preprocessing.OneHotEncoder(categorical_features=dataset.nominalAtributos[:-1],sparse=False)
X = atributos.fit_transform(dataset.datos[:,:-1])
Y = dataset.datos[:,-1]

clasificador = MultinomialNB(alpha=0, class_prior=None, fit_prior=True)
clasificador.fit(X,Y)

v_c = cross_val_score(clasificador, X, Y, cv=v_cruzada)

f.write("MSE de NB usando VCruzada sin laPlace: " + str((1-np.mean((v_c)))) + "con varianza: "+ str((np.var((v_c)))) + "\n")

f.close()